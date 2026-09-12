"""Train a compact exact-action reranker from observable physical features."""
from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.exact_online_action_oracle import CompleteOnlineAction
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from experiments.md_exact_action_candidate_diagnostic import C0
from experiments.md_minimal_action_features import FEATURE_DIM, FEATURE_SCHEMA, build_minimal_action_features
from imitation_learning.md_residual_train import load_legacy_c0_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder, MaskedGreedyDecoder, apply_decoder_result, simulator_hard_mask
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider, ScoreOutput
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator

DATA = Path("reports/md_exact_action_candidate_diagnostic_2026-09-06")
OUT = Path("reports/md_minimal_physics_scorer_2026-09-08")
MODEL_SEEDS = (3101, 3102, 3103)
TRAINING_SEEDS = {3101: 5101, 3102: 5102, 3103: 5103}
INPUT_DIM = FEATURE_DIM + 1


class PhysicsSetScorer(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        return self.head(self.edge(edges).sum(dim=0)).squeeze(-1)


def _key(value: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(pair[0]), int(pair[1])) for pair in value))


def _dispatch(decoder: Any, scorer: Callable[[MDDiscreteSimulator], Any]) -> Callable[[MDDiscreteSimulator], None]:
    def run(simulator: MDDiscreteSimulator) -> None:
        mask = simulator_hard_mask(simulator)
        if not bool(mask.any()):
            return
        output = scorer(simulator)
        scores = output.scores if isinstance(output, ScoreOutput) else output
        result = decoder.decode(scores, simulator, hard_feasibility_mask=mask)
        if result.assignments:
            apply_decoder_result(simulator, result)
    return run


def _capture(seed: int, method: str, times: set[int], dispatch: Callable[[MDDiscreteSimulator], None]) -> dict[int, MDDiscreteSimulator]:
    simulator = MDDiscreteSimulator(generate_md_instance(MDGeneratorConfig(seed=seed)).domain)
    captured: dict[int, MDDiscreteSimulator] = {}
    while not simulator.done and simulator.time <= max(times):
        if simulator.time in times:
            captured[simulator.time] = copy.deepcopy(simulator)
        dispatch(simulator)
        if not simulator.has_advancing_work:
            break
        simulator.step()
    missing = times - set(captured)
    if missing:
        raise RuntimeError(f"failed to replay {seed=} {method=}, missing {sorted(missing)}")
    return captured


def _build_feature_rows() -> list[dict[str, Any]]:
    cache = OUT / "feature_package.jsonl"
    if cache.exists():
        cached = [json.loads(line) for line in cache.read_text().splitlines()]
        if cached and all(row.get("schema") == FEATURE_SCHEMA for row in cached):
            return cached
    labels = {}
    for line in (DATA / "exact_action_package.jsonl").read_text().splitlines():
        row = json.loads(line)
        labels[(row["snapshot_id"], _key(row["complete_first_action"]["assignments"]))] = row
    candidates = [json.loads(line) for line in (DATA / "candidate_records.jsonl").read_text().splitlines()]
    targets: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in candidates:
        targets[(int(row["seed"]), str(row["rollout_method"]))].add(int(row["current_time"]))
    models = {seed: load_legacy_c0_checkpoint(path, device="cpu")[0] for seed, path in C0.items()}
    providers = {seed: OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device="cpu") for seed, model in models.items()}
    snapshots: dict[tuple[int, str, int], MDDiscreteSimulator] = {}
    for (seed, method), times in sorted(targets.items()):
        if method == "masked_greedy":
            dispatch = _dispatch(MaskedGreedyDecoder(), lambda sim: torch.zeros_like(simulator_hard_mask(sim), dtype=torch.float32))
        else:
            dispatch = _dispatch(LearnedConstrainedDecoder(), providers[int(method.removeprefix("legacy_c0_seed"))])
        for current_time, simulator in _capture(seed, method, times, dispatch).items():
            snapshots[(seed, method, current_time)] = simulator
    rows = []
    for row in candidates:
        simulator = snapshots[(int(row["seed"]), str(row["rollout_method"]), int(row["current_time"]))]
        actions = []
        count = len(row["candidate_action_keys"])
        for rank, value in enumerate(row["candidate_action_keys"]):
            action_key = _key(value); label = labels[(row["snapshot_id"], action_key)]
            features = build_minimal_action_features(simulator, CompleteOnlineAction(action_key, ())).tolist()
            actions.append({"rank": rank, "rank_fraction": rank / max(1, count - 1), "features": features,
                            "regret": float(label["regret"]), "tolerance_optimal": bool(label["tolerance_optimal"])})
        rows.append({"schema": FEATURE_SCHEMA, "snapshot_id": row["snapshot_id"], "split": row["split"],
                     "model_seed": int(row["model_seed"]), "pending_count": int(row["pending_count"]), "actions": actions})
    cache.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows


def _normalization(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack([torch.tensor(edge) for row in rows if row["split"] == "train" for action in row["actions"] for edge in action["features"]])
    return values.mean(0), values.std(0).clamp_min(1.0)


def _states(rows: Sequence[Mapping[str, Any]], seed: int, mean: torch.Tensor, std: torch.Tensor, split: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["split"] != split or int(row["model_seed"]) != seed:
            continue
        actions = []
        for action in row["actions"]:
            physical = (torch.tensor(action["features"], dtype=torch.float32) - mean) / std
            rank = torch.full((physical.shape[0], 1), float(action["rank_fraction"]))
            actions.append({"features": torch.cat((physical, rank), 1), "rank": int(action["rank"]),
                            "regret": float(action["regret"]), "optimal": bool(action["tolerance_optimal"])})
        result.append({"pending_count": int(row["pending_count"]), "actions": actions})
    return result


def evaluate(model: nn.Module | None, states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regrets = []; optimal = []; by_pending: dict[int, list[float]] = defaultdict(list)
    with torch.no_grad():
        for state in states:
            if model is None:
                chosen = min(state["actions"], key=lambda action: action["rank"])
            else:
                scores = torch.stack([model(action["features"]) for action in state["actions"]])
                chosen = state["actions"][int(torch.argmin(scores))]
            regrets.append(chosen["regret"]); optimal.append(float(chosen["optimal"])); by_pending[state["pending_count"]].append(float(chosen["optimal"]))
    return {"state_count": len(states), "tolerance_optimal_top1": float(np.mean(optimal)), "mean_regret": float(np.mean(regrets)),
            "tolerance_optimal_top1_by_pending": {str(k): float(np.mean(v)) for k, v in sorted(by_pending.items())}}


def _train_one(model_seed: int, rows: Sequence[Mapping[str, Any]], mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    seed = TRAINING_SEEDS[model_seed]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_states = _states(rows, model_seed, mean, std, "train"); development = _states(rows, model_seed, mean, std, "development")
    model = PhysicsSetScorer(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
    best_state = None; best = None; stale = 0; history = []
    for epoch in range(1, 61):
        order = list(range(len(train_states))); random.Random(seed + epoch).shuffle(order); model.train(); losses = []
        for index in order:
            actions = train_states[index]["actions"]; outputs = torch.stack([model(action["features"]) for action in actions])
            regrets = torch.tensor([action["regret"] for action in actions]); tie = torch.tensor([action["optimal"] for action in actions], dtype=torch.float32); target = tie / tie.sum().clamp_min(1.0)
            loss = -(target * F.log_softmax(-outputs, 0)).sum() + 0.2 * F.smooth_l1_loss(outputs, regrets / regrets.max().clamp_min(1.0))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if epoch % 5 == 0:
            model.eval(); metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), **evaluate(model, development)}; history.append(metrics)
            key = (metrics["tolerance_optimal_top1"], -metrics["mean_regret"], -epoch)
            if best is None or key > (best["tolerance_optimal_top1"], -best["mean_regret"], -best["epoch"]):
                best = metrics; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else: stale += 1
            if stale >= 3: break
    assert best is not None and best_state is not None
    destination = OUT / f"seed{model_seed}"; destination.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "minimal-physics-scorer-1.0", "model_seed": model_seed, "state_dict": best_state, "normalization_mean": mean, "normalization_std": std, "best_development": best}, destination / "best_checkpoint.pt")
    return {"model_seed": model_seed, "baseline_development": evaluate(None, development), "best_development": best, "history": history}


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "minimal-physics-scorer-experiment-1.0", "created_before_training": True, "feature_schema": FEATURE_SCHEMA,
                "inputs": ["16_observable_physics_features", "residual_candidate_rank"], "hidden_dim": 16, "epochs_max": 60,
                "evaluation_interval": 5, "early_stopping_patience_evaluations": 3, "weight_decay": 0.001,
                "model_seeds": list(MODEL_SEEDS), "training_seeds": TRAINING_SEEDS, "train_seeds": [75000, 75599], "development_seeds": [75600, 75749],
                "forbidden_seed_ranges": [[75800, 75949], [76000, 76149]], "acceptance": {"each_seed_not_below_baseline": True, "mean_top1_gain_at_least": 0.02, "mean_regret_strictly_lower": True},
                "oracle_values_used_as_inputs": False, "production_decoder_modified": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter(); rows = _build_feature_rows(); feature_seconds = time.perf_counter() - started; mean, std = _normalization(rows)
    results = [_train_one(seed, rows, mean, std) for seed in MODEL_SEEDS]
    gains = [r["best_development"]["tolerance_optimal_top1"] - r["baseline_development"]["tolerance_optimal_top1"] for r in results]
    baseline_regret = float(np.mean([r["baseline_development"]["mean_regret"] for r in results])); model_regret = float(np.mean([r["best_development"]["mean_regret"] for r in results]))
    accepted = all(g >= 0 for g in gains) and float(np.mean(gains)) >= 0.02 and model_regret < baseline_regret
    summary = {"schema": protocol["schema"], "status": "state_feature_scorer_supported_for_fresh_validation" if accepted else "state_feature_scorer_not_supported", "feature_build_seconds": feature_seconds,
               "feature_record_count": len(rows), "results": results, "mean_top1_gain": float(np.mean(gains)), "baseline_mean_regret": baseline_regret, "model_mean_regret": model_regret,
               "accepted": accepted, "confirmation_read": False, "regression_benchmark_read": False, "oracle_values_used_as_inputs": False, "production_decoder_modified": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Minimal Physics Scorer", "", f"Status: **{summary['status']}**.", "", "| Seed | Baseline top-1 | Physics top-1 | Baseline regret | Physics regret |", "|---:|---:|---:|---:|---:|"]
    for result in results:
        b, p = result["baseline_development"], result["best_development"]
        lines.append(f"| {result['model_seed']} | {b['tolerance_optimal_top1']:.4f} | {p['tolerance_optimal_top1']:.4f} | {b['mean_regret']:.4f} | {p['mean_regret']:.4f} |")
    lines += ["", f"Mean top-1 gain: {summary['mean_top1_gain']:.4f}; mean regret: {baseline_regret:.4f} -> {model_regret:.4f}.", "", "Development diagnostic only. Confirmation and regression benchmark seeds were not read."]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
