"""Small exact-label exploration of early transport/mixed stationary states."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines.exact_online_action_oracle import enumerate_complete_online_actions, solve_exact_action_continuation
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_generation import SnapshotCandidate, residual_state
from experiments.md_exact_action_candidate_diagnostic import MODEL_SEEDS, RESIDUAL, generate_candidates
from experiments.md_minimal_action_features import FEATURE_SCHEMA, build_minimal_action_features
from experiments.md_minimal_physics_scorer import OUT as SCORER_DIR
from experiments.md_minimal_physics_scorer import PhysicsSetScorer
from imitation_learning.md_residual_train import load_residual_tail_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider
from simulation_environment.domain_model import TransportTask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


SEEDS = (76201, 76202, 76206)
OUTPUT_DIR = Path("reports/md_rich_exact_action_exploration_2026-09-08")


def _action_type(action: Any, tasks: dict[int, Any]) -> str:
    kinds = {"transport" if isinstance(tasks[task_id], TransportTask) else "process" for _, task_id in action.assignments}
    return "mixed" if len(kinds) > 1 else next(iter(kinds))


def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "rich-exact-action-exploration-1.0",
        "seeds": list(SEEDS),
        "state": "initial_stationary",
        "selection": "first three exploration seeds with 32-96 complete actions",
        "solver": {"threads": 1, "time_limit_seconds": 10, "require_optimal": True},
        "feature_schema": FEATURE_SCHEMA,
        "confirmation_read": False,
        "regression_benchmark_read": False,
        "training_performed": False,
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    residual_models = {seed: load_residual_tail_checkpoint(path, device="cpu")[0] for seed, path in RESIDUAL.items()}
    providers = {seed: OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device="cpu") for seed, model in residual_models.items()}
    rerankers = {}
    for seed in MODEL_SEEDS:
        payload = torch.load(SCORER_DIR / f"seed{seed}" / "best_checkpoint.pt", map_location="cpu", weights_only=True)
        model = PhysicsSetScorer(); model.load_state_dict(payload["state_dict"]); model.eval()
        rerankers[seed] = (model, payload["normalization_mean"], payload["normalization_std"])
    rows = []
    failures = []
    for seed in SEEDS:
        domain = generate_md_instance(MDGeneratorConfig(seed=seed)).domain
        simulator = MDDiscreteSimulator(domain)
        candidate = SnapshotCandidate(f"md-rich-{seed}", seed, "initial", domain, simulator, {})
        state = residual_state(candidate); legal = enumerate_complete_online_actions(domain, state)
        regrets = {}; types = {}; solve_seconds = 0.0
        costs = {}
        for action in legal:
            started = time.perf_counter()
            result = solve_exact_action_continuation(domain, state, action, time_limit_seconds=10, threads=1)
            solve_seconds += time.perf_counter() - started
            if result.status.value != "optimal" or result.continuation_cost is None or result.audit is None:
                failures.append({"seed": seed, "action": action.key, "status": result.status.value, "message": result.message})
                continue
            costs[action.key] = result.continuation_cost; types[action.key] = _action_type(action, {task.task_id: task for task in domain.tasks})
        if len(costs) != len(legal):
            continue
        minimum = min(costs.values()); regrets = {key: value - minimum for key, value in costs.items()}
        for model_seed in MODEL_SEEDS:
            generated = generate_candidates(candidate, legal, providers[model_seed](simulator).scores)
            optimal = {key for key, regret in regrets.items() if regret <= 1}
            model, norm_mean, norm_std = rerankers[model_seed]
            scored = []
            for rank, action in enumerate(generated):
                physical = (build_minimal_action_features(simulator, action) - norm_mean) / norm_std
                rank_column = torch.full((physical.shape[0], 1), rank / max(1, len(generated) - 1))
                with torch.no_grad():
                    score = float(model(torch.cat((physical, rank_column), dim=1)))
                scored.append((score, action))
            reranked = [action for _, action in sorted(scored, key=lambda item: (item[0], item[1].key))]
            baseline = generated[0]; selected = reranked[0]
            rows.append({
                "seed": seed, "model_seed": model_seed, "pending_count": len(state.pending_task_ids),
                "complete_action_count": len(legal), "candidate_count": len(generated),
                "candidate_recall_at_8": bool(optimal.intersection(action.key for action in generated[:8])),
                "candidate_recall_all": bool(optimal.intersection(action.key for action in generated)),
                "baseline_regret": regrets[baseline.key], "physics_regret": regrets[selected.key],
                "baseline_tolerance_optimal": regrets[baseline.key] <= 1, "physics_tolerance_optimal": regrets[selected.key] <= 1,
                "baseline_action_type": types[baseline.key], "physics_action_type": types[selected.key],
                "exact_solve_seconds_for_state": solve_seconds,
            })
    def metric_mean(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows) if rows else 0.0
    summary = {
        "schema": protocol["schema"], "valid": not failures and len(rows) == len(SEEDS) * len(MODEL_SEEDS),
        "state_count": len(SEEDS), "record_count": len(rows), "failure_count": len(failures),
        "mean_complete_action_count": metric_mean("complete_action_count"), "mean_candidate_count": metric_mean("candidate_count"),
        "candidate_recall_at_8": metric_mean("candidate_recall_at_8"), "candidate_recall_all": metric_mean("candidate_recall_all"),
        "baseline_tolerance_top1": metric_mean("baseline_tolerance_optimal"), "physics_tolerance_top1": metric_mean("physics_tolerance_optimal"),
        "baseline_mean_regret": metric_mean("baseline_regret"), "physics_mean_regret": metric_mean("physics_regret"),
        "records": rows, "confirmation_read": False, "regression_benchmark_read": False,
    }
    (OUTPUT_DIR / "failures.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
