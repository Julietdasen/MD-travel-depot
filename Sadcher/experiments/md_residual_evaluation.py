"""Paired final-rollout evaluation for residual IL checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from baselines.gurobi_md_oracle import (
    GurobiOracleStatus,
    replay_oracle_actions,
    solve_gurobi_md_oracle,
)
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_dataset import FORMAL_RESIDUAL_SPLIT_PLAN
from data_generation.md_residual_generation import (
    ResidualExclusion,
    SnapshotCandidate,
    eligible_snapshot,
    label_snapshot,
)
from data_generation.md_residual_pipeline import encode_training_features
from experiments.protocol import DatasetSplit, FailureReason
from imitation_learning.md_residual_train import (
    load_legacy_c0_checkpoint,
    load_residual_tail_checkpoint,
)
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import (
    LearnedConstrainedDecoder,
    MaskedGreedyDecoder,
    apply_decoder_result,
    simulator_hard_mask,
)
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider, ScoreOutput
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


def _latest_task_completion(simulator: MDDiscreteSimulator) -> float | None:
    values = [
        float(record["completed_at"])
        for record in (
            *simulator.process_execution_records,
            *simulator.transport_execution_records,
        )
        if record.get("completed_at") is not None
    ]
    return max(values) if values else None


def _run_online(
    domain,
    *,
    instance_id: str,
    instance_seed: int,
    method: str,
    model_seed: int | None,
    scorer,
    decoder,
    regret_time_limit: float,
    max_steps: int,
) -> dict[str, object]:
    simulator = MDDiscreteSimulator(domain)
    events = []
    regrets = []
    inference = 0.0
    wall_started = time.perf_counter()
    failure = None
    steps = 0
    while not simulator.done and steps < max_steps:
        hard_mask = simulator_hard_mask(simulator)
        if bool(torch.any(hard_mask)):
            started = time.perf_counter()
            output = scorer(simulator)
            scores = output.scores if isinstance(output, ScoreOutput) else output
            decoded = decoder.decode(
                scores,
                simulator,
                hard_feasibility_mask=hard_mask,
            )
            inference += time.perf_counter() - started
            assignments = tuple(decoded.assignments)
            if assignments and eligible_snapshot(simulator):
                candidate = SnapshotCandidate(
                    instance_id,
                    instance_seed,
                    method,
                    domain,
                    simulator,
                    encode_training_features(simulator),
                )
                labelled = label_snapshot(
                    candidate,
                    split_plan=FORMAL_RESIDUAL_SPLIT_PLAN,
                    time_limit_seconds=regret_time_limit,
                    threads=1,
                )
                if isinstance(labelled, ResidualExclusion):
                    regrets.append({
                        "decision_time": simulator.time,
                        "regret": None,
                        "pending_task_ids": [],
                        "best_total_cost": None,
                        "reason": labelled.reason,
                    })
                else:
                    chosen = next(
                        (
                            label for label in labelled.legal_batches
                            if label.batch.assignments == tuple(sorted(assignments))
                        ),
                        None,
                    )
                    regrets.append({
                        "decision_time": simulator.time,
                        "pending_task_ids": list(labelled.state.pending_task_ids),
                        "best_total_cost": min(
                            label.remaining_task_horizon + label.terminal_return_tail
                            for label in labelled.legal_batches
                        ),
                        "regret": None if chosen is None else chosen.regret,
                        "reason": None if chosen is not None else "joint_batch_not_enumerated",
                    })
            if assignments:
                apply_decoder_result(simulator, decoded)
            elif not simulator.has_advancing_work:
                failure = FailureReason.SCHEDULER_FAILURE
                break
            events.append({
                "decision_time": simulator.time,
                "assignments": [list(pair) for pair in assignments],
                "repaired": decoded.repaired,
            })
        if simulator.done:
            break
        if not simulator.has_advancing_work and not bool(torch.any(hard_mask)):
            failure = FailureReason.DEADLOCK
            break
        simulator.step()
        steps += 1
    if not simulator.done and failure is None:
        failure = FailureReason.TIMEOUT
    experiment = simulator.build_experiment_result(
        run_id=f"residual-eval-{method}-{instance_id}",
        method=method,
        instance_id=instance_id,
        seed=instance_seed,
        split=DatasetSplit.TEST,
        failure_reason=failure,
        inference_time_seconds=inference,
        wall_time_seconds=time.perf_counter() - wall_started,
        illegal_assignment_count=0,
    )
    latest = _latest_task_completion(simulator)
    numeric_regrets = [
        float(row["regret"]) for row in regrets if row["regret"] is not None
    ]
    return {
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "method": method,
        "model_seed": model_seed,
        "success": experiment.success,
        "illegal_assignment_count": experiment.illegal_assignment_count,
        "latest_task_completion": latest,
        "terminal_return_tail": (
            None if experiment.makespan is None or latest is None
            else experiment.makespan - latest
        ),
        "final_makespan": experiment.makespan,
        "inference_time_seconds": inference,
        "wall_runtime_seconds": experiment.wall_time_seconds,
        "decision_regret": (
            None if not numeric_regrets else statistics.mean(numeric_regrets)
        ),
        "state_regret": None,
        "regret_records": regrets,
        "event_sequence": events,
        "execution_records": {
            "process": list(simulator.process_execution_records),
            "transport": list(simulator.transport_execution_records),
        },
    }


def _run_milp(domain, *, instance_id: str, instance_seed: int, time_limit: float, max_steps: int) -> dict[str, object]:
    started = time.perf_counter()
    result = solve_gurobi_md_oracle(
        domain,
        time_limit_seconds=time_limit,
        threads=1,
    )
    if result.status not in (GurobiOracleStatus.OPTIMAL, GurobiOracleStatus.FEASIBLE):
        return {
            "instance_id": instance_id,
            "instance_seed": instance_seed,
            "method": "full_initial_domain_milp",
            "model_seed": None,
            "success": False,
            "illegal_assignment_count": 0,
            "latest_task_completion": None,
            "terminal_return_tail": None,
            "final_makespan": None,
            "inference_time_seconds": result.solve_time_seconds,
            "wall_runtime_seconds": time.perf_counter() - started,
            "decision_regret": 0.0,
            "state_regret": 0.0,
            "solver_status": result.status.value,
            "event_sequence": [],
        }
    replay = replay_oracle_actions(
        domain,
        result.action_order,
        run_id=f"residual-eval-milp-{instance_id}",
        instance_id=instance_id,
        seed=instance_seed,
        split=DatasetSplit.TEST,
        max_steps=max_steps,
    )
    records = (
        *replay.process_execution_records,
        *replay.transport_execution_records,
    )
    latest = max(
        (float(record["completed_at"]) for record in records),
        default=None,
    )
    return {
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "method": "full_initial_domain_milp",
        "model_seed": None,
        "success": replay.success,
        "illegal_assignment_count": replay.illegal_assignment_count,
        "latest_task_completion": latest,
        "terminal_return_tail": (
            None if replay.makespan is None or latest is None
            else replay.makespan - latest
        ),
        "final_makespan": replay.makespan,
        "inference_time_seconds": result.solve_time_seconds,
        "wall_runtime_seconds": time.perf_counter() - started,
        "decision_regret": 0.0,
        "state_regret": 0.0,
        "solver_status": result.status.value,
        "event_sequence": [
            {
                "order": action.order,
                "decision_time": action.planned_assignment,
                "assignments": [[action.robot_id, action.task_id]],
            }
            for action in result.action_order
        ],
    }




def _attach_state_regret(rows: Sequence[dict[str, object]]) -> None:
    references: dict[tuple[str, tuple[int, ...]], float] = {}
    for row in rows:
        for record in row.get("regret_records", ()):
            best = record.get("best_total_cost")
            pending = tuple(record.get("pending_task_ids", ()))
            if best is None or not pending:
                continue
            key = (str(row["instance_id"]), pending)
            references[key] = min(references.get(key, float("inf")), float(best))
    for row in rows:
        values = []
        for record in row.get("regret_records", ()):
            best = record.get("best_total_cost")
            pending = tuple(record.get("pending_task_ids", ()))
            if best is None or not pending:
                record["state_regret"] = None
                continue
            value = float(best) - references[(str(row["instance_id"]), pending)]
            record["state_regret"] = value
            values.append(value)
        if "regret_records" in row:
            row["state_regret"] = None if not values else statistics.mean(values)


def _remaining_gaps(aggregates: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    result = {}
    masked = aggregates.get("masked_greedy", {}).get("mean_final_makespan")
    milp = aggregates.get("full_initial_domain_milp", {}).get("mean_final_makespan")
    for key, metrics in aggregates.items():
        if not key.startswith("residual_seed"):
            continue
        makespan = metrics.get("mean_final_makespan")
        result[key] = {
            "versus_masked_greedy": None if makespan is None or masked is None else float(makespan) - float(masked),
            "versus_full_initial_domain_milp": None if makespan is None or milp is None else float(makespan) - float(milp),
        }
    return result
def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = {}
    for row in rows:
        key = row["method"] if row["model_seed"] is None else f"{row['method']}_seed{row['model_seed']}"
        grouped.setdefault(str(key), []).append(row)
    result = {}
    for key, items in grouped.items():
        successes = [row for row in items if row["success"]]
        def mean(field):
            values = [float(row[field]) for row in successes if row.get(field) is not None]
            return None if not values else statistics.mean(values)
        result[key] = {
            "runs": len(items),
            "successes": len(successes),
            "success_rate": len(successes) / len(items),
            "illegal_assignment_count": sum(int(row["illegal_assignment_count"]) for row in items),
            "mean_latest_task_completion": mean("latest_task_completion"),
            "mean_terminal_return_tail": mean("terminal_return_tail"),
            "mean_final_makespan": mean("final_makespan"),
            "mean_inference_time_seconds": mean("inference_time_seconds"),
            "mean_wall_runtime_seconds": mean("wall_runtime_seconds"),
            "mean_decision_regret": mean("decision_regret"),
            "mean_state_regret": mean("state_regret"),
        }
    return result


def _paired(rows: Sequence[Mapping[str, object]], model_seed: int) -> dict[str, object]:
    c0 = {
        int(row["instance_seed"]): row for row in rows
        if row["method"] == "legacy_c0" and row["model_seed"] == model_seed
    }
    residual = {
        int(row["instance_seed"]): row for row in rows
        if row["method"] == "residual" and row["model_seed"] == model_seed
    }
    if set(c0) != set(residual):
        raise ValueError(f"paired rows differ for model seed {model_seed}")
    common = [
        seed for seed in sorted(c0)
        if c0[seed]["success"] and residual[seed]["success"]
    ]
    deltas = [
        float(residual[seed]["final_makespan"]) - float(c0[seed]["final_makespan"])
        for seed in common
    ]
    task_deltas = [
        float(residual[seed]["latest_task_completion"]) - float(c0[seed]["latest_task_completion"])
        for seed in common
    ]
    return {
        "model_seed": model_seed,
        "matched_instances": len(c0),
        "common_successes": len(common),
        "mean_final_makespan_delta": statistics.mean(deltas) if deltas else None,
        "mean_task_completion_delta": statistics.mean(task_deltas) if task_deltas else None,
        "win_tie_loss": {
            "win": sum(delta < 0 for delta in deltas),
            "tie": sum(delta == 0 for delta in deltas),
            "loss": sum(delta > 0 for delta in deltas),
        },
        "success_not_decreased": sum(bool(row["success"]) for row in residual.values()) >= sum(bool(row["success"]) for row in c0.values()),
        "illegal_assignments_zero": all(int(row["illegal_assignment_count"]) == 0 for row in residual.values()),
        "task_completion_constraint_met": all(delta <= 1.0 for delta in task_deltas),
        "residual_mean_better": bool(deltas) and statistics.mean(deltas) < 0,
        "deltas": [{"instance_seed": seed, "final_makespan_delta": delta} for seed, delta in zip(common, deltas, strict=True)],
    }


def _representative_traces(rows: Sequence[Mapping[str, object]], paired: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (row["method"], row["model_seed"], row["instance_seed"]): row
        for row in rows
    }
    traces = []
    for comparison in paired:
        model_seed = int(comparison["model_seed"])
        deltas = list(comparison["deltas"])
        ordered = sorted(deltas, key=lambda item: float(item["final_makespan_delta"]))
        selections = (
            ("largest_improvement", ordered[:3]),
            ("median", ordered[max(0, len(ordered) // 2 - 1):len(ordered) // 2 + 2]),
            ("largest_regression", ordered[-3:]),
        )
        for category, selected in selections:
            for item in selected:
                instance_seed = int(item["instance_seed"])
                traces.append({
                    "category": category,
                    "model_seed": model_seed,
                    "instance_seed": instance_seed,
                    "final_makespan_delta": item["final_makespan_delta"],
                    "legacy_c0": indexed[("legacy_c0", model_seed, instance_seed)]["event_sequence"],
                    "residual": indexed[("residual", model_seed, instance_seed)]["event_sequence"],
                })
    return traces


def run_residual_paired_evaluation(
    output_dir: str | Path,
    *,
    c0_checkpoints: Mapping[int, str | Path],
    residual_checkpoints: Mapping[int, str | Path],
    seeds: Sequence[int] = FORMAL_RESIDUAL_SPLIT_PLAN.test_seeds,
    device: str | torch.device = "cpu",
    regret_time_limit: float = 10.0,
    milp_time_limit: float = 60.0,
    max_steps: int = 10_000,
) -> Mapping[str, Path]:
    if set(c0_checkpoints) != set(residual_checkpoints):
        raise ValueError("C0 and residual model seeds must correspond exactly")
    if any(seed not in FORMAL_RESIDUAL_SPLIT_PLAN.test_seeds for seed in seeds):
        raise ValueError("evaluation seeds must belong to the fixed test split")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    c0_models = {
        seed: load_legacy_c0_checkpoint(path, device=device)[0]
        for seed, path in c0_checkpoints.items()
    }
    residual_models = {}
    for seed, path in residual_checkpoints.items():
        model, payload = load_residual_tail_checkpoint(path, device=device)
        if int(payload["config"]["seed"]) != seed:
            raise ValueError(f"residual checkpoint key {seed} does not match its training seed")
        residual_models[seed] = model
    rows = []
    for instance_seed in seeds:
        domain = generate_md_instance(MDGeneratorConfig(seed=instance_seed)).domain
        instance_id = f"md-residual-{instance_seed}"
        for model_seed in sorted(c0_models):
            for method, model in (
                ("legacy_c0", c0_models[model_seed]),
                ("residual", residual_models[model_seed]),
            ):
                rows.append(_run_online(
                    domain,
                    instance_id=instance_id,
                    instance_seed=instance_seed,
                    method=method,
                    model_seed=model_seed,
                    scorer=OnlineNeuralScoreProvider(
                        model,
                        build_md_policy_inputs_from_simulator,
                        device=device,
                    ),
                    decoder=LearnedConstrainedDecoder(),
                    regret_time_limit=regret_time_limit,
                    max_steps=max_steps,
                ))
        rows.append(_run_online(
            domain,
            instance_id=instance_id,
            instance_seed=instance_seed,
            method="masked_greedy",
            model_seed=None,
            scorer=lambda simulator: torch.zeros_like(
                simulator_hard_mask(simulator), dtype=torch.float32
            ),
            decoder=MaskedGreedyDecoder(),
            regret_time_limit=regret_time_limit,
            max_steps=max_steps,
        ))
        rows.append(_run_milp(
            domain,
            instance_id=instance_id,
            instance_seed=instance_seed,
            time_limit=milp_time_limit,
            max_steps=max_steps,
        ))

    _attach_state_regret(rows)
    aggregates = _aggregate(rows)
    paired = [_paired(rows, model_seed) for model_seed in sorted(c0_models)]
    acceptance = {
        "all_residual_seeds_better": all(item["residual_mean_better"] for item in paired),
        "success_not_decreased": all(item["success_not_decreased"] for item in paired),
        "illegal_assignments_zero": all(item["illegal_assignments_zero"] for item in paired),
        "task_completion_constraint_met": all(item["task_completion_constraint_met"] for item in paired),
    }
    acceptance["passed"] = all(acceptance.values())
    outputs = {
        "rollouts": destination / "rollouts.jsonl",
        "summary": destination / "summary.json",
        "representative_traces": destination / "representative_traces.json",
    }
    outputs["rollouts"].write_text(
        "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    outputs["summary"].write_text(json.dumps({
        "test_seeds": list(seeds),
        "aggregates": aggregates,
        "remaining_gaps": _remaining_gaps(aggregates),
        "paired_comparisons": paired,
        "acceptance": acceptance,
    }, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["representative_traces"].write_text(
        json.dumps(_representative_traces(rows, paired), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def _mapping(values: Sequence[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed, separator, path = value.partition("=")
        if not separator:
            raise ValueError("checkpoint must use MODEL_SEED=PATH")
        result[int(seed)] = Path(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--c0-checkpoint", action="append", required=True)
    parser.add_argument("--residual-checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--regret-time-limit", type=float, default=10.0)
    parser.add_argument("--milp-time-limit", type=float, default=60.0)
    args = parser.parse_args(argv)
    paths = run_residual_paired_evaluation(
        args.output_dir,
        c0_checkpoints=_mapping(args.c0_checkpoint),
        residual_checkpoints=_mapping(args.residual_checkpoint),
        device=args.device,
        regret_time_limit=args.regret_time_limit,
        milp_time_limit=args.milp_time_limit,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

