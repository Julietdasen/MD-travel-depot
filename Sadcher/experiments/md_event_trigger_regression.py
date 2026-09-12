"""Paired Ticket 46 regression for conservative event-gated scoring.

This diagnostic keeps the frozen C0 instances, checkpoints, decoder, and
fallback fixed.  It only changes whether the scheduler scores states that have
pairwise-feasible edges but no complete dispatchable action while work advances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from data_generation.md_dataset import MDInstanceRecord
from experiments.md_c0_end_to_end_diagnostic_pilot import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXIT_LOCATION,
    FALLBACK_CONFIDENCE_THRESHOLD,
    FALLBACK_THREADS,
    LATENCY_WARMUP_FORWARDS,
    MAX_ROLLOUT_STEPS,
    MODEL_SEEDS,
    OUTPUT_ROOT as TICKET46_ROOT,
    _load_learned_checkpoint,
    _online_raw_record,
    _warm_up_model,
    load_frozen_diagnostic_records,
)
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    OnlineMDScheduler,
    OnlineNeuralScoreProvider,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "reports" / "md_c0_event_trigger_regression_2026-09-02"
)
POLLING = "polling"
EVENT_GATED = "event_gated"
CONDITIONS = (POLLING, EVENT_GATED)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _resolve_single_cuda_device(device: str | torch.device) -> torch.device:
    if str(device).lower() != "cuda:0":
        raise ValueError("event-trigger regression requires device cuda:0")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "," in visible_devices:
        raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("event-trigger regression requires exactly one visible GPU")
    return torch.device("cuda:0")


def _starvation_mean(row: Mapping[str, object]) -> float:
    values = tuple(float(value) for value in dict(row["material_starvation"]).values())
    return statistics.fmean(values) if values else 0.0


def _compact_rollout(
    result,
    *,
    condition: str,
    model_seed: int,
    instance: MDInstanceRecord,
    checkpoint_identity: Mapping[str, object],
) -> dict[str, Any]:
    """Persist outcomes and action traces without duplicating verbose simulator data."""

    raw = _online_raw_record(
        result,
        method="C0",
        model_seed=model_seed,
        instance=instance,
        checkpoint_identity=checkpoint_identity,
    )
    decisions = result.decision_records
    fallbacks = result.fallback_records
    return {
        key: raw[key]
        for key in (
            "schema_version",
            "ticket",
            "method",
            "model_seed",
            "instance_id",
            "instance_seed",
            "success",
            "failure_reason",
            "makespan",
            "material_starvation",
            "robot_utilization",
            "illegal_assignment_count",
            "repair_count",
            "repair_rate",
            "repair_event_count",
            "fallback_count",
            "fallback_rate",
            "fallback_reasons",
            "solver_calls",
            "solver_time_seconds",
            "decision_count",
            "latency_totals",
            "wall_runtime_seconds",
            "terminal_state",
            "checkpoint_identity",
        )
    } | {
        "condition": condition,
        "scored_decision_count": len(decisions),
        "empty_scored_decision_count": sum(not record.assignments for record in decisions),
        "assignment_decision_count": sum(bool(record.assignments) for record in decisions),
        "action_trace": [
            {
                "decision_time": record.decision_time,
                "assignments": [list(assignment) for assignment in record.assignments],
            }
            for record in decisions
            if record.assignments
        ],
        "fallback_trace": [
            {
                "decision_time": record.decision_time,
                "trigger_reason": record.trigger_reason,
                "assignments": [list(assignment) for assignment in record.assignments],
                "failure_reason": record.failure_reason,
            }
            for record in fallbacks
        ],
    }


def _scheduler_for_condition(
    model, *, device: torch.device, condition: str
) -> OnlineMDScheduler:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    return OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            model, build_md_policy_inputs_from_simulator, device=device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=FALLBACK_THREADS),
        confidence_threshold=FALLBACK_CONFIDENCE_THRESHOLD,
        max_steps=MAX_ROLLOUT_STEPS,
        skip_non_dispatchable_states=condition == EVENT_GATED,
    )


def _run_seed(
    source_root: Path,
    *,
    model_seed: int,
    records: Sequence[MDInstanceRecord],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    model, checkpoint_identity = _load_learned_checkpoint(
        source_root, "C0", model_seed, device
    )
    _warm_up_model(model, device)
    schedulers = {
        condition: _scheduler_for_condition(model, device=device, condition=condition)
        for condition in CONDITIONS
    }
    rows = {condition: [] for condition in CONDITIONS}
    for index, instance in enumerate(records):
        # Alternate first-run order to avoid attributing cache or thermal drift to a mode.
        order = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in order:
            result = schedulers[condition].run(
                instance.generated.domain,
                run_id=(
                    f"ticket46-event-trigger-{condition}-C0-{model_seed}-"
                    f"{instance.instance_id}"
                ),
                instance_id=instance.instance_id,
                seed=instance.seed,
                split=instance.split,
                exit_location=EXIT_LOCATION,
            )
            rows[condition].append(
                _compact_rollout(
                    result,
                    condition=condition,
                    model_seed=model_seed,
                    instance=instance,
                    checkpoint_identity=checkpoint_identity,
                )
            )
    return rows


def _bootstrap_ci95(values: Sequence[float], *, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    ]


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = tuple(row for row in rows if bool(row["success"]))
    makespans = tuple(
        float(row["makespan"])
        for row in successes
        if row["makespan"] is not None
    )
    latency_keys = (
        "encoder_seconds",
        "scoring_seconds",
        "decoder_seconds",
        "repair_seconds",
        "fallback_seconds",
        "total_seconds",
    )
    return {
        "run_count": len(rows),
        "success_count": len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "mean_makespan_on_successes": statistics.fmean(makespans) if makespans else None,
        "mean_material_starvation": (
            statistics.fmean(_starvation_mean(row) for row in rows) if rows else None
        ),
        "total_scorer_calls": sum(int(row["scored_decision_count"]) for row in rows),
        "mean_scorer_calls": (
            statistics.fmean(float(row["scored_decision_count"]) for row in rows)
            if rows
            else None
        ),
        "total_empty_scored_decisions": sum(
            int(row["empty_scored_decision_count"]) for row in rows
        ),
        "mean_total_decision_latency_seconds": (
            statistics.fmean(
                float(dict(row["latency_totals"])["total_seconds"]) for row in rows
            )
            if rows
            else None
        ),
        "total_decision_latency_seconds": sum(
            float(dict(row["latency_totals"])["total_seconds"]) for row in rows
        ),
        "latency_component_totals_seconds": {
            key: sum(float(dict(row["latency_totals"])[key]) for row in rows)
            for key in latency_keys
        },
        "mean_wall_runtime_seconds": (
            statistics.fmean(float(row["wall_runtime_seconds"]) for row in rows)
            if rows
            else None
        ),
        "total_wall_runtime_seconds": sum(
            float(row["wall_runtime_seconds"]) for row in rows
        ),
        "fallback_count": sum(int(row["fallback_count"]) for row in rows),
        "solver_time_seconds": sum(float(row["solver_time_seconds"]) for row in rows),
        "illegal_assignment_count": sum(
            int(row["illegal_assignment_count"]) for row in rows
        ),
    }


def _rows_by_pair(
    polling: Sequence[Mapping[str, object]], event_gated: Sequence[Mapping[str, object]]
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], ...]:
    def key(row: Mapping[str, object]) -> tuple[int, str]:
        return int(row["model_seed"]), str(row["instance_id"])

    polling_by_key = {key(row): row for row in polling}
    gated_by_key = {key(row): row for row in event_gated}
    if len(polling_by_key) != len(polling) or len(gated_by_key) != len(event_gated):
        raise ValueError("each condition must contain unique model-seed/instance pairs")
    if polling_by_key.keys() != gated_by_key.keys():
        raise ValueError("conditions must contain identical paired cases")
    return tuple(
        (polling_by_key[pair_key], gated_by_key[pair_key])
        for pair_key in sorted(polling_by_key)
    )


def _paired_delta(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    metric: Callable[[Mapping[str, object]], float | None],
    *,
    seed: int,
) -> dict[str, object]:
    differences = []
    for polling, event_gated in pairs:
        polling_value = metric(polling)
        gated_value = metric(event_gated)
        if polling_value is not None and gated_value is not None:
            differences.append(gated_value - polling_value)
    return {
        "count": len(differences),
        "event_gated_minus_polling_mean": (
            statistics.fmean(differences) if differences else None
        ),
        "event_gated_minus_polling_median": (
            statistics.median(differences) if differences else None
        ),
        "bootstrap_ci95": _bootstrap_ci95(differences, seed=seed),
    }


def _outcome_mismatch(
    polling: Mapping[str, object], event_gated: Mapping[str, object]
) -> list[str]:
    fields = (
        "success",
        "failure_reason",
        "makespan",
        "material_starvation",
        "robot_utilization",
        "illegal_assignment_count",
        "terminal_state",
        "action_trace",
        "fallback_trace",
    )
    return [field for field in fields if polling[field] != event_gated[field]]


def _percent_reduction(polling: float, event_gated: float) -> float | None:
    if polling <= 0.0:
        return None
    return 100.0 * (1.0 - event_gated / polling)


def build_regression_summary(
    polling: Sequence[Mapping[str, object]], event_gated: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Build paired semantic and efficiency statistics from compact rollout rows."""

    pairs = _rows_by_pair(polling, event_gated)
    polling_summary = _summary(polling)
    gated_summary = _summary(event_gated)
    mismatches = []
    for polling_row, gated_row in pairs:
        fields = _outcome_mismatch(polling_row, gated_row)
        if fields:
            mismatches.append(
                {
                    "model_seed": polling_row["model_seed"],
                    "instance_id": polling_row["instance_id"],
                    "different_fields": fields,
                }
            )
    polling_calls = float(polling_summary["total_scorer_calls"])
    gated_calls = float(gated_summary["total_scorer_calls"])
    polling_decision_seconds = float(polling_summary["total_decision_latency_seconds"])
    gated_decision_seconds = float(gated_summary["total_decision_latency_seconds"])
    polling_wall_seconds = float(polling_summary["total_wall_runtime_seconds"])
    gated_wall_seconds = float(gated_summary["total_wall_runtime_seconds"])
    return {
        "comparison": "event_gated_minus_polling",
        "paired_case_count": len(pairs),
        "conditions": {
            POLLING: polling_summary,
            EVENT_GATED: gated_summary,
        },
        "semantic_regression": {
            "identical_outcome_pair_count": len(pairs) - len(mismatches),
            "all_outcomes_identical": not mismatches,
            "mismatches": mismatches,
        },
        "paired_deltas": {
            "makespan_seconds": _paired_delta(
                pairs,
                lambda row: (
                    float(row["makespan"])
                    if bool(row["success"]) and row["makespan"] is not None
                    else None
                ),
                seed=BOOTSTRAP_SEED,
            ),
            "mean_material_starvation": _paired_delta(
                pairs,
                _starvation_mean,
                seed=BOOTSTRAP_SEED + 1,
            ),
            "scorer_calls": _paired_delta(
                pairs,
                lambda row: float(row["scored_decision_count"]),
                seed=BOOTSTRAP_SEED + 2,
            ),
            "total_decision_latency_seconds": _paired_delta(
                pairs,
                lambda row: float(dict(row["latency_totals"])["total_seconds"]),
                seed=BOOTSTRAP_SEED + 3,
            ),
            "wall_runtime_seconds": _paired_delta(
                pairs,
                lambda row: float(row["wall_runtime_seconds"]),
                seed=BOOTSTRAP_SEED + 4,
            ),
        },
        "total_reductions_percent": {
            "scorer_calls": _percent_reduction(polling_calls, gated_calls),
            "decision_latency": _percent_reduction(
                polling_decision_seconds, gated_decision_seconds
            ),
            "wall_runtime": _percent_reduction(polling_wall_seconds, gated_wall_seconds),
        },
    }


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return str(value)
    return str(value)


def _render_report(summary: Mapping[str, object], *, source_root: Path) -> str:
    conditions = dict(summary["conditions"])
    polling = dict(conditions[POLLING])
    gated = dict(conditions[EVENT_GATED])
    deltas = dict(summary["paired_deltas"])
    semantic = dict(summary["semantic_regression"])
    reductions = dict(summary["total_reductions_percent"])
    rows = (
        ("Success rate", polling["success_rate"], gated["success_rate"], None),
        (
            "Mean success makespan",
            polling["mean_makespan_on_successes"],
            gated["mean_makespan_on_successes"],
            dict(deltas["makespan_seconds"])["event_gated_minus_polling_mean"],
        ),
        (
            "Mean material starvation",
            polling["mean_material_starvation"],
            gated["mean_material_starvation"],
            dict(deltas["mean_material_starvation"])[
                "event_gated_minus_polling_mean"
            ],
        ),
        (
            "Total scorer calls",
            polling["total_scorer_calls"],
            gated["total_scorer_calls"],
            dict(deltas["scorer_calls"])["event_gated_minus_polling_mean"],
        ),
        (
            "Total decision latency (s)",
            polling["total_decision_latency_seconds"],
            gated["total_decision_latency_seconds"],
            dict(deltas["total_decision_latency_seconds"])[
                "event_gated_minus_polling_mean"
            ],
        ),
        (
            "Total scheduler wall time (s)",
            polling["total_wall_runtime_seconds"],
            gated["total_wall_runtime_seconds"],
            dict(deltas["wall_runtime_seconds"])[
                "event_gated_minus_polling_mean"
            ],
        ),
        ("Fallback calls", polling["fallback_count"], gated["fallback_count"], None),
    )
    table = "\n".join(
        f"| {label} | {_format(polling_value)} | {_format(gated_value)} | "
        f"{_format(delta)} |"
        for label, polling_value, gated_value, delta in rows
    )
    decision_delta = dict(deltas["total_decision_latency_seconds"])
    wall_delta = dict(deltas["wall_runtime_seconds"])
    return "\n".join(
        (
            "# Ticket 46 C0 Event-Trigger Regression",
            "",
            "This independent diagnostic reuses the frozen Ticket 46 C0 instances and "
            "checkpoints. It does not replace the frozen Ticket 46 report.",
            "",
            "## Protocol",
            "",
            f"- Source package: `{source_root}`",
            f"- Paired cases: {summary['paired_case_count']}",
            "- Within each model seed, polling and event-gated runs alternate first-run "
            "order by instance.",
            "- The only varied scheduler setting is `skip_non_dispatchable_states`.",
            "- Event-gated mode skips a neural forward only while advancing work exists "
            "and no complete assignment can be dispatched.",
            "",
            "## Results",
            "",
            "| Metric | Polling | Event-gated | Mean paired delta (gated - polling) |",
            "|---|---:|---:|---:|",
            table,
            "",
            "## Semantic Check",
            "",
            f"- Identical outcome pairs: {semantic['identical_outcome_pair_count']}/"
            f"{summary['paired_case_count']}.",
            f"- All outcomes identical: {str(semantic['all_outcomes_identical']).lower()}.",
            f"- Makespan paired mean delta (event-gated minus polling): "
            f"{_format(dict(deltas['makespan_seconds'])['event_gated_minus_polling_mean'])}; "
            f"bootstrap 95% CI {dict(deltas['makespan_seconds'])['bootstrap_ci95']}.",
            "",
            "## Efficiency",
            "",
            f"- Total scorer-call reduction: {_format(reductions['scorer_calls'])}%.",
            f"- Total decision-latency reduction: {_format(reductions['decision_latency'])}% "
            f"(paired mean delta { _format(decision_delta['event_gated_minus_polling_mean']) } s).",
            f"- Total scheduler wall-time reduction: {_format(reductions['wall_runtime'])}% "
            f"(paired mean delta { _format(wall_delta['event_gated_minus_polling_mean']) } s).",
            "",
            "Decision latency and scorer-call counts are the primary efficiency measures. "
            "Wall time is retained as a secondary measurement because it also includes "
            "simulator and host scheduling overhead.",
            "",
        )
    )


def run_event_trigger_regression(
    *,
    source_root: str | Path = TICKET46_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str | torch.device = "cuda:0",
    model_seeds: Sequence[int] = MODEL_SEEDS,
    instance_limit: int | None = None,
) -> dict[str, Path]:
    """Run the interleaved paired regression and write independent artifacts."""

    source_path = Path(source_root)
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {destination}")
    selected_seeds = tuple(int(seed) for seed in model_seeds)
    if not selected_seeds or any(seed not in MODEL_SEEDS for seed in selected_seeds):
        raise ValueError("model_seeds must be a non-empty subset of Ticket 46 C0 seeds")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("model_seeds must not contain duplicates")
    records = load_frozen_diagnostic_records(source_path)
    if instance_limit is not None:
        if instance_limit <= 0:
            raise ValueError("instance_limit must be positive")
        records = records[:instance_limit]
    if not records:
        raise ValueError("at least one frozen instance is required")
    runtime_device = _resolve_single_cuda_device(device)
    missing_checkpoints = [
        source_path / "training" / f"C0_seed{seed}" / "checkpoints" / "best_checkpoint.pt"
        for seed in selected_seeds
        if not (
            source_path
            / "training"
            / f"C0_seed{seed}"
            / "checkpoints"
            / "best_checkpoint.pt"
        ).exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "missing C0 checkpoints: " + ", ".join(map(str, missing_checkpoints))
        )

    destination.mkdir(parents=True)
    all_rows = {condition: [] for condition in CONDITIONS}
    for model_seed in selected_seeds:
        seed_rows = _run_seed(
            source_path,
            model_seed=model_seed,
            records=records,
            device=runtime_device,
        )
        for condition in CONDITIONS:
            rows = seed_rows[condition]
            all_rows[condition].extend(rows)
            _write_jsonl(
                destination / "rollouts" / f"C0_seed{model_seed}_{condition}.jsonl",
                rows,
            )
        torch.cuda.empty_cache()

    summary = build_regression_summary(all_rows[POLLING], all_rows[EVENT_GATED])
    runtime = {
        "requested_device": str(device),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(runtime_device),
        "warmup_forwards_per_checkpoint": LATENCY_WARMUP_FORWARDS,
        "interleaving": "alternating polling/event_gated first-run order by instance",
        "source_ticket46_root": str(source_path),
        "model_seeds": list(selected_seeds),
        "instance_count": len(records),
    }
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "runtime.json", runtime)
    report_path = destination / "final_report.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(_render_report(summary, source_root=source_path))
    return {
        "summary": destination / "summary.json",
        "runtime": destination / "runtime.json",
        "final_report": report_path,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired C0 polling versus event-gated Ticket 46 regression."
    )
    parser.add_argument("--source-root", type=Path, default=TICKET46_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--instance-limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    result = run_event_trigger_regression(
        source_root=args.source_root,
        output_root=args.output_root,
        device=args.device,
        model_seeds=args.model_seeds,
        instance_limit=args.instance_limit,
    )
    print(json.dumps(result, default=str, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
