"""Evaluate frozen rerankers on richer initial states with exact selected-action costs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from baselines.exact_online_action_oracle import (
    CompleteOnlineAction,
    enumerate_complete_online_actions,
    solve_exact_action_continuation,
)
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_generation import SnapshotCandidate, residual_state
from experiments.md_exact_action_candidate_diagnostic import MODEL_SEEDS
from experiments.md_minimal_action_features import build_minimal_action_features
from experiments.md_prefix_conditioned_reranker import (
    Candidate,
    PrefixCostScorer,
    State,
    candidate_cost,
)
from experiments.md_prefix_features import PrefixAction
from simulation_environment.domain_model import TransportTask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


PILOT_SEEDS = (76201, 76202, 76206)
EXTENSION_SEEDS = (76207, 76208, 76221, 76223, 76226, 76228, 76229, 76232, 76233)
SOURCE = Path("reports/md_prefix_conditioned_reranker_2026-09-10")
MODES = ("prefix_zero", "prefix")


def _action_type(action: CompleteOnlineAction, tasks: dict[int, Any]) -> str:
    kinds = {
        "transport" if isinstance(tasks[task_id], TransportTask) else "process"
        for _, task_id in action.assignments
    }
    return "mixed" if len(kinds) > 1 else next(iter(kinds))


def _load_reranker(mode: str, model_seed: int) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    checkpoint = SOURCE / mode / f"seed{model_seed}" / "best_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model: torch.nn.Module = PrefixCostScorer()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["normalization_mean"], payload["normalization_std"]


def _select(
    mode: str,
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    state: State,
    *,
    order_index: int = 0,
) -> Candidate:
    costs = torch.stack(
        [
            candidate_cost(
                model,
                state,
                candidate,
                mean,
                std,
                mode=mode,
                order_index=order_index,
            )
            for candidate in state.candidates
        ]
    )
    return state.candidates[int(torch.argmin(costs))]


def run(*, cohort: str = "extension") -> dict[str, Any]:
    seeds = PILOT_SEEDS if cohort == "pilot" else EXTENSION_SEEDS
    output = Path(f"reports/md_rich_prefix_transfer_{cohort}_2026-09-10")
    if (output / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "md-rich-prefix-transfer-pilot-1.0",
        "created_before_evaluation": True,
        "seeds": list(seeds),
        "state": "initial_stationary",
        "selection": (
            "three pre-existing exploration seeds"
            if cohort == "pilot"
            else "first nine seeds from 76207 onward with 24-96 legal complete actions; selected before exact labels"
        ),
        "minimum_pending_tasks": 8,
        "minimum_legal_complete_actions": 24,
        "maximum_legal_complete_actions": 96,
        "candidate_space": "all legal complete actions",
        "models": "frozen late-stage rerankers; no retraining",
        "exact_scope": "only unique actions selected by prefix-zero or prefix orders",
        "solver": {"threads": 1, "time_limit_seconds": 10},
        "decision_checks": {
            "prefix_mean_cost_below_prefix_zero": True,
            "prefix_strict_win_rate_at_least": 0.5,
            "canonical_reverse_choice_disagreement_at_most": 0.25,
        },
        "confirmation_read": False,
        "regression_benchmark_read": False,
        "production_decoder_modified": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rerankers = {
        (mode, seed): _load_reranker(mode, seed)
        for mode in MODES
        for seed in MODEL_SEEDS
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        domain = generate_md_instance(MDGeneratorConfig(seed=seed)).domain
        simulator = MDDiscreteSimulator(domain)
        snapshot = SnapshotCandidate(f"md-rich-{seed}", seed, "initial", domain, simulator, {})
        residual = residual_state(snapshot)
        legal = enumerate_complete_online_actions(domain, residual)
        if len(residual.pending_task_ids) < 8 or not 24 <= len(legal) <= 96:
            failures.append({"seed": seed, "error": "state richness gate failed"})
            continue
        tasks = {task.task_id: task for task in domain.tasks}
        for model_seed in MODEL_SEEDS:
            candidates = [
                Candidate(
                    PrefixAction(action.key, build_minimal_action_features(simulator, action)),
                    regret=0.0,
                    optimal=False,
                    rank=rank,
                )
                for rank, action in enumerate(legal)
            ]
            state = State(
                snapshot_id=f"md-rich-{seed}:initial:t0",
                split="exploration",
                seed=seed,
                model_seed=model_seed,
                pending_count=len(residual.pending_task_ids),
                action_type="rich",
                candidates=candidates,
            )
            selected: dict[str, Candidate] = {}
            for mode in MODES:
                model, mean, std = rerankers[mode, model_seed]
                selected[mode] = _select(mode, model, mean, std, state)
            prefix_model, prefix_mean, prefix_std = rerankers["prefix", model_seed]
            selected["prefix_reverse"] = _select(
                "prefix", prefix_model, prefix_mean, prefix_std, state, order_index=1
            )

            selected_actions = {item.action.assignments for item in selected.values()}
            exact_costs: dict[tuple[tuple[int, int], ...], float] = {}
            for assignments in sorted(selected_actions):
                action = next(action for action in legal if action.key == assignments)
                result = solve_exact_action_continuation(
                    domain, residual, action, time_limit_seconds=10, threads=1
                )
                if result.status.value != "optimal" or result.continuation_cost is None:
                    failures.append(
                        {
                            "seed": seed,
                            "model_seed": model_seed,
                            "action": assignments,
                            "error": f"exact solve {result.status.value}: {result.message}",
                        }
                    )
                    break
                exact_costs[assignments] = float(result.continuation_cost)
            if len(exact_costs) != len(selected_actions):
                continue
            costs = {name: exact_costs[item.action.assignments] for name, item in selected.items()}
            action_types = {
                name: _action_type(
                    next(action for action in legal if action.key == item.action.assignments), tasks
                )
                for name, item in selected.items()
            }
            rows.append(
                {
                    "seed": seed,
                    "model_seed": model_seed,
                    "pending_count": len(residual.pending_task_ids),
                    "legal_complete_action_count": len(legal),
                    "candidate_count": len(legal),
                    "unique_selected_action_count": len(selected_actions),
                    "selected_costs": costs,
                    "selected_action_types": action_types,
                    "prefix_minus_prefix_zero": costs["prefix"] - costs["prefix_zero"],
                    "prefix_beats_prefix_zero": costs["prefix"] < costs["prefix_zero"],
                    "prefix_ties_or_beats_prefix_zero": costs["prefix"] <= costs["prefix_zero"],
                    "prefix_reverse_choice_differs": (
                        selected["prefix"].action.assignments
                        != selected["prefix_reverse"].action.assignments
                    ),
                }
            )

    def mean_cost(name: str) -> float:
        return sum(row["selected_costs"][name] for row in rows) / len(rows)

    pairwise_win_rate = (
        sum(row["prefix_beats_prefix_zero"] for row in rows) / len(rows) if rows else 0.0
    )
    order_disagreement = (
        sum(row["prefix_reverse_choice_differs"] for row in rows) / len(rows)
        if rows
        else 1.0
    )
    metrics = {
        "mean_selected_cost": {
            name: mean_cost(name) for name in ("prefix_zero", "prefix")
        }
        if rows
        else {},
        "prefix_pairwise_strict_win_rate": pairwise_win_rate,
        "prefix_pairwise_tie_or_win_rate": (
            sum(row["prefix_ties_or_beats_prefix_zero"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "canonical_reverse_choice_disagreement_rate": order_disagreement,
        "selected_action_type_counts": dict(
            sorted(Counter(t for row in rows for t in row["selected_action_types"].values()).items())
        ),
    }
    checks = {
        "prefix_mean_cost_below_prefix_zero": bool(
            rows and metrics["mean_selected_cost"]["prefix"] < metrics["mean_selected_cost"]["prefix_zero"]
        ),
        "prefix_strict_win_rate": pairwise_win_rate >= 0.5,
        "order_stability": order_disagreement <= 0.25,
    }
    supported = not failures and len(rows) == len(seeds) * len(MODEL_SEEDS) and all(checks.values())
    summary = {
        "schema": protocol["schema"],
        "status": "rich_state_transfer_signal_supported" if supported else "rich_state_transfer_signal_not_supported",
        "supported": supported,
        "record_count": len(rows),
        "failure_count": len(failures),
        "checks": checks,
        "metrics": metrics,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_read": False,
        "regression_benchmark_read": False,
        "production_decoder_modified": False,
    }
    (output / "failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Rich-State Prefix Transfer Pilot",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "Frozen late-stage prefix and prefix-zero rerankers were evaluated over all legal actions in initial stationary states, with exact costs only for selected actions.",
        "",
        "| Method | Mean selected-action continuation cost |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {value:.4f} |"
        for name, value in metrics.get("mean_selected_cost", {}).items()
    )
    lines += [
        "",
        f"Prefix strict win rate versus prefix-zero: {pairwise_win_rate:.4f}.",
        f"Canonical/reverse prefix choice disagreement: {order_disagreement:.4f}.",
        "",
        "This is an out-of-distribution transfer pilot, not a retrained rich-state confirmation.",
    ]
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("pilot", "extension"), default="extension")
    args = parser.parse_args()
    print(json.dumps(run(cohort=args.cohort), indent=2, sort_keys=True))
