"""Development-only terminal-tail augmentation for the C0 policy."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Final, Sequence

import torch

from experiments.md_context_ablation_relational_data import (
    CANDIDATE_TASK_COUNT,
    RELATIONAL_FAMILIES,
    ROBOT_COUNT,
    RelationalState,
    RelationalTwin,
    build_unconditioned_relational_candidates,
)
from experiments.md_pair_structured_train_diagnostic import (
    _family_balanced_batches,
    _prepare_batch,
)
from experiments.md_flip_curriculum_train_diagnostic import _train_epoch
from experiments.md_task_process_context_models import build_context_ablation_model


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT: Final = (
    REPOSITORY_ROOT / "reports" / "md_terminal_tail_training_development"
)
DEFAULT_LEGACY_ROOT: Final = (
    REPOSITORY_ROOT / "reports" / "md_c0_end_to_end_diagnostic_pilot_2026-09-01"
)
DEFAULT_MODEL_SEEDS: Final = (3101, 3102, 3103)
DEFAULT_EXIT_LOCATION: Final = (0.0, 0.0)


def terminal_action_values(
    state: RelationalState,
    *,
    terminal_weight: float = 2.0,
    exit_location: tuple[float, float] = DEFAULT_EXIT_LOCATION,
) -> tuple[float, ...]:
    """Score forced actions under a joint assignment plus max-return objective."""

    if not math.isfinite(terminal_weight) or terminal_weight < 0:
        raise ValueError("terminal_weight must be non-negative and finite")
    edge_utility = tuple(
        tuple(
            1.25 * state.downstream_priorities[task] - state.eta[robot][task]
            for task in range(CANDIDATE_TASK_COUNT)
        )
        for robot in range(ROBOT_COUNT)
    )
    assignments = []
    for tasks in itertools.permutations(range(CANDIDATE_TASK_COUNT), ROBOT_COUNT):
        return_tail = max(
            math.dist(state.task_deliveries[task], exit_location) / 2.0
            for task in tasks
        )
        assignments.append(
            (
                tasks,
                sum(edge_utility[robot][task] for robot, task in enumerate(tasks))
                - terminal_weight * return_tail,
            )
        )
    values = []
    for robot in range(ROBOT_COUNT):
        for task in range(CANDIDATE_TASK_COUNT):
            joint_value = max(
                total
                for assigned_tasks, total in assignments
                if assigned_tasks[robot] == task
            )
            values.append(joint_value + 0.05 * edge_utility[robot][task])
    return tuple(values)


def relabel_terminal_state(
    state: RelationalState, *, terminal_weight: float = 2.0
) -> RelationalState:
    values = terminal_action_values(state, terminal_weight=terminal_weight)
    oracle = max(range(len(values)), key=lambda action: (values[action], -action))
    return replace(state, action_values=values, oracle_action=oracle)


def build_terminal_tail_training_pairs(
    *,
    candidate_pool_per_family: int = 5_000,
    regular_pairs_per_family: int = 400,
    terminal_pairs_per_family: int = 100,
    regular_seed: int = 3030,
    terminal_seed: int = 7303,
    terminal_weight: float = 2.0,
) -> tuple[tuple[RelationalTwin, ...], dict[str, object]]:
    """Mix ordinary C0 pairs with high-regret terminal-aware pairs."""

    for name, value in (
        ("candidate_pool_per_family", candidate_pool_per_family),
        ("regular_pairs_per_family", regular_pairs_per_family),
        ("terminal_pairs_per_family", terminal_pairs_per_family),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    regular_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=regular_seed,
    )
    terminal_candidates = build_unconditioned_relational_candidates(
        candidates_per_family=candidate_pool_per_family,
        seed=terminal_seed,
    )
    selected: list[RelationalTwin] = []
    counts: dict[str, dict[str, int | float]] = {}
    for family in RELATIONAL_FAMILIES:
        regular = [pair for pair in regular_candidates if pair.family == family]
        if len(regular) < regular_pairs_per_family:
            raise ValueError(f"insufficient regular candidates for {family}")
        regular = regular[:regular_pairs_per_family]

        ranked: list[tuple[float, RelationalTwin]] = []
        for pair in terminal_candidates:
            if pair.family != family:
                continue
            before = relabel_terminal_state(
                pair.before, terminal_weight=terminal_weight
            )
            after = relabel_terminal_state(pair.after, terminal_weight=terminal_weight)
            local_actions = (pair.before.oracle_action, pair.after.oracle_action)
            regrets = (
                before.action_values[before.oracle_action]
                - before.action_values[local_actions[0]],
                after.action_values[after.oracle_action]
                - after.action_values[local_actions[1]],
            )
            if max(regrets) <= 0:
                continue
            terminal_pair_id = f"terminal-tail-{pair.pair_id}"
            ranked.append(
                (
                    max(regrets),
                    RelationalTwin(
                        family=pair.family,
                        pair_id=terminal_pair_id,
                        changed_entity_index=pair.changed_entity_index,
                        before=replace(
                            before,
                            pair_id=terminal_pair_id,
                            state_id=f"{terminal_pair_id}-before",
                        ),
                        after=replace(
                            after,
                            pair_id=terminal_pair_id,
                            state_id=f"{terminal_pair_id}-after",
                        ),
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].pair_id))
        if len(ranked) < terminal_pairs_per_family:
            raise ValueError(f"insufficient terminal-tail candidates for {family}")
        chosen = ranked[:terminal_pairs_per_family]
        terminal = [pair for _regret, pair in chosen]
        cycle_count = math.gcd(regular_pairs_per_family, terminal_pairs_per_family)
        regular_per_cycle = regular_pairs_per_family // cycle_count
        terminal_per_cycle = terminal_pairs_per_family // cycle_count
        for cycle in range(cycle_count):
            selected.extend(
                regular[
                    cycle * regular_per_cycle : (cycle + 1) * regular_per_cycle
                ]
            )
            selected.extend(
                terminal[
                    cycle * terminal_per_cycle : (cycle + 1) * terminal_per_cycle
                ]
            )
        counts[family] = {
            "regular_pairs": regular_pairs_per_family,
            "terminal_pairs": terminal_pairs_per_family,
            "eligible_terminal_pairs": len(ranked),
            "minimum_selected_terminal_regret": chosen[-1][0],
            "maximum_selected_terminal_regret": chosen[0][0],
        }
    return tuple(selected), {
        "candidate_pool_per_family": candidate_pool_per_family,
        "regular_seed": regular_seed,
        "terminal_seed": terminal_seed,
        "terminal_weight": terminal_weight,
        "exit_location": list(DEFAULT_EXIT_LOCATION),
        "family_counts": counts,
        "pair_count": len(selected),
        "terminal_pair_fraction": terminal_pairs_per_family
        / (regular_pairs_per_family + terminal_pairs_per_family),
    }


def train_terminal_tail_seed(
    output_root: str | Path,
    *,
    legacy_root: str | Path,
    model_seed: int,
    device: str = "cuda:0",
    epochs: int = 100,
    pairs_per_family_per_batch: int = 100,
    candidate_pool_per_family: int = 5_000,
    regular_pairs_per_family: int = 400,
    terminal_pairs_per_family: int = 100,
) -> Path:
    """Fine-tune one legacy C0 checkpoint without overwriting it."""

    destination = Path(output_root) / f"C0_terminal_tail_seed{model_seed}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite terminal-tail run: {destination}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if str(device).startswith("cuda") and cublas_workspace not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError(
            "CUDA deterministic training requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    pairs, manifest = build_terminal_tail_training_pairs(
        candidate_pool_per_family=candidate_pool_per_family,
        regular_pairs_per_family=regular_pairs_per_family,
        terminal_pairs_per_family=terminal_pairs_per_family,
    )
    batches = tuple(
        _prepare_batch(batch, device=device)
        for batch in _family_balanced_batches(
            pairs, pairs_per_family_per_batch=pairs_per_family_per_batch
        )
    )
    checkpoint = (
        Path(legacy_root)
        / "training"
        / f"C0_seed{model_seed}"
        / "checkpoints"
        / "best_checkpoint.pt"
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    torch.use_deterministic_algorithms(True)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=torch.device(device)
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    epoch_log = []
    for epoch in range(1, epochs + 1):
        losses = _train_epoch(
            model,
            batches,
            optimizer,
            state_margin=0.1,
            flip_state_margin=0.1,
            pair_margin=0.1,
            flip_state_loss_weight=0.5,
            pair_loss_weight=0.25,
            saturation_loss_weight=0.1,
            saturation_penalty_allowance=0.25,
        )
        epoch_log.append({"epoch": epoch, **losses})
    destination.mkdir(parents=True)
    checkpoint_path = destination / "best_checkpoint.pt"
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": state_dict,
            "variant": "C0_terminal_tail_development",
            "model_seed": model_seed,
            "legacy_checkpoint": str(checkpoint.resolve()),
            "dataset_manifest": manifest,
        },
        checkpoint_path,
    )
    reloaded = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=torch.device("cpu")
    )
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    reload_verified = all(
        torch.equal(reloaded.state_dict()[name], value)
        for name, value in state_dict.items()
    )
    if not reload_verified:
        raise RuntimeError("terminal-tail checkpoint reload mismatch")
    (destination / "training_pairs.jsonl").write_text(
        "".join(json.dumps(asdict(pair), sort_keys=True) + "\n" for pair in pairs),
        encoding="utf-8",
    )
    (destination / "run.json").write_text(
        json.dumps(
            {
                "status": "development_only",
                "model_seed": model_seed,
                "legacy_checkpoint": str(checkpoint.resolve()),
                "dataset_manifest": manifest,
                "epochs": epoch_log,
                "held_out_data_read": False,
                "checkpoint_reload_verified": reload_verified,
                "deterministic_algorithms_enabled": True,
                "cublas_workspace_config": cublas_workspace,
                "training_pairs_file": "training_pairs.jsonl",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--seed", type=int, choices=DEFAULT_MODEL_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_terminal_tail_seed(
        args.output_root,
        legacy_root=args.legacy_root,
        model_seed=args.seed,
        device=args.device,
        epochs=args.epochs,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
