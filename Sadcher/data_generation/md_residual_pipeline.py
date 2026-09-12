"""Resumable, manifest-driven residual snapshot generation CLI."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_dataset import (
    FORMAL_RESIDUAL_SPLIT_PLAN,
    ResidualDecisionSample,
    ResidualSplitPlan,
)
from data_generation.md_residual_generation import (
    ResidualExclusion,
    SnapshotCandidate,
    collect_rollout_snapshots,
    label_snapshot,
    snapshot_key,
)
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


@dataclass(frozen=True, slots=True)
class ResidualGenerationConfig:
    split_plan: ResidualSplitPlan = FORMAL_RESIDUAL_SPLIT_PLAN
    workers: int = 4
    time_limit_seconds: float = 10.0
    solver_threads: int = 1
    max_steps: int = 10_000
    model_seed: int | None = None

    def __post_init__(self) -> None:
        if self.workers <= 0 or self.solver_threads <= 0 or self.max_steps <= 0:
            raise ValueError("workers, solver_threads and max_steps must be positive")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.model_seed is not None and self.model_seed not in (3101, 3102, 3103):
            raise ValueError("model_seed must be one of 3101, 3102 or 3103")


def encode_training_features(simulator) -> dict[str, object]:
    robot, task, adjacency, md = build_md_policy_inputs_from_simulator(simulator)
    return {
        "robot_ids": sorted(simulator.robot_states),
        "task_ids": sorted(simulator.task_states),
        "robot_features": robot[0].tolist(),
        "task_features": task[0].tolist(),
        "task_adjacency": adjacency[0].tolist(),
        "md_inputs": {
            "robot_metadata": md.robot_metadata[0].tolist(),
            "task_metadata": md.task_metadata[0].tolist(),
            "pair_metadata": md.pair_metadata[0].tolist(),
            "task_is_transport": md.task_is_transport[0].tolist(),
            "typed_adjacency": md.typed_adjacency[0].tolist(),
            "downstream_task_index": md.downstream_task_index[0].tolist(),
            "hard_feasibility_mask": md.hard_feasibility_mask[0].tolist(),
            "opportunity_context": (
                None
                if md.opportunity_context is None
                else md.opportunity_context[0].tolist()
            ),
        },
    }


def _dispatch(decoder, scorer):
    def dispatch(simulator) -> None:
        hard_mask = simulator_hard_mask(simulator)
        if not bool(torch.any(hard_mask)):
            return
        output = scorer(simulator)
        scores = output.scores if isinstance(output, ScoreOutput) else output
        decoded = decoder.decode(
            scores,
            simulator,
            hard_feasibility_mask=hard_mask,
        )
        if decoded.assignments:
            apply_decoder_result(simulator, decoded)

    return dispatch


def collect_seed_snapshots(
    seed: int,
    *,
    c0_checkpoints: Mapping[int, str | Path],
    residual_checkpoints: Mapping[int, str | Path] | None = None,
    device: str | torch.device = "cpu",
    max_steps: int = 10_000,
) -> tuple[SnapshotCandidate, ...]:
    generated = generate_md_instance(MDGeneratorConfig(seed=seed))
    domain = generated.domain
    instance_id = f"md-residual-{seed}"
    candidates = list(
        collect_rollout_snapshots(
            instance_id,
            seed,
            "masked_greedy",
            domain,
            _dispatch(
                MaskedGreedyDecoder(),
                lambda simulator: torch.zeros_like(
                    simulator_hard_mask(simulator), dtype=torch.float32
                ),
            ),
            encode_training_features,
            max_steps=max_steps,
        )
    )
    for model_seed, checkpoint in sorted(c0_checkpoints.items()):
        model, _ = load_legacy_c0_checkpoint(checkpoint, device=device)
        candidates.extend(
            collect_rollout_snapshots(
                instance_id,
                seed,
                f"legacy_c0_seed{model_seed}",
                domain,
                _dispatch(
                    LearnedConstrainedDecoder(),
                    OnlineNeuralScoreProvider(
                        model,
                        build_md_policy_inputs_from_simulator,
                        device=device,
                    ),
                ),
                encode_training_features,
                max_steps=max_steps,
            )
        )
    for model_seed, checkpoint in sorted((residual_checkpoints or {}).items()):
        model, _ = load_residual_tail_checkpoint(checkpoint, device=device)
        candidates.extend(
            collect_rollout_snapshots(
                instance_id,
                seed,
                f"residual_dagger_seed{model_seed}",
                domain,
                _dispatch(
                    LearnedConstrainedDecoder(),
                    OnlineNeuralScoreProvider(
                        model,
                        build_md_policy_inputs_from_simulator,
                        device=device,
                    ),
                ),
                encode_training_features,
                max_steps=max_steps,
            )
        )
    return tuple(candidates)


def _probe_worker(_: int) -> bool:
    try:
        import gurobipy as gp

        model = gp.Model()
        model.Params.OutputFlag = 0
        value = model.addVar(lb=0.0)
        model.setObjective(value)
        model.optimize()
        return int(model.Status) == int(gp.GRB.OPTIMAL)
    except Exception:
        return False


def probe_gurobi_workers(requested: int) -> int:
    if requested <= 1:
        return 1
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=requested) as pool:
            supported = tuple(pool.map(_probe_worker, range(requested)))
    except Exception:
        return 1
    return requested if all(supported) else 1


def _shard_name(candidate: SnapshotCandidate, ordinal: int) -> str:
    method = candidate.rollout_method.replace("/", "_")
    return f"{candidate.seed}_{method}_t{candidate.simulator.time}.json"


def _label_shard(job) -> str:
    candidate, split_plan, time_limit, threads, shard_path, batch_dir = job
    result = label_snapshot(
        candidate,
        split_plan=split_plan,
        time_limit_seconds=time_limit,
        threads=threads,
        batch_cache_dir=batch_dir,
    )
    envelope = {
        "kind": "exclusion" if isinstance(result, ResidualExclusion) else "sample",
        "payload": asdict(result) if isinstance(result, ResidualExclusion) else result.to_dict(),
    }
    temporary = shard_path.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(envelope, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(shard_path)
    return envelope["kind"]


def generate_sharded_dataset(
    candidates: Sequence[SnapshotCandidate],
    output_dir: str | Path,
    *,
    config: ResidualGenerationConfig = ResidualGenerationConfig(),
    resume: bool = False,
) -> Mapping[str, Path]:
    started = time.perf_counter()
    destination = Path(output_dir)
    if destination.exists() and not resume:
        raise FileExistsError(f"residual output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = destination / "manifest.json"
    if resume and existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing.get("split_plan") != config.split_plan.to_dict():
            raise ValueError("resume split plan differs from existing manifest")
        if existing.get("model_seed") != config.model_seed:
            raise ValueError("resume model seed differs from existing manifest")
    shard_root = destination / "shards"
    batch_root = destination / "batch_shards"

    unique = []
    seen = set()
    instance_splits = {}
    for candidate in candidates:
        split = config.split_plan.split_for_seed(candidate.seed)
        previous = instance_splits.setdefault(candidate.instance_id, split)
        if previous != split:
            raise ValueError(f"instance {candidate.instance_id} occurs in multiple splits")
        key = snapshot_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    jobs = []
    resumed = 0
    for ordinal, candidate in enumerate(unique):
        split = config.split_plan.split_for_seed(candidate.seed)
        shard_path = shard_root / split / _shard_name(candidate, ordinal)
        if shard_path.exists():
            resumed += 1
            continue
        jobs.append((
            candidate,
            config.split_plan,
            config.time_limit_seconds,
            config.solver_threads,
            shard_path,
            batch_root / split / shard_path.stem,
        ))

    workers = probe_gurobi_workers(config.workers)
    if workers == 1:
        for job in jobs:
            _label_shard(job)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            tuple(pool.map(_label_shard, jobs))

    samples: list[ResidualDecisionSample] = []
    exclusions: list[ResidualExclusion] = []
    for path in sorted(shard_root.glob("*/*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope["kind"] == "sample":
            samples.append(ResidualDecisionSample.from_dict(envelope["payload"]))
        else:
            exclusions.append(ResidualExclusion(**envelope["payload"]))

    paths = {
        split: destination / f"{split}.jsonl"
        for split in ("train", "development", "test")
    }
    for split, path in paths.items():
        rows = [sample.to_dict() for sample in samples if sample.split == split]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    exclusion_path = destination / "exclusions.jsonl"
    exclusion_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in exclusions),
        encoding="utf-8",
    )

    pending = {
        str(count): sum(len(sample.state.pending_task_ids) == count for sample in samples)
        for count in (1, 2, 3)
    }
    sources = sorted({sample.simulator_snapshot["rollout_method"] for sample in samples})
    source_counts = {
        source: sum(sample.simulator_snapshot["rollout_method"] == source for sample in samples)
        for source in sources
    }
    distribution_gate = {
        "pending_1_2_3_each_at_least_20_percent": bool(samples) and all(
            pending[str(count)] / len(samples) >= 0.20 for count in (1, 2, 3)
        ),
        "masked_greedy_present": source_counts.get("masked_greedy", 0) > 0,
        "legacy_c0_present": any(
            count > 0 and source.startswith("legacy_c0")
            for source, count in source_counts.items()
        ),
    }

    manifest = {
        "schema_version": "residual-il-2.0",
        "model_seed": config.model_seed,
        "split_plan": config.split_plan.to_dict(),
        "instance_count": len({sample.instance_id for sample in samples}),
        "snapshot_count": len(samples),
        "snapshots_by_split": {
            split: sum(sample.split == split for sample in samples) for split in paths
        },
        "forced_solve_count": sum(sample.batch_count for sample in samples) + sum(row.attempted_batch_count for row in exclusions),
        "optimal_solve_count": sum(sample.batch_count for sample in samples),
        "exclusion_count": len(exclusions),
        "exclusions_by_reason": {
            reason: sum(row.reason == reason for row in exclusions)
            for reason in sorted({row.reason for row in exclusions})
        },
        "pending_count_distribution": pending,
        "rollout_sources": source_counts,
        "distribution_gate": distribution_gate,
        "joint_projection_error_mean": (
            sum(sample.joint_projection_error for sample in samples) / len(samples)
            if samples else 0.0
        ),
        "joint_head_recommended": (
            bool(samples)
            and sum(sample.joint_projection_error for sample in samples) / len(samples) > 0.20
        ),
        "workers_requested": config.workers,
        "workers_used": workers,
        "solver": {
            "threads": config.solver_threads,
            "time_limit_seconds": config.time_limit_seconds,
        },
        "resumed_snapshot_count": resumed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**paths, "exclusions": exclusion_path, "manifest": manifest_path}


def _checkpoint_mapping(values: Sequence[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError("checkpoint must use MODEL_SEED=PATH")
        result[int(seed_text)] = Path(path_text)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--c0-checkpoint", action="append", default=[])
    parser.add_argument("--model-seed", type=int, choices=(3101, 3102, 3103), required=True)
    parser.add_argument("--residual-checkpoint", action="append", default=[])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "development", "test"),
        default=("train", "development"),
    )
    parser.add_argument("--unlock-test-checkpoint")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if "test" in args.splits:
        if not args.unlock_test_checkpoint or not Path(args.unlock_test_checkpoint).is_file():
            raise ValueError("test generation requires a fixed checkpoint path")
        _, fixed_payload = load_residual_tail_checkpoint(
            args.unlock_test_checkpoint, device=args.device
        )
        if int(fixed_payload["config"]["seed"]) != args.model_seed:
            raise ValueError("test unlock checkpoint does not match --model-seed")
    c0_available = _checkpoint_mapping(args.c0_checkpoint)
    if args.model_seed not in c0_available:
        raise ValueError("the corresponding --c0-checkpoint MODEL_SEED=PATH is required")
    c0 = {args.model_seed: c0_available[args.model_seed]}
    residual_available = _checkpoint_mapping(args.residual_checkpoint)
    if residual_available and set(residual_available) != {args.model_seed}:
        raise ValueError("DAgger checkpoint must correspond to --model-seed")
    residual = residual_available
    if residual and tuple(args.splits) != ("train",):
        raise ValueError("residual-policy DAgger is restricted to the train split")

    candidates = []
    for split in args.splits:
        for seed in FORMAL_RESIDUAL_SPLIT_PLAN.as_mapping()[split]:
            candidates.extend(collect_seed_snapshots(
                seed,
                c0_checkpoints=c0,
                residual_checkpoints=residual,
                device=args.device,
                max_steps=args.max_steps,
            ))
    paths = generate_sharded_dataset(
        candidates,
        args.output_dir,
        config=ResidualGenerationConfig(
            workers=args.workers,
            time_limit_seconds=args.time_limit,
            max_steps=args.max_steps,
            model_seed=args.model_seed,
        ),
        resume=args.resume,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

