"""Batch generation of MD expert records from offline oracle schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from baselines.md_oracle_types import (
    GurobiOracleResult,
    GurobiOracleStatus,
    OracleScheduleEntry,
)
from baselines.md_oracle_dispatch import solve_md_oracle
from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_expert_dataset import (
    MDExpertRecord,
    generate_md_expert_record_from_oracle,
    save_md_expert_record,
)
from data_generation.md_instance_generator import (
    MDGeneratorConfig,
    generate_md_instance,
)


@dataclass(frozen=True, slots=True)
class MDExpertGenerationConfig:
    """Deterministic batch and oracle settings for Ticket 17."""

    count: int
    seed_start: int = 0
    task_group_prefix: str = "md-family"
    instance_prefix: str = "md-instance"
    task_count: int = 12
    transport_ratio: float = 0.25
    precedence_density: float = 0.25
    critical_path_length: int = 3
    capacity_slack: float = 0.2
    speed_ratio: float = 0.8
    process_robot_count: int = 3
    transport_robot_count: int = 2
    skill_count: int = 3
    scarce_skill_count: int = 0
    material_downstream_stratified: bool = False
    process_duration_range: tuple[int, int] | None = None
    time_limit_seconds: float = 60.0
    threads: int = 1
    max_steps: int = 10_000
    exit_location: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        _positive_integer(self.count, "count")
        _nonnegative_integer(self.seed_start, "seed_start")
        for name in ("task_group_prefix", "instance_prefix"):
            _safe_identifier(getattr(self, name), name)
        _positive_number(self.time_limit_seconds, "time_limit_seconds")
        _positive_integer(self.threads, "threads")
        _positive_integer(self.max_steps, "max_steps")
        if (
            not isinstance(self.exit_location, tuple)
            or len(self.exit_location) != 2
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in self.exit_location
            )
        ):
            raise ValueError("exit_location must contain two finite numbers")
        self.generator_config(self.seed_start)

    def generator_config(self, seed: int) -> MDGeneratorConfig:
        return MDGeneratorConfig(
            seed=seed,
            task_count=self.task_count,
            transport_ratio=self.transport_ratio,
            precedence_density=self.precedence_density,
            critical_path_length=self.critical_path_length,
            capacity_slack=self.capacity_slack,
            speed_ratio=self.speed_ratio,
            process_robot_count=self.process_robot_count,
            transport_robot_count=self.transport_robot_count,
            skill_count=self.skill_count,
            scarce_skill_count=self.scarce_skill_count,
            material_downstream_stratified=self.material_downstream_stratified,
            process_duration_range=self.process_duration_range,
        )


@dataclass(frozen=True, slots=True)
class MDExpertGenerationSkip:
    instance_id: str
    seed: int
    status: str
    message: str


@dataclass(frozen=True, slots=True)
class MDExpertGenerationSummary:
    output_dir: Path
    requested_count: int
    generated_count: int
    skipped: tuple[MDExpertGenerationSkip, ...]
    record_paths: tuple[Path, ...]
    manifest_path: Path | None = None

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "requested_count": self.requested_count,
            "generated_count": self.generated_count,
            "skipped_count": self.skipped_count,
            "records": [str(path) for path in self.record_paths],
            "manifest_path": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
            "skipped": [
                {
                    "instance_id": item.instance_id,
                    "seed": item.seed,
                    "status": item.status,
                    "message": item.message,
                }
                for item in self.skipped
            ],
        }


OracleSolver = Callable[..., GurobiOracleResult]


def generate_md_expert_dataset(
    output_dir: str | Path,
    *,
    config: MDExpertGenerationConfig,
    oracle_solver: OracleSolver = solve_md_oracle,
    overwrite: bool = False,
) -> MDExpertGenerationSummary:
    """Generate one replayed expert record per feasible offline oracle result.

    The solver is deliberately injected for tests and alternative offline
    solver wrappers. Normal use dispatches through
    :func:`baselines.md_oracle_dispatch.solve_md_oracle` (OR-Tools CP-SAT by
    default; Gurobi selectable via ``MRTA_MILP_SOLVER=gurobi``); no solver is
    called during later IL training.
    """

    if not isinstance(config, MDExpertGenerationConfig):
        raise TypeError("config must be an MDExpertGenerationConfig")
    if not callable(oracle_solver):
        raise TypeError("oracle_solver must be callable")
    destination = Path(output_dir)
    manifest_path = _generation_manifest_path(destination, config)
    plans = tuple(
        _generation_plan(destination, config, config.seed_start + index)
        for index in range(config.count)
    )
    if len({path for _, _, _, path in plans}) != len(plans):
        raise ValueError("generation plan contains duplicate output paths")
    conflicts = tuple(
        path for _, _, _, path in plans if path.exists() and not overwrite
    )
    if manifest_path.exists() and not overwrite:
        conflicts = (*conflicts, manifest_path)
    if conflicts:
        raise FileExistsError(
            "expert records already exist: "
            + ", ".join(str(path) for path in conflicts)
        )

    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    skipped: list[MDExpertGenerationSkip] = []
    for seed, instance_id, task_group_id, path in plans:
        generated = generate_md_instance(config.generator_config(seed))
        instance = MDInstanceRecord(
            instance_id=instance_id,
            task_group_id=task_group_id,
            seed=seed,
            generated=generated,
        )
        oracle = oracle_solver(
            generated.domain,
            exit_location=config.exit_location,
            time_limit_seconds=config.time_limit_seconds,
            threads=config.threads,
        )
        if not isinstance(oracle, GurobiOracleResult):
            raise TypeError("oracle_solver must return GurobiOracleResult")
        if oracle.status not in {
            GurobiOracleStatus.OPTIMAL,
            GurobiOracleStatus.FEASIBLE,
        } or not oracle.feasible:
            skipped.append(
                MDExpertGenerationSkip(
                    instance_id,
                    seed,
                    oracle.status.value,
                    oracle.message,
                )
            )
            continue
        schedule_error = _oracle_schedule_error(oracle)
        if schedule_error is not None:
            skipped.append(
                MDExpertGenerationSkip(
                    instance_id,
                    seed,
                    "oracle_schedule_mismatch",
                    schedule_error,
                )
            )
            continue
        try:
            record = generate_md_expert_record_from_oracle(
                instance,
                oracle,
                exit_location=config.exit_location,
                max_steps=config.max_steps,
            )
        except (RuntimeError, ValueError) as error:
            skipped.append(
                MDExpertGenerationSkip(instance_id, seed, "replay_failed", str(error))
            )
            continue
        replay_error = _replay_schedule_error(record, oracle)
        if replay_error is not None:
            skipped.append(
                MDExpertGenerationSkip(
                    instance_id,
                    seed,
                    "replay_schedule_mismatch",
                    replay_error,
                )
            )
            continue
        save_md_expert_record(path, record)
        paths.append(path)

    manifest_path.parent.mkdir(exist_ok=True)
    summary = MDExpertGenerationSummary(
        destination,
        config.count,
        len(paths),
        tuple(skipped),
        tuple(paths),
        manifest_path,
    )
    manifest_payload = {
        "schema_version": "1.0.0",
        "config": asdict(config),
        **summary.to_dict(),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return summary



def _generation_manifest_path(
    destination: Path, config: MDExpertGenerationConfig
) -> Path:
    config_json = json.dumps(
        asdict(config), allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]
    return destination / "_generation" / (
        f"seed-{config.seed_start:06d}-count-{config.count:06d}-{digest}.json"
    )


def _replay_schedule_error(
    record: MDExpertRecord, oracle: GurobiOracleResult
) -> str | None:
    source_by_task: dict[int, list[OracleScheduleEntry]] = {}
    for entry in oracle.schedule:
        source_by_task.setdefault(entry.task_id, []).append(entry)
    execution = record.terminal_result["execution_records"]
    actual_by_task = {
        item["task_id"]: ("process", item) for item in execution["process"]
    }
    actual_by_task.update(
        (item["task_id"], ("transport", item))
        for item in execution["transport"]
    )
    if source_by_task.keys() != actual_by_task.keys():
        return "source schedule and replay contain different task IDs"

    for task_id, entries in source_by_task.items():
        starts = {entry.start for entry in entries}
        completions = {entry.completion for entry in entries}
        if len(starts) != 1 or len(completions) != 1:
            return f"source schedule disagrees within task {task_id}"
        kind, actual = actual_by_task[task_id]
        scheduled_robots = {entry.robot_id for entry in entries}
        if kind == "process":
            actual_robots = set(actual["robot_ids"])
            actual_start = actual["started_at"]
        else:
            actual_robots = {actual["robot_id"]}
            actual_start = actual["loading_started_at"]
        if scheduled_robots != actual_robots:
            return f"source and replay robot assignment differ for task {task_id}"
        actual_completion = actual["completed_at"]
        expected_start = next(iter(starts))
        expected_completion = next(iter(completions))
        if not math.isclose(
            float(actual_start), expected_start, rel_tol=0.0, abs_tol=1e-6
        ):
            return f"source and replay start differ for task {task_id}"
        if not math.isclose(
            float(actual_completion),
            expected_completion,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            return f"source and replay completion differ for task {task_id}"
    return None
def _generation_plan(
    destination: Path,
    config: MDExpertGenerationConfig,
    seed: int,
) -> tuple[int, str, str, Path]:
    instance_id = f"{config.instance_prefix}-seed-{seed:06d}"
    task_group_id = f"{config.task_group_prefix}-seed-{seed:06d}"
    path = destination / f"{instance_id}.json"
    resolved_destination = destination.resolve()
    if path.resolve().parent != resolved_destination:
        raise ValueError("expert record path must remain inside output_dir")
    return seed, instance_id, task_group_id, path


def _oracle_schedule_error(oracle: GurobiOracleResult) -> str | None:
    if not oracle.schedule:
        return "feasible oracle result has no source schedule"
    actions = {(item.robot_id, item.task_id): item for item in oracle.action_order}
    schedule = {(item.robot_id, item.task_id): item for item in oracle.schedule}
    if len(actions) != len(oracle.action_order):
        return "oracle actions contain duplicate robot-task pairs"
    if len(schedule) != len(oracle.schedule):
        return "oracle schedule contains duplicate robot-task pairs"
    if actions.keys() != schedule.keys():
        return "oracle actions and schedule contain different robot-task pairs"
    for pair, entry in schedule.items():
        action = actions[pair]
        values = (entry.start, entry.completion, action.planned_start)
        if any(not math.isfinite(value) for value in values):
            return "oracle schedule contains non-finite timing"
        if entry.completion < entry.start:
            return "oracle schedule completion precedes start"
        if not math.isclose(
            action.planned_start, entry.start, rel_tol=0.0, abs_tol=1e-6
        ):
            return "oracle action planned_start differs from source schedule"
        if action.planned_assignment > action.planned_start + 1e-6:
            return "oracle assignment occurs after its planned task start"
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate replayed MD expert records with the offline Gurobi oracle"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--task-group-prefix", default="md-family")
    parser.add_argument("--instance-prefix", default="md-instance")
    parser.add_argument("--task-count", type=int, default=12)
    parser.add_argument("--transport-ratio", type=float, default=0.25)
    parser.add_argument("--precedence-density", type=float, default=0.25)
    parser.add_argument("--critical-path-length", type=int, default=3)
    parser.add_argument("--capacity-slack", type=float, default=0.2)
    parser.add_argument("--speed-ratio", type=float, default=0.8)
    parser.add_argument("--process-robot-count", type=int, default=3)
    parser.add_argument("--transport-robot-count", type=int, default=2)
    parser.add_argument("--skill-count", type=int, default=3)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--exit-x", type=float, default=0.0)
    parser.add_argument("--exit-y", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = MDExpertGenerationConfig(
        count=args.count,
        seed_start=args.seed_start,
        task_group_prefix=args.task_group_prefix,
        instance_prefix=args.instance_prefix,
        task_count=args.task_count,
        transport_ratio=args.transport_ratio,
        precedence_density=args.precedence_density,
        critical_path_length=args.critical_path_length,
        capacity_slack=args.capacity_slack,
        speed_ratio=args.speed_ratio,
        process_robot_count=args.process_robot_count,
        transport_robot_count=args.transport_robot_count,
        skill_count=args.skill_count,
        time_limit_seconds=args.time_limit_seconds,
        threads=args.threads,
        max_steps=args.max_steps,
        exit_location=(args.exit_x, args.exit_y),
    )
    summary = generate_md_expert_dataset(
        args.output_dir, config=config, overwrite=args.overwrite
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return _generation_exit_code(summary, allow_partial=args.allow_partial)


def _generation_exit_code(
    summary: MDExpertGenerationSummary, *, allow_partial: bool
) -> int:
    if summary.generated_count == summary.requested_count:
        return 0
    if allow_partial and summary.generated_count > 0:
        return 0
    return 2


def _safe_identifier(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None
    ):
        raise ValueError(
            f"{name} must use only letters, digits, underscores, or hyphens"
        )


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MDExpertGenerationConfig",
    "MDExpertGenerationSkip",
    "MDExpertGenerationSummary",
    "generate_md_expert_dataset",
]
