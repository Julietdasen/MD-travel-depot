"""Run an isolated, profile-selected synthetic MD instance."""

from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.md_dataset import MDInstanceRecord, save_md_instance
from data_generation.md_instance_generator import generate_md_instance
from experiments.md_scale_profiles import (
    DEFAULT_PROFILE_PATH,
    generator_config_for_profile,
    load_scale_profiles,
    profile_summary,
)
from schedulers.md_greedy_baselines import run_material_solo_greedy
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.md_static_validator import require_valid_md_domain


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "reports" / "md_scaling_runs"
MAX_ROLLOUT_STEPS = 100_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="scale profile name")
    parser.add_argument("--seed", type=int, help="override the profile seed")
    parser.add_argument("--scheduler", choices=("greedy", "sadcher"), default="greedy")
    parser.add_argument("--checkpoint", type=Path, help="modern Sadcher checkpoint")
    parser.add_argument("--profile-file", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--list-profiles", action="store_true")
    return parser


def _run(config, *, profile: str, scheduler: str, checkpoint: Path | None, output: Path) -> None:
    if scheduler == "sadcher":
        raise NotImplementedError(
            "modern Sadcher requires a checkpoint and model-specific scorer wiring; "
            "use the existing MD policy experiment runner or --scheduler greedy for this CLI"
        )
    generated = generate_md_instance(config)
    require_valid_md_domain(generated.domain)
    instance_id = f"md-scale-{profile}-seed-{config.seed}"
    task_group_id = f"md-scale-{profile}"
    instance = MDInstanceRecord(
        instance_id=instance_id,
        task_group_id=task_group_id,
        seed=config.seed,
        generated=generated,
    )
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / profile / f"seed-{config.seed}" / scheduler
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {run_dir}")
    run_dir.mkdir(parents=True)
    save_md_instance(run_dir / "instance.json", instance)
    summary = profile_summary(profile, config)
    (run_dir / "config.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    result = run_material_solo_greedy(
        MDDiscreteSimulator(generated.domain),
        run_id=f"{instance_id}-{scheduler}",
        instance_id=instance_id,
        seed=config.seed,
        split=instance.split,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    payload = result.to_dict()
    payload["profile"] = summary
    payload["wall_time_seconds_cli"] = time.perf_counter() - started
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "rollout.json").write_text(
        json.dumps(payload["execution_records"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"profile={profile}\nscheduler={scheduler}\nseed={config.seed}\n"
        f"success={result.success}\nmakespan={result.makespan}\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), **summary, "success": result.success}, indent=2, sort_keys=True))


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    profiles = load_scale_profiles(args.profile_file)
    if args.list_profiles:
        for name in sorted(profiles):
            config = generator_config_for_profile(name, path=args.profile_file)
            summary = profile_summary(name, config)
            print(f"{name}: tasks={summary['task_count']} robots={summary['robot_count']} "
                  f"(process={summary['process_robot_count']}, transport={summary['transport_robot_count']}) "
                  f"default_seed={summary['seed']}")
        return 0
    if args.profile is None:
        parser.error("--profile is required unless --list-profiles is used")
    config = generator_config_for_profile(args.profile, seed=args.seed, path=args.profile_file)
    print(f"Running profile={args.profile} tasks={config.task_count} robots="
          f"{config.process_robot_count + config.transport_robot_count} seed={config.seed}")
    if args.profile == "large":
        print("Warning: large profile may take substantially longer to roll out.")
    _run(config, profile=args.profile, scheduler=args.scheduler, checkpoint=args.checkpoint, output=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
