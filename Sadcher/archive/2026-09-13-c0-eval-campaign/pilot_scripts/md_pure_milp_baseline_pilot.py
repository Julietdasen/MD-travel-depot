"""Pure end-to-end MILP (Gurobi) baseline for the six MD instance profiles
and the scale-up ladder used by md_c0_scale_baseline_comparison_pilot.

The MIP fallback used by C0 solves a *per-decision* MIP over the current
step's assignment only. This runs the full end-to-end MILP formulation
(baselines.gurobi_md_oracle) on the same instances so we can see how a
pure exact-optimization baseline compares to C0 and the greedy heuristics
on makespan and on solver wall time, and where the exact formulation
starts to time out at scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.gurobi_md_oracle import GurobiOracleStatus, solve_gurobi_md_oracle
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_instance_profiles import INSTANCE_PROFILES

PROFILE_SEEDS = (101, 102, 201, 301)
SCALE_TASK_COUNTS = (12, 24, 42, 60, 90, 114, 150)
SCALE_SEEDS = (9601, 9602, 9603)
DEFAULT_TIME_LIMIT_SECONDS = 300.0


def _scaled_config(task_count: int, seed: int) -> MDGeneratorConfig:
    scale = task_count / 18
    return MDGeneratorConfig(
        seed=seed,
        task_count=task_count,
        transport_ratio=1.0 / 3.0,
        precedence_density=0.30,
        critical_path_length=6,
        capacity_slack=0.10,
        speed_ratio=0.70,
        process_robot_count=max(1, round(4 * scale)),
        transport_robot_count=max(1, round(3 * scale)),
        skill_count=3,
    )


def _run_one(domain, time_limit_seconds: float, threads: int) -> dict:
    result = solve_gurobi_md_oracle(
        domain,
        time_limit_seconds=time_limit_seconds,
        threads=threads,
    )
    makespan = None
    if result.schedule:
        makespan = max(entry.completion for entry in result.schedule)
    return {
        "status": result.status.value,
        "feasible": result.feasible,
        "timeout": result.timeout,
        "solve_time_seconds": result.solve_time_seconds,
        "objective": result.objective,
        "optimality_gap": result.optimality_gap,
        "makespan": makespan,
        "message": result.message,
    }


def run(output: Path, time_limit_seconds: float, threads: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    profile_rows: list[dict] = []
    for profile_name, profile in INSTANCE_PROFILES.items():
        for seed in PROFILE_SEEDS:
            domain = generate_md_instance(profile.config(seed)).domain
            row = {"profile": profile_name, "seed": seed, **_run_one(domain, time_limit_seconds, threads)}
            profile_rows.append(row)
            print(json.dumps(row), flush=True)

    scale_rows: list[dict] = []
    for task_count in SCALE_TASK_COUNTS:
        for seed in SCALE_SEEDS:
            domain = generate_md_instance(_scaled_config(task_count, seed)).domain
            row = {"task_count": task_count, "seed": seed, **_run_one(domain, time_limit_seconds, threads)}
            scale_rows.append(row)
            print(json.dumps(row), flush=True)

    (output / "profile_rows.json").write_text(json.dumps(profile_rows, indent=2) + "\n")
    (output / "scale_rows.json").write_text(json.dumps(scale_rows, indent=2) + "\n")

    lines = [
        f"# Pure MILP baseline (time_limit={time_limit_seconds}s, threads={threads})",
        "",
        "## Six named profiles",
        "",
        "| profile | n | optimal | feasible-only | timeout | mean solve time (s) | mean makespan |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile_name in INSTANCE_PROFILES:
        subset = [r for r in profile_rows if r["profile"] == profile_name]
        optimal = [r for r in subset if r["status"] == "optimal"]
        feasible_only = [
            r for r in subset if r["status"] == "feasible" or (r["status"] == "timeout" and r["makespan"] is not None)
        ]
        timeout = [r for r in subset if r["timeout"]]
        mean_solve = sum(r["solve_time_seconds"] for r in subset) / len(subset)
        makespans = [r["makespan"] for r in subset if r["makespan"] is not None]
        mean_ms = sum(makespans) / len(makespans) if makespans else "NA"
        lines.append(
            f"| {profile_name} | {len(subset)} | {len(optimal)} | {len(feasible_only)} | "
            f"{len(timeout)} | {mean_solve:.3f} | {mean_ms} |"
        )

    lines += [
        "",
        "## Scale ladder",
        "",
        "| task_count | n | optimal | feasible-only | timeout | mean solve time (s) | mean makespan |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_count in SCALE_TASK_COUNTS:
        subset = [r for r in scale_rows if r["task_count"] == task_count]
        optimal = [r for r in subset if r["status"] == "optimal"]
        feasible_only = [
            r for r in subset if r["status"] == "feasible" or (r["status"] == "timeout" and r["makespan"] is not None)
        ]
        timeout = [r for r in subset if r["timeout"]]
        mean_solve = sum(r["solve_time_seconds"] for r in subset) / len(subset)
        makespans = [r["makespan"] for r in subset if r["makespan"] is not None]
        mean_ms = sum(makespans) / len(makespans) if makespans else "NA"
        lines.append(
            f"| {task_count} | {len(subset)} | {len(optimal)} | {len(feasible_only)} | "
            f"{len(timeout)} | {mean_solve:.3f} | {mean_ms} |"
        )

    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-pure-milp-baseline-1.0",
        "time_limit_seconds": time_limit_seconds,
        "threads": threads,
        "profile_row_count": len(profile_rows),
        "scale_row_count": len(scale_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/md_pure_milp_baseline_pilot_2026-09-13"))
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.time_limit, args.threads), indent=2))
