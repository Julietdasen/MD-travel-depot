"""Diagnose how much coalition is actually forced in process_scarce instances.

For each (tc, pr, scarce_skill_count) config:
  - required_skills_per_task: mean number of skills each process task requires
  - solo_capable_ratio: fraction of process tasks that at least one process robot
    can complete alone (union of its capabilities covers all requirements)
  - min_coalition_size: for each task, the minimum robot count that covers its
    requirements (set-cover heuristic); we report the distribution
  - coalition_forced_ratio: fraction of tasks with min_coalition_size >= 2

A "strong coalition" regime means most tasks require >= 2 robots to be legally
executed (i.e., no single robot has all required skills). If the ratio is low,
coalition is optional — greedy singleton-assignment often works.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from data_generation.md_instance_generator import (
    MDGeneratorConfig,
    generate_md_instance,
)
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from simulation_environment.domain_model import ProcessTask, ProcessRobot


def _min_cover_size(required: tuple[bool, ...], caps: list[tuple[bool, ...]]) -> int:
    req = [i for i, v in enumerate(required) if v]
    if not req:
        return 0
    for k in range(1, len(caps) + 1):
        for combo in combinations(range(len(caps)), k):
            covered = set()
            for r in combo:
                for i, v in enumerate(caps[r]):
                    if v:
                        covered.add(i)
            if all(i in covered for i in req):
                return k
    return len(caps) + 1


def _analyze(cfg: MDGeneratorConfig) -> dict:
    inst = generate_md_instance(cfg)
    domain = inst.domain
    process_robots = [r for r in domain.robots if isinstance(r, ProcessRobot)]
    process_tasks = [t for t in domain.tasks if isinstance(t, ProcessTask)]
    caps = [r.capabilities for r in process_robots]

    task_reqs = [t.requirements for t in process_tasks]
    n_req = [sum(r) for r in task_reqs]
    min_sizes = [_min_cover_size(r, caps) for r in task_reqs]
    solo_capable = sum(1 for s in min_sizes if s == 1)
    coalition_forced = sum(1 for s in min_sizes if s >= 2)
    return {
        "n_process_tasks": len(process_tasks),
        "n_process_robots": len(process_robots),
        "skill_count": cfg.skill_count,
        "scarce_skill_count": cfg.scarce_skill_count,
        "mean_required_skills": sum(n_req) / len(n_req) if n_req else 0,
        "solo_capable_ratio": solo_capable / len(min_sizes) if min_sizes else 0,
        "coalition_forced_ratio": coalition_forced / len(min_sizes) if min_sizes else 0,
        "min_coalition_dist": dict(Counter(min_sizes)),
    }


def _scaled_cfg(profile: str, tc: int, seed: int, override: dict) -> MDGeneratorConfig:
    base = dict(INSTANCE_PROFILES[profile].parameters)
    base_tc = base["task_count"]
    scale = tc / base_tc
    from data_generation.md_instance_generator import MDGeneratorConfig as _C
    cfg = _C(
        seed=seed,
        task_count=tc,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=min(
            max(base["critical_path_length"], round(base["critical_path_length"] * scale)),
            int(tc * (1 - base["transport_ratio"])),
        ),
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=max(1, round(base["process_robot_count"] * scale)),
        transport_robot_count=max(base.get("transport_robot_count", 1), round(base["transport_robot_count"] * scale)),
        skill_count=base["skill_count"],
        scarce_skill_count=base.get("scarce_skill_count", 0),
    )
    if override:
        from dataclasses import replace
        cfg = replace(cfg, **override)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="process_scarce")
    p.add_argument("--tc-list", default="12,24,42,60")
    p.add_argument("--seeds", default="301,302,303,304")
    p.add_argument("--override-scarce", type=int, default=None)
    p.add_argument("--override-pr", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    tcs = [int(x) for x in args.tc_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    override = {}
    if args.override_scarce is not None:
        override["scarce_skill_count"] = args.override_scarce
    if args.override_pr is not None:
        override["process_robot_count"] = args.override_pr

    rows = []
    for tc in tcs:
        for seed in seeds:
            cfg = _scaled_cfg(args.profile, tc, seed, override)
            r = _analyze(cfg)
            r.update({"tc": tc, "seed": seed, "profile": args.profile})
            rows.append(r)

    # aggregate per tc
    agg = {}
    for tc in tcs:
        sub = [r for r in rows if r["tc"] == tc]
        agg[tc] = {
            "mean_required_skills": sum(r["mean_required_skills"] for r in sub) / len(sub),
            "solo_capable_ratio": sum(r["solo_capable_ratio"] for r in sub) / len(sub),
            "coalition_forced_ratio": sum(r["coalition_forced_ratio"] for r in sub) / len(sub),
            "n_process_robots": sub[0]["n_process_robots"],
            "n_process_tasks_mean": sum(r["n_process_tasks"] for r in sub) / len(sub),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "agg.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
