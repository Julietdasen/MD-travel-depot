"""Gantt-chart dump for scarce=3 envelope.

For each (checkpoint_seed, task_count) in the stability x scarce=3 grid we
re-run the C0 MILP-IL v2 online scheduler on a fixed problem seed, pull
per-task start/end/robot data out of the ExperimentResult, and render
a per-robot Gantt chart. Process tasks that require a scarce skill (a
skill owned by exactly one process robot) are highlighted so we can see
whether the schedule serialises around the bottleneck.

Outputs land under
  reports/md_gantt_scarce3_2026-09-21/seed{s}/tc{tc}_pr{pr}/gantt.png
plus a schedule.json with the raw records.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import load_md_policy_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    OnlineMDScheduler,
    OnlineNeuralScoreProvider,
)
from simulation_environment.domain_model import ProcessRobot, ProcessTask

MAX_ROLLOUT_STEPS = 20_000


def _scaled_process_scarce_config(task_count, problem_seed, scarce_skill_count,
                                  process_robot_count, transport_robot_count) -> MDGeneratorConfig:
    base = dict(INSTANCE_PROFILES["process_scarce"].parameters)
    base_task_count = base["task_count"]
    scale = task_count / base_task_count
    process_tasks = int(task_count * (1 - base["transport_ratio"]))
    critical_path = min(
        max(base["critical_path_length"], round(base["critical_path_length"] * scale)),
        process_tasks,
    )
    return MDGeneratorConfig(
        seed=problem_seed,
        task_count=task_count,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=critical_path,
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=process_robot_count,
        transport_robot_count=transport_robot_count,
        skill_count=base["skill_count"],
        scarce_skill_count=scarce_skill_count,
        material_downstream_stratified=base.get("material_downstream_stratified", False),
        process_duration_range=base.get("process_duration_range", None),
    )


def _run_c0(model, domain, *, run_id, instance_id, seed, device):
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device=device),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=1),
        confidence_threshold=0.0,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    started = time.perf_counter()
    result = scheduler.run(domain, run_id=run_id, instance_id=instance_id, seed=seed,
                          split=DatasetSplit.TEST)
    elapsed = time.perf_counter() - started
    return result.experiment, elapsed


def _scarce_skill_ids(domain) -> set[int]:
    counts: dict[int, int] = {}
    for robot in domain.robots:
        if not isinstance(robot, ProcessRobot):
            continue
        for skill_idx, has in enumerate(robot.capabilities):
            if has:
                counts[skill_idx] = counts.get(skill_idx, 0) + 1
    return {sid for sid, c in counts.items() if c == 1}


def _plot(exp_result, domain, out_path: Path, title: str) -> None:
    process_records = exp_result.process_execution_records
    transport_records = exp_result.transport_execution_records
    makespan = exp_result.makespan

    process_robots = [r for r in domain.robots if isinstance(r, ProcessRobot)]
    transport_robots = [r for r in domain.robots if not isinstance(r, ProcessRobot)]
    robot_rows: dict[int, int] = {}
    labels: list[str] = []
    for r in process_robots:
        robot_rows[r.robot_id] = len(labels)
        caps = [i for i, c in enumerate(r.capabilities) if c]
        labels.append(f"P{r.robot_id} skills={caps}")
    for r in transport_robots:
        robot_rows[r.robot_id] = len(labels)
        labels.append(f"T{r.robot_id}")

    scarce_skills = _scarce_skill_ids(domain)
    task_by_id = {t.task_id: t for t in domain.tasks}

    fig, ax = plt.subplots(figsize=(max(10, makespan / 25), max(3, 0.4 * len(labels) + 1.2)))

    for rec in process_records:
        task_id = rec["task_id"]
        started = rec["started_at"]
        completed = rec["completed_at"]
        if completed is None:
            continue
        duration = completed - started
        task = task_by_id[task_id]
        assert isinstance(task, ProcessTask)
        req_scarce = any(task.requirements[sid] for sid in scarce_skills)
        color = "#d62728" if req_scarce else "#1f77b4"
        for robot_id in rec["robot_ids"]:
            row = robot_rows[robot_id]
            ax.barh(row, duration, left=started, height=0.7, color=color,
                    edgecolor="black", linewidth=0.4)
            ax.text(started + duration / 2, row, f"P{task_id}",
                    ha="center", va="center", fontsize=6, color="white")

    for rec in transport_records:
        task_id = rec["task_id"]
        robot_id = rec.get("robot_id")
        if robot_id is None:
            continue
        assigned = rec.get("assigned_at")
        arr_pickup = rec.get("arrived_pickup_at")
        loading_s = rec.get("loading_started_at")
        loading_e = rec.get("loading_completed_at")
        arr_delivery = rec.get("arrived_delivery_at")
        unload_s = rec.get("unloading_started_at")
        unload_e = rec.get("unloading_completed_at")
        row = robot_rows[robot_id]
        # empty travel (assigned -> pickup)
        if assigned is not None and arr_pickup is not None and arr_pickup > assigned:
            ax.barh(row, arr_pickup - assigned, left=assigned, height=0.45,
                    color="#e0e0e0", edgecolor="black", linewidth=0.2)
        if loading_s is not None and loading_e is not None:
            ax.barh(row, loading_e - loading_s, left=loading_s, height=0.55,
                    color="#ffbb78", edgecolor="black", linewidth=0.3)
        # loaded travel (loading_completed -> arrived_delivery)
        if loading_e is not None and arr_delivery is not None and arr_delivery > loading_e:
            ax.barh(row, arr_delivery - loading_e, left=loading_e, height=0.45,
                    color="#c49c74", edgecolor="black", linewidth=0.2)
        if unload_s is not None and unload_e is not None:
            ax.barh(row, unload_e - unload_s, left=unload_s, height=0.55,
                    color="#ff7f0e", edgecolor="black", linewidth=0.3)
            ax.text((unload_s + unload_e) / 2, row, f"T{task_id}",
                    ha="center", va="center", fontsize=6, color="black")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time (discrete)")
    ax.set_xlim(0, makespan * 1.02)
    ax.axvline(makespan, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(makespan, -0.5, f" makespan={makespan}", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)

    handles = [
        mpatches.Patch(color="#d62728", label="process task, needs scarce skill"),
        mpatches.Patch(color="#1f77b4", label="process task, common skill"),
        mpatches.Patch(color="#e0e0e0", label="transport: empty travel"),
        mpatches.Patch(color="#ffbb78", label="transport: loading"),
        mpatches.Patch(color="#c49c74", label="transport: loaded travel"),
        mpatches.Patch(color="#ff7f0e", label="transport: unloading"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/md_gantt_scarce3_2026-09-21"))
    parser.add_argument("--problem-seed", type=int, default=301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("reports/md_c0_pathC_v1_stability_2026-09-18"))
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3101, 3102, 3103])
    parser.add_argument("--scales", nargs="+", default=["24,4,4", "42,7,7", "60,10,10"])
    parser.add_argument("--scarce-skill-count", type=int, default=3)
    args = parser.parse_args()

    scales = []
    for s in args.scales:
        tc, pr, tr = (int(x) for x in s.split(","))
        scales.append((tc, pr, tr))

    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    index = []
    for seed_ckpt in args.seeds:
        ckpt = args.checkpoint_root / f"seed{seed_ckpt}" / "training" / f"checkpoint_epoch_{args.epoch}.pt"
        if not ckpt.exists():
            print(f"skip seed={seed_ckpt}: {ckpt} not found")
            continue
        print(f"loading {ckpt}", flush=True)
        model, _ = load_md_policy_checkpoint(ckpt, device=device)
        model.eval()

        for tc, pr, tr in scales:
            cfg = _scaled_process_scarce_config(tc, args.problem_seed, args.scarce_skill_count, pr, tr)
            domain = generate_md_instance(cfg).domain

            run_id = f"gantt-{seed_ckpt}-{tc}"
            exp, wall = _run_c0(model, domain,
                                run_id=run_id,
                                instance_id=f"scaled-process_scarce-{tc}-{args.problem_seed}",
                                seed=args.problem_seed, device=device)

            out_dir = args.output_root / f"seed{seed_ckpt}" / f"tc{tc}_pr{pr}"
            title = (f"C0 seed{seed_ckpt} epoch{args.epoch} — process_scarce "
                     f"tc={tc} pr={pr} tr={tr} scarce=3 "
                     f"(problem seed {args.problem_seed}) makespan={exp.makespan}")
            _plot(exp, domain, out_dir / "gantt.png", title)

            record = {
                "seed_ckpt": seed_ckpt,
                "tc": tc, "pr": pr, "tr": tr,
                "problem_seed": args.problem_seed,
                "makespan": exp.makespan,
                "wall_time_seconds": wall,
                "process_records": list(exp.process_execution_records),
                "transport_records": list(exp.transport_execution_records),
            }
            (out_dir / "schedule.json").write_text(json.dumps(record, indent=2, default=list))
            index.append({"seed_ckpt": seed_ckpt, "tc": tc, "pr": pr,
                          "makespan": exp.makespan,
                          "gantt": str(out_dir / "gantt.png"),
                          "schedule": str(out_dir / "schedule.json")})
            print(f"  tc={tc} pr={pr}: makespan={exp.makespan} wall={wall:.1f}s -> {out_dir}",
                  flush=True)

    (args.output_root / "index.json").write_text(json.dumps(index, indent=2))
    print("done ->", args.output_root)


if __name__ == "__main__":
    main()
