"""Depot+travel MRTA gap analysis: C0 IL vs MILP.

Reruns two representative instances (tight-gap and wide-gap) with both
schedulers, captures per-task execution records, and produces:
  - Gantt PNGs (C0 vs MILP, side-by-side per robot)
  - Numeric breakdown (per-robot travel/service/idle) as JSON
  - Textual failure-mode notes
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from baselines.md_oracle_dispatch import solve_md_oracle
from baselines.md_oracle_types import replay_oracle_actions
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
from simulation_environment.domain_model import ProcessRobot, ProcessTask, TransportRobot
from simulation_environment.transport_timing import travel_duration


CKPT = Path("reports/md_c0_pathA_2026-09-21-depot/training/best_checkpoint.pt")
OUT = Path("reports/depot_travel_gap_analysis")
MAX_ROLLOUT_STEPS = 20_000


def scaled_cfg(profile: str, tc: int, seed: int) -> MDGeneratorConfig:
    base = dict(INSTANCE_PROFILES[profile].parameters)
    base_tc = base["task_count"]
    scale = tc / base_tc
    ptasks = int(tc * (1 - base["transport_ratio"]))
    cp = min(max(base["critical_path_length"], round(base["critical_path_length"] * scale)), ptasks)
    return MDGeneratorConfig(
        seed=seed, task_count=tc,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=cp,
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=max(1, round(base["process_robot_count"] * scale)),
        transport_robot_count=max(base.get("transport_robot_count", 1),
                                  round(base["transport_robot_count"] * scale)),
        skill_count=base["skill_count"],
        scarce_skill_count=base.get("scarce_skill_count", 0),
        material_downstream_stratified=base.get("material_downstream_stratified", False),
        process_duration_range=base.get("process_duration_range", None),
    )


def run_c0(model, domain, *, run_id, instance_id, seed, device):
    sched = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device=device),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=1),
        confidence_threshold=0.0,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    t0 = time.perf_counter()
    r = sched.run(domain, run_id=run_id, instance_id=instance_id,
                  seed=seed, split=DatasetSplit.TEST)
    return r.experiment, time.perf_counter() - t0


def run_milp(domain, *, run_id, instance_id, seed):
    r = solve_md_oracle(domain, time_limit_seconds=60.0, threads=4)
    exp = replay_oracle_actions(
        domain, r.action_order,
        run_id=run_id, instance_id=instance_id,
        seed=seed, split=DatasetSplit.TEST, max_steps=MAX_ROLLOUT_STEPS,
    )
    return exp, r


def per_robot_breakdown(exp, domain):
    """Compute per-robot travel/service/idle breakdown from execution records."""
    robots = {r.robot_id: r for r in domain.robots}
    tasks = {t.task_id: t for t in domain.tasks}
    depots = {r.robot_id: r.location for r in domain.robots}

    # Aggregate per robot
    ms = exp.makespan
    breakdown = {}
    for rid, robot in robots.items():
        service = 0
        travel = 0
        breakdown[rid] = {
            "role": "P" if isinstance(robot, ProcessRobot) else "T",
            "service_time": 0,
            "travel_time": 0,
            "wait_time": 0,   # arrived at task but coalition/precursor not ready
            "idle_time": 0,
            "makespan": ms,
        }

    # Process robots: per-task, we need to know
    #  - travel from prev location to task location
    #  - wait until service starts
    #  - service duration
    # We reconstruct per-robot timeline from process_records (only for process robots)
    proc_records = list(exp.process_execution_records)
    trans_records = list(exp.transport_execution_records)

    # Per-robot timeline: list of (start, end, kind)
    timelines = {rid: [] for rid in robots}

    # For process robots, sort assignments by started_at
    for rec in proc_records:
        for rid in rec["robot_ids"]:
            timelines[rid].append({
                "task_id": rec["task_id"],
                "started_at": rec["started_at"],
                "completed_at": rec["completed_at"] if rec["completed_at"] is not None else ms,
                "service_duration": rec["service_duration"],
                "waiting_duration": rec["waiting_duration"],
                "kind": "process",
            })

    for rec in trans_records:
        rid = rec["robot_id"]
        if rid is None:
            continue
        timelines[rid].append({
            "task_id": rec["task_id"],
            "assigned_at": rec["assigned_at"],
            "arrived_pickup_at": rec["arrived_pickup_at"],
            "loading_started_at": rec["loading_started_at"],
            "loading_completed_at": rec["loading_completed_at"],
            "arrived_delivery_at": rec["arrived_delivery_at"],
            "unloading_started_at": rec["unloading_started_at"],
            "unloading_completed_at": rec["unloading_completed_at"],
            "empty_travel_duration": rec["empty_travel_duration"],
            "loading_duration": rec["loading_duration"],
            "loaded_travel_duration": rec["loaded_travel_duration"],
            "unloading_duration": rec["unloading_duration"],
            "kind": "transport",
        })

    # For process robots: sort by started_at (service start)
    for rid, robot in robots.items():
        if isinstance(robot, ProcessRobot):
            timelines[rid].sort(key=lambda e: e["started_at"])
            speed = robot.speed
            prev_loc = depots[rid]
            prev_end = 0
            total_service = 0
            total_travel = 0
            total_wait = 0
            for e in timelines[rid]:
                task = tasks[e["task_id"]]
                assert isinstance(task, ProcessTask)
                td = travel_duration(prev_loc, task.location, speed)
                # The robot could have been idle between prev_end and (started_at - td)
                # then travelled td, then waited for coalition until started_at.
                total_travel += td
                total_service += e["service_duration"]
                # wait = simulator "waiting_duration" already accounts for time from
                # earliest assignment to start; but we want to split cleanly.
                # Approximate: gap = started_at - prev_end - td
                # Anything before travel = idle; wait_at_task = waiting_duration from record.
                # We'll trust waiting_duration for the coalition wait.
                total_wait += e["waiting_duration"]
                prev_loc = task.location
                prev_end = e["completed_at"]
            # Return-to-home travel
            home = robot.home_location or (0.0, 0.0)
            return_td = travel_duration(prev_loc, home, speed)
            total_travel += return_td
            breakdown[rid]["service_time"] = total_service
            breakdown[rid]["travel_time"] = total_travel
            breakdown[rid]["wait_time"] = total_wait
            # idle = makespan - service - travel - wait, floored at 0
            breakdown[rid]["idle_time"] = max(0, ms - total_service - total_travel - total_wait)
        else:
            timelines[rid].sort(key=lambda e: e["assigned_at"])
            total_service = 0
            total_travel = 0
            for e in timelines[rid]:
                total_travel += e["empty_travel_duration"] + e["loaded_travel_duration"]
                total_service += e["loading_duration"] + e["unloading_duration"]
            breakdown[rid]["service_time"] = total_service
            breakdown[rid]["travel_time"] = total_travel
            breakdown[rid]["idle_time"] = max(0, ms - total_service - total_travel)

    return breakdown, timelines


def plot_gantt(exp, domain, title, out_png):
    process_robots = [r for r in domain.robots if isinstance(r, ProcessRobot)]
    transport_robots = [r for r in domain.robots if isinstance(r, TransportRobot)]
    robot_rows = {}
    labels = []
    for r in process_robots:
        robot_rows[r.robot_id] = len(labels)
        caps = [i for i, c in enumerate(r.capabilities) if c]
        labels.append(f"P{r.robot_id} skills={caps}")
    for r in transport_robots:
        robot_rows[r.robot_id] = len(labels)
        labels.append(f"T{r.robot_id}")

    ms = exp.makespan
    fig, ax = plt.subplots(figsize=(max(10, ms / 25), max(3, 0.4 * len(labels) + 1.5)))

    for rec in exp.process_execution_records:
        task_id = rec["task_id"]
        started = rec["started_at"]
        completed = rec["completed_at"]
        if completed is None:
            continue
        for rid in rec["robot_ids"]:
            row = robot_rows[rid]
            ax.barh(row, completed - started, left=started, height=0.7,
                    color="#1f77b4", edgecolor="black", linewidth=0.4)
            ax.text(started + (completed - started) / 2, row, f"P{task_id}",
                    ha="center", va="center", fontsize=6, color="white")

    for rec in exp.transport_execution_records:
        rid = rec.get("robot_id")
        if rid is None:
            continue
        row = robot_rows[rid]
        a = rec.get("assigned_at")
        ap = rec.get("arrived_pickup_at")
        ls_, le_ = rec.get("loading_started_at"), rec.get("loading_completed_at")
        ad = rec.get("arrived_delivery_at")
        us, ue = rec.get("unloading_started_at"), rec.get("unloading_completed_at")
        if a is not None and ap is not None and ap > a:
            ax.barh(row, ap - a, left=a, height=0.45, color="#e0e0e0",
                    edgecolor="black", linewidth=0.2)
        if ls_ is not None and le_ is not None:
            ax.barh(row, le_ - ls_, left=ls_, height=0.55, color="#ffbb78",
                    edgecolor="black", linewidth=0.3)
        if le_ is not None and ad is not None and ad > le_:
            ax.barh(row, ad - le_, left=le_, height=0.45, color="#c49c74",
                    edgecolor="black", linewidth=0.2)
        if us is not None and ue is not None:
            ax.barh(row, ue - us, left=us, height=0.55, color="#ff7f0e",
                    edgecolor="black", linewidth=0.3)
            ax.text((us + ue) / 2, row, f"T{rec['task_id']}", ha="center",
                    va="center", fontsize=6, color="black")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time (discrete)")
    ax.set_xlim(0, ms * 1.02)
    ax.axvline(ms, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(ms, -0.5, f" makespan={ms}", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    handles = [
        mpatches.Patch(color="#1f77b4", label="process task service"),
        mpatches.Patch(color="#e0e0e0", label="transport: empty travel"),
        mpatches.Patch(color="#ffbb78", label="transport: loading"),
        mpatches.Patch(color="#c49c74", label="transport: loaded travel"),
        mpatches.Patch(color="#ff7f0e", label="transport: unloading"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def dispatch_order_c0(exp, domain):
    """Return chronological (robot_id, task_id, started_at, dist_from_depot,
    task_location) list from C0's execution records."""
    depots = {r.robot_id: r.location for r in domain.robots}
    tasks = {t.task_id: t for t in domain.tasks}
    events = []
    for rec in exp.process_execution_records:
        for rid in rec["robot_ids"]:
            task = tasks[rec["task_id"]]
            depot = depots[rid]
            dist = math.dist(depot, task.location)
            events.append({
                "kind": "process",
                "robot_id": rid,
                "task_id": rec["task_id"],
                "started_at": rec["started_at"],
                "completed_at": rec["completed_at"],
                "dist_from_depot": round(dist, 2),
                "task_location": [round(task.location[0], 2), round(task.location[1], 2)],
            })
    events.sort(key=lambda e: e["started_at"])
    return events


def analyze_instance(model, device, profile, tc, seed, label):
    print(f"\n=== {label}: profile={profile} tc={tc} seed={seed} ===", flush=True)
    cfg = scaled_cfg(profile, tc, seed)
    domain = generate_md_instance(cfg).domain
    depot = domain.robots[0].location
    print(f"  depot={depot} n_tasks={len(domain.tasks)} n_robots={len(domain.robots)}", flush=True)

    # MILP
    print("  solving MILP...", flush=True)
    milp_exp, milp_r = run_milp(domain,
                                run_id=f"milp-{label}",
                                instance_id=f"{profile}-{tc}-{seed}",
                                seed=seed)
    print(f"  MILP makespan={milp_exp.makespan} status={milp_r.status.value}", flush=True)

    # C0
    print("  running C0...", flush=True)
    c0_exp, c0_wall = run_c0(model, domain,
                             run_id=f"c0-{label}",
                             instance_id=f"{profile}-{tc}-{seed}",
                             device=device, seed=seed)
    print(f"  C0 makespan={c0_exp.makespan} wall={c0_wall:.1f}s", flush=True)

    # Breakdowns
    milp_bd, _ = per_robot_breakdown(milp_exp, domain)
    c0_bd, _ = per_robot_breakdown(c0_exp, domain)

    def totals(bd):
        return {
            "makespan": next(iter(bd.values()))["makespan"],
            "sum_service": sum(v["service_time"] for v in bd.values()),
            "sum_travel": sum(v["travel_time"] for v in bd.values()),
            "sum_wait": sum(v.get("wait_time", 0) for v in bd.values()),
            "sum_idle": sum(v["idle_time"] for v in bd.values()),
        }

    milp_totals = totals(milp_bd)
    c0_totals = totals(c0_bd)

    # C0 dispatch order (for geometry analysis)
    c0_order = dispatch_order_c0(c0_exp, domain)

    # Plots
    out_dir = OUT / f"{label}_{profile}_tc{tc}_s{seed}"
    plot_gantt(milp_exp, domain,
               f"MILP oracle — {profile} tc={tc} seed={seed} makespan={milp_exp.makespan}",
               out_dir / "gantt_milp.png")
    plot_gantt(c0_exp, domain,
               f"C0 IL — {profile} tc={tc} seed={seed} makespan={c0_exp.makespan}",
               out_dir / "gantt_c0.png")

    record = {
        "label": label,
        "profile": profile, "task_count": tc, "seed": seed,
        "depot": list(depot),
        "n_process_robots": sum(1 for r in domain.robots if isinstance(r, ProcessRobot)),
        "n_transport_robots": sum(1 for r in domain.robots if isinstance(r, TransportRobot)),
        "milp_makespan": milp_exp.makespan,
        "c0_makespan": c0_exp.makespan,
        "gap_percent": (c0_exp.makespan - milp_exp.makespan) / milp_exp.makespan * 100.0,
        "milp_per_robot": milp_bd,
        "c0_per_robot": c0_bd,
        "milp_totals": milp_totals,
        "c0_totals": c0_totals,
        "c0_dispatch_order": c0_order,
    }
    (out_dir / "breakdown.json").write_text(json.dumps(record, indent=2, default=list))
    print(f"  -> {out_dir}", flush=True)
    return record


def main():
    device = torch.device("cpu")
    print(f"loading checkpoint {CKPT}", flush=True)
    model, _ = load_md_policy_checkpoint(CKPT, device=device)
    model.eval()

    records = []
    # tight-gap (from rows.json: process_scarce tc=24 seed=301, C0=1095 MILP=771, +42%)
    records.append(analyze_instance(model, device, "process_scarce", 24, 301, "tight"))
    # wide-gap (dependency_deep tc=42 seed=304, C0=996 MILP=422, +136%)
    records.append(analyze_instance(model, device, "dependency_deep", 42, 304, "wide"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(records, indent=2, default=list))
    print("done", flush=True)


if __name__ == "__main__":
    main()
