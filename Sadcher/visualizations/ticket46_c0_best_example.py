"""Render the most informative paired example from the Ticket 46 C0 pilot.

The figure is intentionally based on the frozen rollout records.  It does not
load a checkpoint or rerun a scheduler, so the visualization cannot change the
diagnostic result it describes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch


MODEL_SEEDS = (3101, 3102, 3103)
REPORT_NAME = "md_c0_end_to_end_diagnostic_pilot_2026-09-01"
BASELINE_NAME = "process_md_greedy"

COLORS = {
    "ink": "#17212B",
    "muted": "#5B6870",
    "grid": "#D8E0E5",
    "panel": "#F6F8F9",
    "c0": "#087F8C",
    "c0_light": "#D8F0F0",
    "baseline": "#6B7280",
    "baseline_light": "#E5E7EB",
    "normal_edge": "#7C8A93",
    "material_edge": "#D97706",
    "transport_empty": "#CBD5E1",
    "transport_loading": "#F59E0B",
    "transport_loaded": "#14B8A6",
    "transport_unloading": "#EF6C57",
    "return": "#94A3B8",
    "fallback": "#C2413B",
    "exit": "#334155",
}

TASK_COLORS = (
    "#3976A8",
    "#5B8E7D",
    "#8A6FAE",
    "#C77932",
    "#B85C5C",
    "#4E8A9B",
    "#927A3A",
    "#6B7FA4",
    "#9B637D",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _starvation_total(row: Mapping[str, Any]) -> float:
    return sum(float(value) for value in dict(row["material_starvation"]).values())


def _total_latency(row: Mapping[str, Any]) -> float:
    return float(dict(row["latency_totals"])["total_seconds"])


def _execution_records(row: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    experiment = row.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError(f"rollout has no experiment payload: {row.get('instance_id')}")
    records = experiment.get("execution_records")
    if not isinstance(records, Mapping):
        raise ValueError(f"rollout has no execution records: {row.get('instance_id')}")
    return {
        "process": [dict(item) for item in records.get("process", [])],
        "transport": [dict(item) for item in records.get("transport", [])],
    }


def _real_task_completion(row: Mapping[str, Any]) -> int:
    records = _execution_records(row)
    completed = [
        int(record["completed_at"])
        for group in records.values()
        for record in group
        if record.get("completed_at") is not None
    ]
    return max(completed, default=0)


def _by_instance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row["instance_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("rollout file contains duplicate instance IDs")
    return indexed


def select_best_case(report_root: Path) -> dict[str, Any]:
    """Select the robust local win, using only paired frozen rollouts.

    A case is eligible when every C0 seed succeeds and beats the same fixed
    baseline.  Ranking by the worst-seed improvement favors a repeatable
    example over a one-seed outlier; the remaining keys make ties deterministic.
    """

    rollout_root = report_root / "rollouts_json_safe_recheck"
    baseline = _by_instance(_read_jsonl(rollout_root / f"{BASELINE_NAME}.jsonl"))
    learned = {
        seed: _by_instance(_read_jsonl(rollout_root / f"C0_seed{seed}.jsonl"))
        for seed in MODEL_SEEDS
    }
    common_ids = set(baseline)
    for rows in learned.values():
        common_ids &= set(rows)

    candidates: list[dict[str, Any]] = []
    for instance_id in sorted(common_ids):
        baseline_row = baseline[instance_id]
        seed_rows = [learned[seed][instance_id] for seed in MODEL_SEEDS]
        if not bool(baseline_row["success"]) or not all(
            bool(row["success"]) for row in seed_rows
        ):
            continue
        improvements = [
            float(baseline_row["makespan"]) - float(row["makespan"])
            for row in seed_rows
        ]
        if min(improvements) <= 0:
            continue
        starvation_deltas = [
            _starvation_total(row) - _starvation_total(baseline_row)
            for row in seed_rows
        ]
        candidates.append(
            {
                "instance_id": instance_id,
                "baseline": baseline_row,
                "seed_rows": dict(zip(MODEL_SEEDS, seed_rows)),
                "improvements": improvements,
                "min_improvement": min(improvements),
                "mean_improvement": sum(improvements) / len(improvements),
                "mean_abs_starvation_delta": sum(
                    abs(value) for value in starvation_deltas
                )
                / len(starvation_deltas),
                "mean_fallback_count": sum(
                    int(row["fallback_count"]) for row in seed_rows
                )
                / len(seed_rows),
            }
        )

    if not candidates:
        raise RuntimeError("no robust C0 win exists in the supplied rollout package")
    candidates.sort(
        key=lambda item: (
            -item["min_improvement"],
            -item["mean_improvement"],
            item["mean_abs_starvation_delta"],
            item["mean_fallback_count"],
            item["instance_id"],
        )
    )
    selection = candidates[0]
    representative_seed = min(
        MODEL_SEEDS,
        key=lambda seed: (
            int(selection["seed_rows"][seed]["fallback_count"]),
            _total_latency(selection["seed_rows"][seed]),
            int(selection["seed_rows"][seed]["repair_count"]),
            seed,
        ),
    )
    selection["representative_seed"] = representative_seed
    selection["c0"] = selection["seed_rows"][representative_seed]
    return selection


def _load_domain(report_root: Path, instance_id: str) -> dict[str, Any]:
    package = _read_json(report_root / "frozen_instances.json")
    instances = package.get("instances")
    if not isinstance(instances, list):
        raise ValueError("frozen instance package has no instances list")
    for payload in instances:
        if isinstance(payload, Mapping) and payload.get("instance_id") == instance_id:
            record = payload.get("record")
            if isinstance(record, Mapping) and isinstance(record.get("domain"), Mapping):
                return dict(record["domain"])
    raise KeyError(f"instance is absent from frozen package: {instance_id}")


def _task_maps(domain: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    process: dict[int, dict[str, Any]] = {}
    transport: dict[int, dict[str, Any]] = {}
    for task in domain["tasks"]:
        task_id = int(task["task_id"])
        if task["task_type"] == "PROCESS":
            process[task_id] = dict(task)
        else:
            transport[task_id] = dict(task)
    return process, transport


def _robot_maps(domain: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(robot["robot_id"]): dict(robot) for robot in domain["robots"]}


def _task_color(task_id: int) -> str:
    return TASK_COLORS[(task_id - 1) % len(TASK_COLORS)]


def _draw_arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.2,
    alpha: float = 1.0,
    mutation_scale: int = 10,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        connectionstyle=connectionstyle,
        shrinkA=8,
        shrinkB=10,
    )
    ax.add_patch(arrow)


def _process_path(row: Mapping[str, Any], robot_id: int, domain: Mapping[str, Any]) -> list[tuple[float, float]]:
    process_tasks, _ = _task_maps(domain)
    robot = next(
        item for item in domain["robots"] if int(item["robot_id"]) == robot_id
    )
    records = [
        record
        for record in _execution_records(row)["process"]
        if robot_id in [int(value) for value in record["robot_ids"]]
    ]
    records.sort(key=lambda record: (int(record["started_at"]), int(record["task_id"])))
    points = [tuple(float(value) for value in robot["location"])]
    points.extend(tuple(float(value) for value in process_tasks[int(record["task_id"])]["location"]) for record in records)
    return points


def draw_spatial_map(
    ax: Any,
    domain: Mapping[str, Any],
    c0_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
) -> None:
    process_tasks, transport_tasks = _task_maps(domain)
    robots = _robot_maps(domain)

    ax.set_facecolor("white")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position", color=COLORS["muted"])
    ax.set_ylabel("Y position", color=COLORS["muted"])
    ax.set_title("Frozen instance geometry and relations", loc="left", pad=10)
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    # Normal precedence edges are drawn first so task nodes remain legible.
    for source_id, target_id in domain["normal_edges"]:
        source = process_tasks[int(source_id)]
        target = process_tasks[int(target_id)]
        _draw_arrow(
            ax,
            tuple(source["location"]),
            tuple(target["location"]),
            color=COLORS["normal_edge"],
            linewidth=1.1,
            alpha=0.7,
            connectionstyle="arc3,rad=0.06",
        )

    # The pickup-to-delivery route is also the material edge into its target.
    for task_id, task in transport_tasks.items():
        pickup = tuple(float(value) for value in task["pickup_location"])
        delivery = tuple(float(value) for value in task["delivery_location"])
        _draw_arrow(
            ax,
            pickup,
            delivery,
            color=COLORS["material_edge"],
            linestyle="--",
            linewidth=1.7,
            alpha=0.9,
            connectionstyle="arc3,rad=-0.04",
        )
        midpoint = ((pickup[0] + delivery[0]) / 2, (pickup[1] + delivery[1]) / 2)
        downstream = next(
            int(target)
            for source, target in domain["material_edges"]
            if int(source) == task_id
        )
        ax.scatter(
            [pickup[0]],
            [pickup[1]],
            s=60,
            marker="^",
            facecolor="white",
            edgecolor=COLORS["material_edge"],
            linewidth=1.5,
            zorder=5,
        )
        label_offsets = {
            10: (0.0, 5.0),
            11: (6.0, -5.5),
            12: (-7.0, 5.0),
        }
        offset_x, offset_y = label_offsets.get(task_id, (0.0, 3.0))
        ax.text(
            midpoint[0] + offset_x,
            midpoint[1] + offset_y,
            f"T{task_id} -> P{downstream}",
            fontsize=7.5,
            color=COLORS["material_edge"],
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
            zorder=6,
        )

    # Light assignment paths provide context for the highlighted R1 choice.
    for robot_id in (0, 1, 2):
        points = _process_path(c0_row, robot_id, domain)
        for start, end in zip(points, points[1:]):
            _draw_arrow(
                ax,
                start,
                end,
                color=COLORS["c0"],
                linewidth=0.8,
                alpha=0.15,
                mutation_scale=7,
            )

    c0_r1 = _process_path(c0_row, 1, domain)
    base_r1 = _process_path(baseline_row, 1, domain)
    for start, end in zip(c0_r1, c0_r1[1:]):
        _draw_arrow(ax, start, end, color=COLORS["c0"], linewidth=2.5, alpha=0.9, mutation_scale=11)
    for start, end in zip(base_r1, base_r1[1:]):
        _draw_arrow(
            ax,
            start,
            end,
            color=COLORS["fallback"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            mutation_scale=9,
        )

    # Process task nodes.
    material_targets = {int(target) for _, target in domain["material_edges"]}
    for task_id, task in process_tasks.items():
        location = tuple(float(value) for value in task["location"])
        facecolor = COLORS["material_edge"] if task_id in material_targets else _task_color(task_id)
        size = 240 + 8 * float(task["duration"])
        ax.scatter(
            [location[0]],
            [location[1]],
            s=size,
            marker="o",
            facecolor=facecolor,
            edgecolor="white",
            linewidth=1.5,
            zorder=7,
        )
        ax.text(
            location[0],
            location[1],
            f"P{task_id}",
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )
        duration_offsets = {
            4: (10.0, 7.0),
            7: (-11.0, 7.0),
            8: (-10.0, -8.0),
        }
        offset_x, offset_y = duration_offsets.get(task_id, (0.0, -6.0))
        ax.text(
            location[0] + offset_x,
            location[1] + offset_y,
            f"{int(task['duration'])}t",
            color=COLORS["ink"],
            fontsize=7,
            ha="center",
            va="top",
            zorder=8,
        )

    # Initial robot positions.
    for robot_id, robot in robots.items():
        location = tuple(float(value) for value in robot["location"])
        is_transport = robot["robot_type"] == "TRANSPORT_ROBOT"
        color = COLORS["material_edge"] if is_transport else COLORS["c0"]
        marker = "D" if is_transport else "s"
        ax.scatter(
            [location[0]],
            [location[1]],
            s=95,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=2,
            zorder=9,
        )
        ax.annotate(
            f"R{robot_id}",
            xy=location,
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=10,
        )

    ax.scatter([0], [0], s=130, marker="X", color=COLORS["exit"], zorder=9)
    ax.text(2.5, 2.5, "exit", fontsize=8, color=COLORS["exit"], va="bottom")

    legend = [
        Line2D([0], [0], color=COLORS["normal_edge"], lw=1.5, label="normal precedence"),
        Line2D([0], [0], color=COLORS["material_edge"], lw=1.8, ls="--", label="material delivery"),
        Line2D([0], [0], color=COLORS["c0"], lw=2.5, label="C0 R1 assignment path"),
        Line2D([0], [0], color=COLORS["fallback"], lw=1.8, ls="--", label="baseline R1 path"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor=COLORS["c0"], markerfacecolor="white", label="process robot"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor=COLORS["material_edge"], markerfacecolor="white", label="transport robot"),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        fontsize=7,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=COLORS["grid"],
        ncol=2,
        handlelength=2.0,
        columnspacing=0.8,
    )


def _bar(ax: Any, y: float, left: float, right: float, *, color: str, edgecolor: str, hatch: str | None = None, alpha: float = 1.0) -> None:
    if right <= left:
        return
    ax.barh(
        y,
        right - left,
        left=left,
        height=0.52,
        color=color,
        edgecolor=edgecolor,
        linewidth=0.8,
        hatch=hatch,
        alpha=alpha,
        zorder=3,
    )


def draw_timeline(
    ax: Any,
    row: Mapping[str, Any],
    domain: Mapping[str, Any],
    *,
    title: str,
    accent: str,
    x_max: float,
) -> None:
    robots = _robot_maps(domain)
    robot_ids = sorted(robots)
    y_positions = {robot_id: len(robot_ids) - index for index, robot_id in enumerate(robot_ids)}
    exit_y = 0
    records = _execution_records(row)
    process_tasks, _ = _task_maps(domain)

    ax.set_facecolor("white")
    ax.set_title(title, loc="left", color=accent, pad=9, fontsize=11, fontweight="bold")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.65, len(robot_ids) + 1.05)
    ax.set_yticks([y_positions[robot_id] for robot_id in robot_ids] + [exit_y])
    ax.set_yticklabels(
        [
            f"R{robot_id} {'P' if robots[robot_id]['robot_type'] == 'PROCESS_ROBOT' else 'T'}"
            for robot_id in robot_ids
        ]
        + ["terminal"],
        fontsize=8,
    )
    ax.set_xlabel("simulation time step")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Process execution bars, including a pale prefix for recorded waiting.
    for record in sorted(records["process"], key=lambda item: int(item["started_at"])):
        task_id = int(record["task_id"])
        start = float(record["started_at"])
        end = float(record["completed_at"])
        wait = float(record.get("waiting_duration", 0))
        robot_ids_for_task = [int(value) for value in record["robot_ids"]]
        for robot_id in robot_ids_for_task:
            y = y_positions[robot_id]
            _bar(
                ax,
                y,
                start - wait,
                start,
                color=COLORS["baseline_light"],
                edgecolor=COLORS["baseline"],
                hatch="..",
                alpha=0.8,
            )
            _bar(
                ax,
                y,
                start,
                end,
                color=_task_color(task_id),
                edgecolor=accent,
                alpha=0.88,
            )
        label_y = sum(y_positions[item] for item in robot_ids_for_task) / len(robot_ids_for_task)
        if end - start >= 4:
            ax.text(
                (start + end) / 2,
                label_y,
                f"P{task_id}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=5,
            )

    # Transport bars are split into the four canonical simulator phases.
    for record in sorted(records["transport"], key=lambda item: int(item["assigned_at"])):
        robot_id = int(record["robot_id"])
        y = y_positions[robot_id]
        task_id = int(record["task_id"])
        boundaries = (
            ("empty", int(record["assigned_at"]), int(record["arrived_pickup_at"]), COLORS["transport_empty"]),
            ("load", int(record["loading_started_at"]), int(record["loading_completed_at"]), COLORS["transport_loading"]),
            ("loaded", int(record["loading_completed_at"]), int(record["arrived_delivery_at"]), COLORS["transport_loaded"]),
            ("unload", int(record["unloading_started_at"]), int(record["completed_at"]), COLORS["transport_unloading"]),
        )
        for _phase, start, end, color in boundaries:
            _bar(ax, y, float(start), float(end), color=color, edgecolor=COLORS["ink"], alpha=0.9)
        ax.text(
            (float(record["assigned_at"]) + float(record["completed_at"])) / 2,
            y,
            f"T{task_id}",
            ha="center",
            va="center",
            fontsize=7,
            color=COLORS["ink"],
            fontweight="bold",
            zorder=5,
        )

    real_complete = _real_task_completion(row)
    makespan = float(row["makespan"])
    _bar(
        ax,
        exit_y,
        float(real_complete),
        makespan,
        color=COLORS["return"],
        edgecolor=COLORS["exit"],
        hatch="///",
        alpha=0.85,
    )
    if makespan > real_complete:
        ax.text(
            (real_complete + makespan) / 2,
            exit_y,
            f"return / exit +{int(makespan - real_complete)}",
            ha="center",
            va="center",
            fontsize=7,
            color=COLORS["ink"],
            zorder=5,
        )

    # Decision times sit above the robot lanes; fallback decisions are red.
    decisions = row.get("decision_records", [])
    top_y = len(robot_ids) + 0.58
    for decision in decisions:
        time = float(decision["decision_time"])
        fallback = bool(decision.get("fallback_used", False))
        ax.scatter(
            [time],
            [top_y],
            s=30 if fallback else 13,
            marker="v" if fallback else "|",
            color=COLORS["fallback"] if fallback else accent,
            edgecolor="white" if fallback else None,
            linewidth=0.5,
            zorder=7,
        )

    ax.axvline(real_complete, color=COLORS["c0"], linestyle=":", linewidth=1.2, alpha=0.9)
    ax.axvline(makespan, color=COLORS["exit"], linestyle="-.", linewidth=1.1, alpha=0.9)
    ax.text(
        real_complete,
        len(robot_ids) + 0.78,
        f"tasks done {real_complete}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["c0"],
    )
    ax.text(
        makespan,
        len(robot_ids) + 0.78,
        f"finish {int(makespan)}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["exit"],
    )
    ax.set_xticks(range(0, int(x_max) + 1, 50))


def draw_summary(
    ax: Any,
    selection: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    c0_row: Mapping[str, Any],
) -> None:
    ax.axis("off")
    ax.set_facecolor(COLORS["panel"])
    instance_id = str(selection["instance_id"])
    baseline_makespan = int(float(baseline_row["makespan"]))
    c0_makespan = int(float(c0_row["makespan"]))
    improvement = baseline_makespan - c0_makespan
    improvement_pct = 100.0 * improvement / baseline_makespan
    c0_real = _real_task_completion(c0_row)
    base_real = _real_task_completion(baseline_row)

    ax.text(0.02, 0.98, "CASE SELECTION", transform=ax.transAxes, fontsize=8, color=COLORS["muted"], va="top", fontweight="bold")
    ax.text(0.02, 0.91, "Robust local win", transform=ax.transAxes, fontsize=18, color=COLORS["ink"], va="top", fontweight="bold")
    ax.text(
        0.02,
        0.84,
        f"{instance_id}  |  representative C0 seed {selection['representative_seed']}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
        va="top",
    )

    ax.text(0.03, 0.72, f"-{improvement}", transform=ax.transAxes, fontsize=31, color=COLORS["c0"], va="center", fontweight="bold")
    ax.text(0.27, 0.735, f"time steps\n{improvement_pct:.1f}% shorter", transform=ax.transAxes, fontsize=10, color=COLORS["ink"], va="center", linespacing=1.35)

    table_data = [
        ["metric", "C0", "process_md_greedy"],
        ["success", "true", "true"],
        ["makespan", str(c0_makespan), str(baseline_makespan)],
        ["real tasks complete", str(c0_real), str(base_real)],
        ["starvation sum", f"{_starvation_total(c0_row):.0f}", f"{_starvation_total(baseline_row):.0f}"],
        ["fallback / solver calls", f"{c0_row['fallback_count']}", "0"],
        ["decision latency (s)", f"{_total_latency(c0_row):.3f}", f"{_total_latency(baseline_row):.3f}"],
    ]
    table = ax.table(cellText=table_data, cellLoc="left", colWidths=[0.43, 0.2, 0.37], bbox=[0.02, 0.48, 0.96, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(0.6)
        cell.set_facecolor("white" if row_index else COLORS["ink"])
        cell.get_text().set_color("white" if row_index == 0 else COLORS["ink"])
        if row_index == 0:
            cell.get_text().set_fontweight("bold")
        if col_index in (1, 2) and row_index > 0:
            cell.get_text().set_ha("center")

    ax.text(0.02, 0.445, "Cross-seed stability on the same frozen instance", transform=ax.transAxes, fontsize=9, color=COLORS["ink"], va="top", fontweight="bold")
    cross_seed = [
        ["C0 seed", "makespan", "gain vs baseline"]
    ]
    for seed in MODEL_SEEDS:
        row = selection["seed_rows"][seed]
        gain = baseline_makespan - int(float(row["makespan"]))
        cross_seed.append([str(seed), str(int(float(row["makespan"]))), f"-{gain}"])
    cross_table = ax.table(cellText=cross_seed, cellLoc="center", colWidths=[0.28, 0.3, 0.42], bbox=[0.02, 0.285, 0.96, 0.13])
    cross_table.auto_set_font_size(False)
    cross_table.set_fontsize(8)
    for (row_index, _col_index), cell in cross_table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(0.6)
        cell.set_facecolor(COLORS["c0_light"] if row_index else COLORS["ink"])
        cell.get_text().set_color(COLORS["ink"] if row_index else "white")
        if row_index == 0:
            cell.get_text().set_fontweight("bold")

    ax.text(0.02, 0.235, "Observed schedule distinction", transform=ax.transAxes, fontsize=9, color=COLORS["ink"], va="top", fontweight="bold")
    ax.text(
        0.02,
        0.195,
        "C0: P1 runs at t=61-66 with R0+R2; R1 handles P6 at t=67-82.\n"
        "Baseline: P1 runs at t=83-88 with R0+R1+R2. Both finish real tasks at t=146;\n"
        "the visible terminal interval is +56 steps for C0 versus +110 for baseline.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLORS["muted"],
        va="top",
        linespacing=1.4,
    )


def _figure_legend(fig: Any) -> None:
    handles = [
        Patch(facecolor=COLORS["transport_empty"], edgecolor=COLORS["ink"], label="transport: empty travel"),
        Patch(facecolor=COLORS["transport_loading"], edgecolor=COLORS["ink"], label="transport: loading"),
        Patch(facecolor=COLORS["transport_loaded"], edgecolor=COLORS["ink"], label="transport: loaded travel"),
        Patch(facecolor=COLORS["transport_unloading"], edgecolor=COLORS["ink"], label="transport: unloading"),
        Patch(facecolor=COLORS["return"], edgecolor=COLORS["exit"], hatch="///", label="return / exit"),
        Line2D([0], [0], marker="v", color=COLORS["fallback"], linestyle="None", label="C0 fallback decision"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=3, frameon=False, fontsize=8, handlelength=1.8, columnspacing=1.3)


def render(report_root: Path, output_stem: Path) -> dict[str, Any]:
    selection = select_best_case(report_root)
    domain = _load_domain(report_root, str(selection["instance_id"]))
    baseline_row = selection["baseline"]
    c0_row = selection["c0"]

    fig = plt.figure(figsize=(18, 13), facecolor="white")
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.55, 4.0, 5.8],
        width_ratios=[1.08, 0.92],
        hspace=0.33,
        wspace=0.22,
    )
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.text(0.0, 0.78, "Ticket 46 / C0 end-to-end diagnostic", fontsize=22, color=COLORS["ink"], fontweight="bold", va="top")
    header.text(0.0, 0.18, "Best robust paired example from the frozen 30-instance package | diagnostic pilot evidence only", fontsize=10.5, color=COLORS["muted"], va="top")
    header.text(1.0, 0.52, f"C0 seed {selection['representative_seed']}  vs  process_md_greedy", fontsize=10, color=COLORS["c0"], ha="right", va="center", fontweight="bold")

    map_ax = fig.add_subplot(grid[1, 0])
    draw_spatial_map(map_ax, domain, c0_row, baseline_row)
    summary_ax = fig.add_subplot(grid[1, 1])
    summary_ax.set_facecolor(COLORS["panel"])
    draw_summary(summary_ax, selection, baseline_row, c0_row)

    max_time = max(float(baseline_row["makespan"]), float(c0_row["makespan"])) + 10
    c0_ax = fig.add_subplot(grid[2, 0])
    draw_timeline(c0_ax, c0_row, domain, title=f"C0 / seed {selection['representative_seed']}", accent=COLORS["c0"], x_max=max_time)
    baseline_ax = fig.add_subplot(grid[2, 1])
    draw_timeline(baseline_ax, baseline_row, domain, title="process_md_greedy baseline", accent=COLORS["baseline"], x_max=max_time)
    _figure_legend(fig)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "instance_id": selection["instance_id"],
        "representative_seed": selection["representative_seed"],
        "improvements": selection["improvements"],
        "png": str(png_path),
        "svg": str(svg_path),
    }


def _default_report_root() -> Path:
    return Path(__file__).resolve().parents[1] / "reports" / REPORT_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=_default_report_root())
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="output path without extension; defaults inside --report-root",
    )
    args = parser.parse_args()
    output_stem = args.output_stem or args.report_root / "ticket46_c0_best_example"
    result = render(args.report_root, output_stem)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
