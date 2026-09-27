"""Experiment A: Inference-time beam@k over the MILP-IL v2 C0 policy.

Runs K parallel policy rollouts on scaled `process_scarce` instances. Branch 0
is the deterministic baseline (identical to the standard online scheduler with
no MIP fallback path exercised for scoring diversification). Branches 1..K-1
apply per-branch Gumbel score perturbations at every decision event, seeded
deterministically by branch index, and roll out to completion. The best
(minimum makespan) branch is reported.

Outputs (all under a new dated report directory):
  - rows.json    per (task_count, seed, K) result plus per-branch details
  - summary.json aggregate makespan / wall-time per (task_count, K), plus
                 gaps vs. MILP and best-greedy from the 2026-09-14 rows
  - final_report.md human-readable table

No historical checkpoints or reports are overwritten. This file does NOT touch
`experiments/md_c0_scaled_target_profiles_eval.py` and does NOT re-run MILP or
greedy baselines: those numbers are read from the existing scaled-eval outputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import load_md_policy_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import (
    LearnedConstrainedDecoder,
    apply_decoder_result,
    simulator_hard_mask,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator

PROFILE = "process_scarce"
TASK_COUNTS = (12, 24, 42, 60)
SEEDS = (301, 302, 303, 304)
K_VALUES = (1, 4, 8)
MAX_ROLLOUT_STEPS = 20_000
GUMBEL_TEMPERATURE = 0.5  # scaled by (score.max - score.min) per decision


def scaled_profile_config(profile_name: str, task_count: int, seed: int) -> MDGeneratorConfig:
    base = dict(INSTANCE_PROFILES[profile_name].parameters)
    base_task_count = base["task_count"]
    scale = task_count / base_task_count
    process_tasks = int(task_count * (1 - base["transport_ratio"]))
    critical_path = min(
        max(base["critical_path_length"], round(base["critical_path_length"] * scale)),
        process_tasks,
    )
    return MDGeneratorConfig(
        seed=seed,
        task_count=task_count,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=critical_path,
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=max(1, round(base["process_robot_count"] * scale)),
        transport_robot_count=max(
            base.get("transport_robot_count", 1),
            round(base["transport_robot_count"] * scale),
        ),
        skill_count=base["skill_count"],
    )


def _score_state(model, sim, device):
    robot_features, task_features, adjacency, md_inputs = build_md_policy_inputs_from_simulator(sim)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        scores = model(
            robot_features.to(device),
            task_features.to(device),
            adjacency.to(device),
            md_inputs=md_inputs.to(device),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return scores[0].detach().cpu(), elapsed


def _rollout_single(model, domain, *, device, decoder, branch_index, generator):
    """Run a single deterministic-or-perturbed policy rollout to completion."""
    sim = MDDiscreteSimulator(domain)
    steps = 0
    inference_time = 0.0
    failure_reason = None
    while not sim.done and steps < MAX_ROLLOUT_STEPS:
        hard_mask = simulator_hard_mask(sim)
        has_pair = bool(torch.any(hard_mask))
        if has_pair:
            scores, elapsed = _score_state(model, sim, device)
            inference_time += elapsed
            if branch_index > 0 and generator is not None:
                score_range = float(scores.max() - scores.min())
                if score_range > 0:
                    uniform = torch.rand(scores.shape, generator=generator).clamp(min=1e-9, max=1 - 1e-9)
                    gumbel = -torch.log(-torch.log(uniform))
                    scores = scores + GUMBEL_TEMPERATURE * score_range * gumbel
            decoded = decoder.decode(scores, sim, hard_feasibility_mask=hard_mask)
            if decoded.assignments:
                apply_decoder_result(sim, decoded)
            elif not sim.has_advancing_work:
                failure_reason = "decoder_no_assignment"
                break
        if sim.done:
            break
        if not sim.has_advancing_work and not has_pair:
            failure_reason = "deadlock"
            break
        sim.step()
        steps += 1
    if not sim.done and failure_reason is None:
        failure_reason = "timeout"
    makespan = float(sim.time) if sim.done else None
    return {
        "success": sim.done,
        "makespan": makespan,
        "inference_time_seconds": inference_time,
        "failure_reason": failure_reason,
        "steps": steps,
    }


def run_beam(model, domain, K, *, device, base_seed=0):
    """Run K parallel rollouts (branch 0 deterministic, others Gumbel-perturbed)."""
    decoder = LearnedConstrainedDecoder()
    branch_records = []
    started = time.perf_counter()
    for b in range(K):
        gen = None if b == 0 else torch.Generator().manual_seed(base_seed + 991 * b + 17)
        result = _rollout_single(
            model, domain, device=device, decoder=decoder, branch_index=b, generator=gen
        )
        result["branch"] = b
        branch_records.append(result)
    wall_time = time.perf_counter() - started
    successful = [r for r in branch_records if r["success"]]
    if not successful:
        best = None
        best_makespan = None
    else:
        best = min(successful, key=lambda r: r["makespan"])
        best_makespan = best["makespan"]
    return {
        "K": K,
        "wall_time_seconds": wall_time,
        "branch_makespans": [r["makespan"] for r in branch_records],
        "branch_success": [r["success"] for r in branch_records],
        "branch_inference_seconds": [r["inference_time_seconds"] for r in branch_records],
        "best_makespan": best_makespan,
        "best_branch": None if best is None else best["branch"],
        "success": best is not None,
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load_baseline_rows(baseline_path: Path):
    """Return dicts keyed by (task_count, seed) with MILP + greedy baselines."""
    payload = json.loads(baseline_path.read_text())
    baselines = {}
    for row in payload:
        if row.get("profile") != PROFILE:
            continue
        key = (row["task_count"], row["seed"])
        greedies = row.get("greedies", {})
        best_g_name = None
        best_g_ms = None
        for name, g in greedies.items():
            if g.get("success") and g.get("makespan") is not None:
                if best_g_ms is None or g["makespan"] < best_g_ms:
                    best_g_ms = g["makespan"]
                    best_g_name = name
        baselines[key] = {
            "milp_makespan": row.get("milp", {}).get("makespan"),
            "milp_timeout": row.get("milp", {}).get("timeout"),
            "c0_baseline_makespan": row.get("c0_milp_il_v2", {}).get("makespan"),
            "greedy_unlock_makespan": greedies.get("greedy_unlock", {}).get("makespan"),
            "best_greedy_name": best_g_name,
            "best_greedy_makespan": best_g_ms,
        }
    return baselines


def run(checkpoint: Path, output: Path, baseline_rows: Path, device_name: str):
    device = torch.device(device_name)
    model, _ = load_md_policy_checkpoint(checkpoint, device=device)
    model.eval()

    baselines = load_baseline_rows(baseline_rows)

    rows = []
    for tc in TASK_COUNTS:
        for seed in SEEDS:
            cfg = scaled_profile_config(PROFILE, tc, seed)
            domain = generate_md_instance(cfg).domain
            base_stats = baselines.get((tc, seed), {})
            for K in K_VALUES:
                # Deterministic base seed per (tc, seed, K) so branch schedules are reproducible.
                base_seed = 100_000 + tc * 1_000 + seed * 10 + K
                result = run_beam(model, domain, K, device=device, base_seed=base_seed)
                row = {
                    "profile": PROFILE,
                    "task_count": tc,
                    "seed": seed,
                    "K": K,
                    "wall_time_seconds": result["wall_time_seconds"],
                    "best_makespan": result["best_makespan"],
                    "best_branch": result["best_branch"],
                    "branch_makespans": result["branch_makespans"],
                    "branch_success": result["branch_success"],
                    "branch_inference_seconds": result["branch_inference_seconds"],
                    "success": result["success"],
                    "milp_baseline": base_stats.get("milp_makespan"),
                    "greedy_unlock_baseline": base_stats.get("greedy_unlock_makespan"),
                    "best_greedy_baseline_name": base_stats.get("best_greedy_name"),
                    "best_greedy_baseline_makespan": base_stats.get("best_greedy_makespan"),
                    "c0_baseline_makespan": base_stats.get("c0_baseline_makespan"),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "tc": tc,
                            "seed": seed,
                            "K": K,
                            "best_ms": result["best_makespan"],
                            "milp": base_stats.get("milp_makespan"),
                            "c0_base": base_stats.get("c0_baseline_makespan"),
                            "gu": base_stats.get("greedy_unlock_makespan"),
                            "wall_s": round(result["wall_time_seconds"], 2),
                            "branch_ms": result["branch_makespans"],
                        }
                    ),
                    flush=True,
                )

    output.mkdir(parents=True, exist_ok=True)
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    # Aggregate per (task_count, K).
    per_tier = []
    for tc in TASK_COUNTS:
        milp_list = []
        gu_list = []
        best_g_list = []
        c0_base_list = []
        for seed in SEEDS:
            b = baselines.get((tc, seed), {})
            if b.get("milp_makespan") is not None:
                milp_list.append(b["milp_makespan"])
            if b.get("greedy_unlock_makespan") is not None:
                gu_list.append(b["greedy_unlock_makespan"])
            if b.get("best_greedy_makespan") is not None:
                best_g_list.append(b["best_greedy_makespan"])
            if b.get("c0_baseline_makespan") is not None:
                c0_base_list.append(b["c0_baseline_makespan"])
        milp_mean = _mean(milp_list)
        gu_mean = _mean(gu_list)
        best_g_mean = _mean(best_g_list)
        c0_base_mean = _mean(c0_base_list)

        for K in K_VALUES:
            makespans = [r["best_makespan"] for r in rows if r["task_count"] == tc and r["K"] == K]
            walls = [r["wall_time_seconds"] for r in rows if r["task_count"] == tc and r["K"] == K]
            per_tier.append(
                {
                    "task_count": tc,
                    "K": K,
                    "beam_mean_makespan": _mean(makespans),
                    "beam_mean_wall_time_seconds": _mean(walls),
                    "milp_mean_makespan": milp_mean,
                    "greedy_unlock_mean_makespan": gu_mean,
                    "best_greedy_mean_makespan": best_g_mean,
                    "c0_baseline_mean_makespan": c0_base_mean,
                }
            )

    def gap(a, b):
        if a is None or b is None or b == 0:
            return None
        return (a - b) / b * 100.0

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"

    def fmt_pct(v):
        return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"

    lines = [
        f"# Beam@K inference-time evaluation (profile: `{PROFILE}`)",
        "",
        f"Checkpoint: `{checkpoint}`  ",
        f"Task counts: {list(TASK_COUNTS)}  ",
        f"Seeds: {list(SEEDS)}  ",
        f"K values: {list(K_VALUES)}  ",
        f"Perturbation: per-branch Gumbel noise scaled by `{GUMBEL_TEMPERATURE} * (score.max - score.min)` at every decision (branch 0 is deterministic baseline).  ",
        f"MILP / greedy baselines read from `{baseline_rows}` (no re-run).",
        "",
        "## Makespan and gap vs MILP",
        "",
        "| task_count | K | beam mean ms | MILP ms | gap vs MILP | best greedy ms | gap vs best greedy | wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tc in TASK_COUNTS:
        for K in K_VALUES:
            entry = next(e for e in per_tier if e["task_count"] == tc and e["K"] == K)
            beam_ms = entry["beam_mean_makespan"]
            milp_ms = entry["milp_mean_makespan"]
            bestg_ms = entry["best_greedy_mean_makespan"]
            lines.append(
                f"| {tc} | {K} | {fmt(beam_ms)} | {fmt(milp_ms)} | {fmt_pct(gap(beam_ms, milp_ms))} | "
                f"{fmt(bestg_ms)} | {fmt_pct(gap(beam_ms, bestg_ms))} | {fmt(entry['beam_mean_wall_time_seconds'])} |"
            )
    lines.append("")

    # K=1 vs K=4 vs K=8 improvement over K=1
    lines.append("## Beam@K improvement over K=1 (same policy, same seeds)")
    lines.append("")
    lines.append("| task_count | K=1 ms | K=4 ms | K=4 vs K=1 | K=8 ms | K=8 vs K=1 | K=4 wall (s) | K=8 wall (s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tc in TASK_COUNTS:
        e1 = next(e for e in per_tier if e["task_count"] == tc and e["K"] == 1)
        e4 = next((e for e in per_tier if e["task_count"] == tc and e["K"] == 4), None)
        e8 = next((e for e in per_tier if e["task_count"] == tc and e["K"] == 8), None)
        lines.append(
            f"| {tc} | {fmt(e1['beam_mean_makespan'])} | "
            f"{fmt(e4['beam_mean_makespan']) if e4 else 'n/a'} | "
            f"{fmt_pct(gap(e4['beam_mean_makespan'], e1['beam_mean_makespan']) if e4 else None)} | "
            f"{fmt(e8['beam_mean_makespan']) if e8 else 'n/a'} | "
            f"{fmt_pct(gap(e8['beam_mean_makespan'], e1['beam_mean_makespan']) if e8 else None)} | "
            f"{fmt(e4['beam_mean_wall_time_seconds']) if e4 else 'n/a'} | "
            f"{fmt(e8['beam_mean_wall_time_seconds']) if e8 else 'n/a'} |"
        )
    lines.append("")

    # Success criteria callout for 60-task tier.
    e60 = {K: next(e for e in per_tier if e["task_count"] == 60 and e["K"] == K) for K in K_VALUES}
    milp60 = e60[1]["milp_mean_makespan"]
    gu60 = e60[1]["greedy_unlock_mean_makespan"]
    lines.append("## Success criteria (60-task tier)")
    lines.append("")
    lines.append(
        f"- MILP mean makespan @60 = {fmt(milp60)}; greedy_unlock @60 = {fmt(gu60)}.  "
    )
    for K in K_VALUES:
        b = e60[K]["beam_mean_makespan"]
        lines.append(
            f"- K={K}: beam ms {fmt(b)}, gap vs MILP {fmt_pct(gap(b, milp60))}, vs greedy_unlock {fmt_pct(gap(b, gu60))}, wall {fmt(e60[K]['beam_mean_wall_time_seconds'])}s"
        )
    lines.append("")

    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "schema": "md-c0-beam-search-pilot-1.0",
        "checkpoint": str(checkpoint),
        "profile": PROFILE,
        "task_counts": list(TASK_COUNTS),
        "seeds": list(SEEDS),
        "K_values": list(K_VALUES),
        "gumbel_temperature": GUMBEL_TEMPERATURE,
        "max_rollout_steps": MAX_ROLLOUT_STEPS,
        "baseline_rows": str(baseline_rows),
        "per_tier": per_tier,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.output, args.baseline_rows, args.device), indent=2))
