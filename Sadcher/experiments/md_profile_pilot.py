"""Small profile-stratified MD pilot manifest and feasibility report."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from data_generation.md_instance_profiles import INSTANCE_PROFILES
from data_generation.md_instance_generator import generate_md_instance
from data_generation.md_residual_generation import SnapshotCandidate, residual_state
from baselines.exact_online_action_oracle import enumerate_complete_online_actions
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from schedulers.md_greedy_baselines import run_greedy_distance, run_greedy_eta, run_greedy_unlock
from experiments.protocol import DatasetSplit

PROFILES = tuple(INSTANCE_PROFILES)
SPLITS = ("train", "development", "test")
SEEDS = {"train": (101, 102), "development": (201,), "test": (301,)}

def inspect(profile: str, seed: int) -> dict:
    domain = generate_md_instance(INSTANCE_PROFILES[profile].config(seed)).domain
    sim = MDDiscreteSimulator(domain)
    snap = SnapshotCandidate(f"profile-{profile}-{seed}", seed, "initial", domain, sim, {})
    state = residual_state(snap)
    count = len(enumerate_complete_online_actions(domain, state))
    return {"profile": profile, "seed": seed, "task_count": len(domain.tasks),
            "robot_count": len(domain.robots), "pending_count": len(state.pending_task_ids),
            "legal_action_count": count, "exact_labeling": count <= 96}

def run(output: Path) -> dict:
    rows = []
    for split in SPLITS:
        for profile in PROFILES:
            for seed in SEEDS[split]:
                row = inspect(profile, seed); row["split"] = split; rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {split: [r for r in rows if r["split"] == split] for split in SPLITS}
    (output / "profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline_rows = []
    baseline_fns = {"greedy_distance": run_greedy_distance, "greedy_eta": run_greedy_eta,
                    "greedy_unlock": run_greedy_unlock}
    for row in rows:
        domain = generate_md_instance(INSTANCE_PROFILES[row["profile"]].config(row["seed"])).domain
        for name, fn in baseline_fns.items():
            result = fn(MDDiscreteSimulator(domain), run_id=f"profile-pilot-{name}",
                        instance_id=f"{row['profile']}-{row['seed']}", seed=row["seed"],
                        split=DatasetSplit.TRAIN, max_steps=10_000)
            baseline_rows.append({"split": row["split"], "profile": row["profile"], "seed": row["seed"],
                                  "baseline": name, "success": result.success,
                                  "makespan": result.makespan, "wall_time_seconds": result.wall_time_seconds})
    (output / "baseline_results.json").write_text(json.dumps(baseline_rows, indent=2) + "\n")
    summary = {"schema": "md-profile-pilot-1.0", "profiles": list(PROFILES),
               "split_counts": {s: len(manifest[s]) for s in SPLITS},
               "rows": rows, "baseline_rows": baseline_rows,
               "labelable_rows": sum(r["exact_labeling"] for r in rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Profile-stratified MD pilot", "", "Each profile is represented in train, development, and test.", "",
             "| split | profile | seed | tasks | robots | legal actions | exact labels |", "|---|---|---:|---:|---:|---:|---|"]
    lines += [f"| {r['split']} | {r['profile']} | {r['seed']} | {r['task_count']} | {r['robot_count']} | {r['legal_action_count']} | {r['exact_labeling']} |" for r in rows]
    lines += ["", "## Greedy baseline pilot", "", "| profile | baseline | n | success | mean makespan |", "|---|---|---:|---:|---:|"]
    for profile in PROFILES:
        for name in baseline_fns:
            subset = [r for r in baseline_rows if r["profile"] == profile and r["baseline"] == name]
            ok = [r["makespan"] for r in subset if r["success"] and r["makespan"] is not None]
            lines.append(f"| {profile} | {name} | {len(subset)} | {len(ok)}/{len(subset)} | {(sum(ok)/len(ok)) if ok else 'NA'} |")
    (output / "final_report.md").write_text("\n".join(lines) + "\n")
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=Path("reports/md_profile_pilot_2026-09-11"))
    print(json.dumps(run(p.parse_args().output), indent=2))
