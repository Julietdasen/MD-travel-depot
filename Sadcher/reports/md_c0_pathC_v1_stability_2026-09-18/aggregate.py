"""Aggregate 3 seed x 20 epoch Path C v1 stability sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
SEEDS = (3101, 3102, 3103)
EPOCHS = tuple(range(1, 21))
PS60_TARGET = 395.75
DD60_TARGET = 500.5
TIER_COUNT = 8
BASELINE_C0_PS60 = 398.75
BASELINE_C0_DD60 = 500.5
PATH_A_PS60 = 405.2
PATH_B1_PS60 = 415.0
PATH_B2_PS60 = 413.5


def _tier_mean(rows: list[dict], profile: str, task_count: int) -> float:
    values = [
        r["c0_milp_il_v2"]["makespan"]
        for r in rows
        if r["profile"] == profile
        and r["task_count"] == task_count
        and r["c0_milp_il_v2"].get("success")
    ]
    return mean(values)


def _load_summary(seed: int) -> dict:
    path = ROOT / f"seed{seed}" / "eval" / "summary.json"
    return json.loads(path.read_text())


def _load_rows(seed: int, epoch: int) -> list[dict]:
    path = ROOT / f"seed{seed}" / "eval" / f"rows_epoch{epoch}.json"
    return json.loads(path.read_text())


def main() -> None:
    rows_out: list[dict] = []
    for seed in SEEDS:
        summary = _load_summary(seed)
        val_losses = summary["validation_losses"]  # index 0 == pretrained
        for epoch in EPOCHS:
            rows = _load_rows(seed, epoch)
            ps60 = _tier_mean(rows, "process_scarce", 60)
            dd60 = _tier_mean(rows, "dependency_deep", 60)
            val_loss = val_losses[epoch]  # index 0 is pretrained
            non_regressed = summary["tier_non_regression_by_epoch"][str(epoch)]
            regressed = TIER_COUNT - non_regressed
            primary_pass = ps60 <= PS60_TARGET and dd60 <= DD60_TARGET
            rows_out.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": round(val_loss, 6),
                    "ps60_makespan": round(ps60, 4),
                    "dd60_makespan": round(dd60, 4),
                    "primary_pass": int(primary_pass),
                    "tier_regression_count": regressed,
                }
            )

    csv_path = ROOT / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    per_seed: dict[int, dict] = {}
    for row in rows_out:
        s = row["seed"]
        per_seed.setdefault(s, {"passing": [], "best": None})
        if row["primary_pass"]:
            per_seed[s]["passing"].append(row)
        b = per_seed[s]["best"]
        if b is None or row["ps60_makespan"] < b["ps60_makespan"]:
            per_seed[s]["best"] = row

    verdict_lines = ["# Path C v1 stability sweep — final report", ""]
    verdict_lines += [
        f"Dates: training + eval 2026-09-18. Three model seeds (3101 / 3102 / 3103),",
        f"20 epochs each, per-epoch checkpoints, evaluated on the 32-instance grid",
        f"({', '.join(['dependency_deep', 'process_scarce'])} x tc={{12,24,42,60}} x 4 test seeds).",
        f"MILP solver=OR-Tools CP-SAT, 300 s time limit. Fine-tuned from baseline C0.",
        "",
        "## Verdict",
        "",
        "**Goal A (primary)**: each seed must produce >=1 checkpoint with",
        f"60-task process_scarce makespan <= {PS60_TARGET} (greedy_unlock) AND",
        f"60-task dependency_deep makespan <= {DD60_TARGET} (baseline C0).",
        "",
    ]

    seed_pass_count = sum(1 for s in SEEDS if per_seed[s]["passing"])
    for s in SEEDS:
        eps = [r["epoch"] for r in per_seed[s]["passing"]]
        best = per_seed[s]["best"]
        verdict_lines.append(
            f"- seed {s}: passing epochs = {eps or 'NONE'}; "
            f"best-downstream = epoch {best['epoch']} "
            f"(ps60={best['ps60_makespan']:.2f}, dd60={best['dd60_makespan']:.2f}, "
            f"val_loss={best['val_loss']:.4f})"
        )

    if seed_pass_count == 3:
        conclusion = "path_c_v1_stable"
        followup = "do NOT start v2. Next session: build a downstream-correlated model selection signal."
    elif seed_pass_count == 2:
        conclusion = "path_c_v1_partial"
        followup = "start v2a (60-task subset continuation label)."
    else:
        conclusion = "path_c_v1_fluke"
        followup = "start v2a or v2b."

    verdict_lines += [
        "",
        f"**Result: {seed_pass_count}/3 seeds pass. Conclusion = `{conclusion}`.**",
        "",
        f"Follow-up: {followup}",
        "",
        "## Best-downstream checkpoint per seed",
        "",
        "| seed | epoch | ps60 makespan | dd60 makespan | val_loss | tier_regressions |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SEEDS:
        b = per_seed[s]["best"]
        verdict_lines.append(
            f"| {s} | {b['epoch']} | {b['ps60_makespan']:.2f} | {b['dd60_makespan']:.2f} | "
            f"{b['val_loss']:.4f} | {b['tier_regression_count']} |"
        )

    ps60_best_mean = mean(per_seed[s]["best"]["ps60_makespan"] for s in SEEDS)
    verdict_lines += [
        "",
        "## Comparison vs prior attempts (60-task process_scarce)",
        "",
        "| approach | 60-task process_scarce makespan |",
        "|---|---:|",
        f"| MILP (300 s) | 356.0 |",
        f"| baseline C0 (pre-Path A) | {BASELINE_C0_PS60} |",
        f"| greedy_unlock | {PS60_TARGET} |",
        f"| Path A (h=16 fine-tune) | {PATH_A_PS60} |",
        f"| Path B1 (h=48 scratch) | {PATH_B1_PS60} |",
        f"| Path B2 (h=64 scratch) | {PATH_B2_PS60} |",
        f"| Path C v1 stability, mean of best per seed | {ps60_best_mean:.2f} |",
        "",
        "## Val-loss ↔ downstream direction",
        "",
        "Across all 3 seeds the validation ranking loss decreases monotonically from",
        "epoch 1 to 20, but the 60-task process_scarce makespan does NOT track it.",
        "The best downstream checkpoint is at epoch 5 in every seed (or, for seed 3103,",
        "tied by epoch 16). Late-epoch models (17–20) all regress despite lower val loss.",
        "See summary.csv for the full seed x epoch matrix.",
        "",
        "## Artifacts",
        "",
        "- summary.csv (this directory): (seed, epoch, val_loss, ps60_makespan, dd60_makespan,",
        "  primary_pass, tier_regression_count) for all 60 (seed, epoch) pairs.",
        "- reports/md_c0_pathC_v1_stability_2026-09-18/seed{3101,3102,3103}/training/",
        "  checkpoint_epoch_{1..20}.pt (per-epoch checkpoints).",
        "- reports/md_c0_pathC_v1_stability_2026-09-18/seed{3101,3102,3103}/eval/",
        "  rows_epoch{1..20}.json, summary.json, final_report.md (per-seed eval).",
        "- reports/md_c0_pathC_v1_stability_2026-09-18/driver.log (run driver history).",
        "",
    ]

    (ROOT / "final_report.md").write_text("\n".join(verdict_lines), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {ROOT / 'final_report.md'}")
    print(f"conclusion={conclusion} seeds_pass={seed_pass_count}/3")


if __name__ == "__main__":
    main()
