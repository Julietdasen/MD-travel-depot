"""Evaluate Path C ranking-v1 checkpoints on the scaled target profiles grid.

Wraps `experiments.md_c0_scaled_target_profiles_eval.run` and calls it once
per checkpoint (epoch 5, 10, 15, 20 by default). Aggregates the per-tier
makespans into a single summary and writes `final_report.md` including the
primary / secondary judgement described in the Path C v1 spec:

    Primary : 60-task process_scarce mean makespan across the 4 checkpoints
              must be <= 395.75 for at least one checkpoint (i.e. beat
              greedy_unlock 395.8 and pre-tuning C0 398.75).
    Secondary:
      * Val loss vs downstream makespan direction check.
      * Fraction of the 8 (profile, tc) tiers not regressed vs baseline C0.

Baseline reference numbers are hard-coded from the 2026-09-14 scaled-target
run (`reports/md_c0_scaled_target_profiles_eval_2026-09-14/summary.json`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.md_c0_scaled_target_profiles_eval import (
    SEEDS,
    TARGET_PROFILES,
    TASK_COUNTS,
    run as scaled_run,
)


# 2026-09-14 baseline C0 (pre-Path A) makespans -- pulled from
# reports/md_c0_scaled_target_profiles_eval_2026-09-14/summary.json.
BASELINE_C0 = {
    ("process_scarce", 12): 175.75,
    ("process_scarce", 24): 263.5,
    ("process_scarce", 42): 332.25,
    ("process_scarce", 60): 398.75,
    ("dependency_deep", 12): 245.5,
    ("dependency_deep", 24): 284.0,
    ("dependency_deep", 42): 383.25,
    ("dependency_deep", 60): 500.5,
}
BASELINE_BEST_GREEDY = {
    ("process_scarce", 12): ("greedy_distance", 207.0),
    ("process_scarce", 24): ("greedy_unlock", 304.0),
    ("process_scarce", 42): ("greedy_unlock", 346.5),
    ("process_scarce", 60): ("greedy_unlock", 395.75),
    ("dependency_deep", 12): ("greedy_unlock", 245.75),
    ("dependency_deep", 24): ("greedy_unlock", 281.5),
    ("dependency_deep", 42): ("greedy_unlock", 392.25),
    ("dependency_deep", 60): ("greedy_unlock", 492.5),
}
BASELINE_MILP = {
    ("process_scarce", 12): 164.25,
    ("process_scarce", 24): 250.0,
    ("process_scarce", 42): 304.0,
    ("process_scarce", 60): 356.0,
    ("dependency_deep", 12): 226.5,
    ("dependency_deep", 24): 262.25,
    ("dependency_deep", 42): 342.75,
    ("dependency_deep", 60): 444.5,
}

PRIMARY_TIER = ("process_scarce", 60)
PRIMARY_TARGET_MAKESPAN = 395.75  # >= improvement of >= 3 vs baseline 398.75
PATH_A_C0_60_PS = 405.2
PATH_B1_C0_60_PS = 415.0
PATH_B2_C0_60_PS = 413.5


def _evaluate_checkpoints(
    training_dir: Path,
    output_dir: Path,
    epochs: tuple[int, ...],
    device_name: str,
    log_lines: list[str],
) -> dict:
    """Run scaled_target_profiles_eval for each of the requested checkpoints.

    Writes rows_epoch{N}.json for each and returns a dict of per-epoch
    per-tier makespans.
    """
    per_epoch = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for ep in epochs:
        ckpt_path = training_dir / f"checkpoint_epoch_{ep}.pt"
        if not ckpt_path.is_file():
            msg = f"[SKIP] epoch {ep}: checkpoint missing at {ckpt_path}"
            print(msg, flush=True)
            log_lines.append(msg)
            continue
        started = time.perf_counter()
        msg = f"[{time.strftime('%H:%M:%S')}] EVAL START epoch={ep} ckpt={ckpt_path}"
        print(msg, flush=True)
        log_lines.append(msg)
        epoch_out = output_dir / f"_scratch_epoch_{ep}"
        summary = scaled_run(ckpt_path, epoch_out, device_name)
        # move rows.json and summary.json into flat files
        rows_src = epoch_out / "rows.json"
        rows_dst = output_dir / f"rows_epoch{ep}.json"
        rows_dst.write_text(rows_src.read_text(), encoding="utf-8")
        per_epoch[ep] = {
            "checkpoint": str(ckpt_path),
            "per_tier": summary["per_tier"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        msg = (
            f"[{time.strftime('%H:%M:%S')}] EVAL DONE epoch={ep} "
            f"elapsed={per_epoch[ep]['elapsed_seconds']:.1f}s"
        )
        print(msg, flush=True)
        log_lines.append(msg)
    return per_epoch


def _makespan_for(per_tier, profile, tc):
    for row in per_tier:
        if row["profile"] == profile and row["task_count"] == tc:
            return row["c0_makespan"]
    return None


def _write_final_report(
    per_epoch: dict,
    training_summary: dict,
    output_dir: Path,
) -> dict:
    epochs = sorted(per_epoch.keys())
    if not epochs:
        raise RuntimeError("no epoch results to report")

    # Judgement values
    p60_ps_by_epoch = {
        ep: _makespan_for(per_epoch[ep]["per_tier"], *PRIMARY_TIER) for ep in epochs
    }
    primary_pass_epochs = [
        ep for ep, ms in p60_ps_by_epoch.items()
        if ms is not None and ms <= PRIMARY_TARGET_MAKESPAN
    ]
    # tier non-regression counts vs baseline C0
    tier_pass_by_epoch = {}
    for ep in epochs:
        rows = per_epoch[ep]["per_tier"]
        passes = 0
        for row in rows:
            baseline = BASELINE_C0[(row["profile"], row["task_count"])]
            if row["c0_makespan"] is not None and row["c0_makespan"] <= baseline + 1e-6:
                passes += 1
        tier_pass_by_epoch[ep] = passes

    # Val loss trend and downstream trend
    val_losses = training_summary.get("validation_losses", [])
    train_losses = training_summary.get("train_losses", [])

    lines = []
    lines.append(
        "# Path C v1 (margin ranking loss) evaluation, 2026-09-17"
    )
    lines.append("")
    lines.append(
        f"Fine-tuned from `{training_summary.get('pretrained_checkpoint', 'n/a')}` "
        f"on `{training_summary.get('dataset_dir', 'n/a')}` "
        f"({training_summary.get('train_sample_count', '?')} train / "
        f"{training_summary.get('validation_sample_count', '?')} val samples), "
        f"lr={training_summary.get('learning_rate')} K={training_summary.get('num_negatives')} "
        f"margin={training_summary.get('margin')} seed={training_summary.get('seed')} "
        f"epochs={training_summary.get('epochs_run')}."
    )
    lines.append("")

    lines.append("## Primary judgement (60-task process_scarce)")
    lines.append("")
    lines.append(
        f"Target: mean makespan across the 4 seeds <= {PRIMARY_TARGET_MAKESPAN} "
        f"(baseline C0 = {BASELINE_C0[PRIMARY_TIER]}, greedy_unlock = "
        f"{BASELINE_BEST_GREEDY[PRIMARY_TIER][1]}). Improvement threshold "
        f">= 3 timesteps."
    )
    lines.append("")
    lines.append("| epoch | 60-task process_scarce makespan | vs baseline C0 | vs greedy_unlock | pass? |")
    lines.append("|---:|---:|---:|---:|:---:|")
    for ep in epochs:
        ms = p60_ps_by_epoch[ep]
        if ms is None:
            lines.append(f"| {ep} | n/a | n/a | n/a | n/a |")
            continue
        diff_baseline = ms - BASELINE_C0[PRIMARY_TIER]
        diff_greedy = ms - BASELINE_BEST_GREEDY[PRIMARY_TIER][1]
        passed = "YES" if ms <= PRIMARY_TARGET_MAKESPAN else "no"
        lines.append(
            f"| {ep} | {ms:.2f} | {diff_baseline:+.2f} | {diff_greedy:+.2f} | {passed} |"
        )
    lines.append("")
    verdict = "PASSED" if primary_pass_epochs else "FAILED"
    lines.append(f"**Primary verdict: {verdict}.**")
    if primary_pass_epochs:
        lines.append(
            f"Checkpoints beating threshold: {primary_pass_epochs}."
        )
    else:
        best_ep = min(epochs, key=lambda e: (p60_ps_by_epoch[e] if p60_ps_by_epoch[e] is not None else 1e12))
        best_ms = p60_ps_by_epoch[best_ep]
        lines.append(
            f"Best 60-task process_scarce makespan across epochs = {best_ms} (epoch {best_ep}); "
            f"still worse than target {PRIMARY_TARGET_MAKESPAN} and baseline C0 "
            f"{BASELINE_C0[PRIMARY_TIER]}."
        )
    lines.append("")

    # Comparison table vs Path A / B1 / B2 / baseline
    lines.append("### Comparison vs prior attempts (60-task process_scarce)")
    lines.append("")
    lines.append("| approach | 60-task process_scarce makespan |")
    lines.append("|---|---:|")
    lines.append(f"| baseline C0 (pre-Path A) | {BASELINE_C0[PRIMARY_TIER]} |")
    lines.append(f"| greedy_unlock | {BASELINE_BEST_GREEDY[PRIMARY_TIER][1]} |")
    lines.append(f"| Path A (h=16 fine-tune) | {PATH_A_C0_60_PS} |")
    lines.append(f"| Path B1 (h=48 scratch) | {PATH_B1_C0_60_PS} |")
    lines.append(f"| Path B2 (h=64 scratch) | {PATH_B2_C0_60_PS} |")
    for ep in epochs:
        ms = p60_ps_by_epoch[ep]
        lines.append(f"| Path C v1 epoch {ep} | {ms if ms is not None else 'n/a'} |")
    lines.append("")

    # Full per-tier table
    lines.append("## Full per-tier table (all epochs)")
    lines.append("")
    lines.append(
        "For each (profile, task_count) the row shows baseline C0 (pre-Path A) "
        "then Path C v1 makespan per epoch."
    )
    lines.append("")
    header = "| profile | tc | baseline C0 | best greedy | MILP | " + " | ".join(f"ep{ep}" for ep in epochs) + " |"
    align = "|---|---:|---:|---:|---:|" + "|".join(["---:"] * len(epochs)) + "|"
    lines.append(header)
    lines.append(align)
    for profile in TARGET_PROFILES:
        for tc in TASK_COUNTS:
            baseline = BASELINE_C0[(profile, tc)]
            greedy_name, greedy_ms = BASELINE_BEST_GREEDY[(profile, tc)]
            milp = BASELINE_MILP[(profile, tc)]
            per_ep_ms = []
            for ep in epochs:
                ms = _makespan_for(per_epoch[ep]["per_tier"], profile, tc)
                per_ep_ms.append("n/a" if ms is None else f"{ms:.2f}")
            lines.append(
                f"| {profile} | {tc} | {baseline} | {greedy_ms} ({greedy_name}) | {milp} | "
                + " | ".join(per_ep_ms) + " |"
            )
    lines.append("")

    # Secondary judgement: val loss vs downstream direction
    lines.append("## Secondary trends")
    lines.append("")
    lines.append(
        f"Validation loss per epoch (index 0 = pretrained, ranking objective): "
        f"{[round(v, 4) for v in val_losses]}"
    )
    lines.append("")
    lines.append(
        f"Train loss per epoch: {[round(v, 4) for v in train_losses]}"
    )
    lines.append("")
    lines.append("### Val loss vs 60-task process_scarce makespan direction")
    lines.append("")
    for ep in epochs:
        # validation_losses index is epoch (since we recorded epoch-0 first).
        vl = val_losses[ep] if ep < len(val_losses) else None
        ms = p60_ps_by_epoch[ep]
        lines.append(f"* epoch {ep}: val_loss = {vl if vl is None else round(vl, 4)}, 60-task process_scarce makespan = {ms}")
    lines.append("")

    # Tier non-regression count
    lines.append("### Tier non-regression count vs baseline C0 (out of 8)")
    lines.append("")
    for ep in epochs:
        lines.append(f"* epoch {ep}: {tier_pass_by_epoch[ep]}/8 tiers not regressed vs baseline C0")
    lines.append("")

    # Interpretation guidance
    lines.append("## Interpretation guidance")
    lines.append("")
    lines.append(
        "* If **any** checkpoint has 60-task process_scarce makespan <= 395.75 "
        "(i.e. beats greedy_unlock and improves >=3 timesteps over baseline C0 "
        "398.75), the ranking signal is directionally correct. Advance to Path C v2 "
        "(add continuation-makespan-based labels + listwise loss)."
    )
    lines.append(
        "* If no checkpoint clears the primary, but *all* four checkpoints stay "
        "at least as good as Path A (405.2) / B1 (415.0) / B2 (413.5), the "
        "ranking loss is not enough on its own; continuation-value supervision "
        "is needed."
    )
    lines.append(
        "* If any checkpoint is worse than Path A/B1/B2 on 60-task process_scarce "
        "(makespan > 415), ranking loss is a strictly worse direction; abandon "
        "and pivot to candidate beta (add MILP-provided score margins) or "
        "candidate gamma."
    )
    lines.append("")

    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "schema": "md-c0-pathC-ranking-v1-eval-1.0",
        "primary_tier": list(PRIMARY_TIER),
        "primary_target_makespan": PRIMARY_TARGET_MAKESPAN,
        "primary_pass_epochs": primary_pass_epochs,
        "primary_verdict": verdict,
        "p60_ps_by_epoch": {str(ep): p60_ps_by_epoch[ep] for ep in epochs},
        "tier_non_regression_by_epoch": {str(ep): tier_pass_by_epoch[ep] for ep in epochs},
        "epochs_evaluated": epochs,
        "training_pretrained_checkpoint": training_summary.get("pretrained_checkpoint"),
        "training_learning_rate": training_summary.get("learning_rate"),
        "training_num_negatives": training_summary.get("num_negatives"),
        "training_margin": training_summary.get("margin"),
        "training_seed": training_summary.get("seed"),
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "per_epoch": {
            str(ep): {
                "checkpoint": per_epoch[ep]["checkpoint"],
                "per_tier": per_epoch[ep]["per_tier"],
                "elapsed_seconds": per_epoch[ep]["elapsed_seconds"],
            }
            for ep in epochs
        },
        "baseline_c0": {f"{k[0]}_tc{k[1]}": v for k, v in BASELINE_C0.items()},
        "baseline_best_greedy": {
            f"{k[0]}_tc{k[1]}": {"name": v[0], "makespan": v[1]}
            for k, v in BASELINE_BEST_GREEDY.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(
    *,
    training_dir: Path,
    training_summary_path: Path,
    output_dir: Path,
    epochs: tuple[int, ...],
    device: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    training_summary = json.loads(training_summary_path.read_text())
    per_epoch = _evaluate_checkpoints(
        training_dir, output_dir, epochs, device, log_lines
    )
    summary = _write_final_report(per_epoch, training_summary, output_dir)
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("reports/md_c0_pathC_ranking_v1_2026-09-17/training"),
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("reports/md_c0_pathC_ranking_v1_2026-09-17/training/training_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/md_c0_pathC_ranking_v1_eval_2026-09-17"),
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(
        training_dir=args.training_dir,
        training_summary_path=args.training_summary,
        output_dir=args.output,
        epochs=tuple(args.epochs),
        device=args.device,
    )
    print(json.dumps({
        "primary_verdict": result["primary_verdict"],
        "p60_ps_by_epoch": result["p60_ps_by_epoch"],
        "tier_non_regression_by_epoch": result["tier_non_regression_by_epoch"],
    }, indent=2))
