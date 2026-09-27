"""Dump per-decision confidence margins during rollouts to pick a threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_generation.md_instance_generator import generate_md_instance
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


def _cfg(profile, tc, seed, scarce, pr, tr):
    base = dict(INSTANCE_PROFILES[profile].parameters)
    scale = tc / base["task_count"]
    from data_generation.md_instance_generator import MDGeneratorConfig
    from dataclasses import replace
    process_tasks = int(tc * (1 - base["transport_ratio"]))
    cpl = min(max(base["critical_path_length"], round(base["critical_path_length"] * scale)), process_tasks)
    cfg = MDGeneratorConfig(
        seed=seed, task_count=tc, transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"], critical_path_length=cpl,
        capacity_slack=base["capacity_slack"], speed_ratio=base["speed_ratio"],
        process_robot_count=pr, transport_robot_count=tr,
        skill_count=base["skill_count"], scarce_skill_count=scarce,
    )
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tc", type=int, default=24)
    p.add_argument("--pr", type=int, default=4)
    p.add_argument("--tr", type=int, default=4)
    p.add_argument("--scarce", type=int, default=0)
    p.add_argument("--seeds", default="301,302,303,304")
    p.add_argument("--profile", default="process_scarce")
    args = p.parse_args()
    device = torch.device("cuda:0")
    model, _ = load_md_policy_checkpoint(args.checkpoint, device=device)
    model.eval()

    all_confs = []
    per_seed = {}
    for seed in [int(x) for x in args.seeds.split(",")]:
        cfg = _cfg(args.profile, args.tc, seed, args.scarce, args.pr, args.tr)
        domain = generate_md_instance(cfg).domain
        scheduler = OnlineMDScheduler(
            scorer=OnlineNeuralScoreProvider(model, build_md_policy_inputs_from_simulator, device=device),
            decoder=LearnedConstrainedDecoder(),
            fallback=ExplicitMIPFallback(threads=1),
            confidence_threshold=0.0,
            max_steps=20000,
        )
        result = scheduler.run(domain, run_id=f"conf-{seed}", instance_id=f"conf-{args.tc}-{seed}",
                               seed=seed, split=DatasetSplit.TEST)
        confs = [d.confidence for d in result.decision_records if d.confidence is not None]
        per_seed[seed] = {"n_decisions": len(confs),
                          "makespan": result.experiment.makespan,
                          "confidences": confs}
        all_confs.extend(confs)

    if all_confs:
        finite = [c for c in all_confs if c != float("inf")]
        summary = {
            "n_decisions_total": len(all_confs),
            "n_inf_margin": sum(1 for c in all_confs if c == float("inf")),
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "p10": sorted(finite)[len(finite)//10] if finite else None,
            "p25": sorted(finite)[len(finite)//4] if finite else None,
            "p50": sorted(finite)[len(finite)//2] if finite else None,
            "p75": sorted(finite)[3*len(finite)//4] if finite else None,
            "p90": sorted(finite)[9*len(finite)//10] if finite else None,
            "mean": sum(finite)/len(finite) if finite else None,
        }
    else:
        summary = {"n_decisions_total": 0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "config": {"tc": args.tc, "pr": args.pr, "scarce": args.scarce},
        "summary": summary,
        "per_seed": {k: {"n_decisions": v["n_decisions"], "makespan": v["makespan"]}
                     for k, v in per_seed.items()},
    }, indent=2) + "\n")
    (args.output / "confidences.json").write_text(json.dumps(per_seed, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
