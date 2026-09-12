"""Held-out rollout evaluation for frozen joint-correction checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_dataset import FORMAL_RESIDUAL_SPLIT_PLAN
from data_generation.md_residual_pipeline import encode_training_features
from data_generation.md_residual_generation import eligible_snapshot, residual_state
from baselines.gurobi_md_residual_oracle import enumerate_forced_batches
from experiments.md_joint_residual_training import JointResidualTailPolicy
from experiments.md_residual_evaluation import _aggregate, _attach_state_regret, _run_milp, _run_online
from imitation_learning.md_residual_train import load_legacy_c0_checkpoint, load_residual_tail_checkpoint
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.joint_batch_decoder import JointBatchDecoder
from schedulers.md_constrained_decoder import FastAssignmentRepair, LearnedConstrainedDecoder, MaskedGreedyDecoder, simulator_hard_mask
from schedulers.online_md_scheduler import OnlineNeuralScoreProvider


def load_frozen_joint_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    source, _ = load_residual_tail_checkpoint(payload["residual_checkpoint"], device=device)
    policy = JointResidualTailPolicy(source.base_policy, pair_feature_dim=int(payload["pair_feature_dim"]), hidden_dim=int(payload["config"]["hidden_dim"])).to(device)
    policy.load_state_dict(payload["model_state_dict"], strict=True); policy.eval()
    return policy, payload


class FrozenJointScoreProvider:
    def __init__(self, policy, correction_limit, device):
        self.policy = policy; self.limit = float(correction_limit); self.device = torch.device(device)
        self.pair = None; self.costs = None; self.scores = None; self.robot_ids = (); self.task_ids = ()

    def __call__(self, simulator):
        robot, task, adjacency, md = build_md_policy_inputs_from_simulator(simulator)
        robot = robot.to(self.device); task = task.to(self.device); adjacency = adjacency.to(self.device); md = md.to(self.device)
        with torch.no_grad():
            diagnostics = self.policy.base_policy.forward_with_diagnostics(robot, task, adjacency, md_inputs=md)
            self.pair = diagnostics.pair_representation
            from experiments.md_residual_tail_training import ResidualCostPredictions
            self.costs = ResidualCostPredictions(*(head(self.pair).squeeze(-1) for head in self.policy.cost_heads))
            total = self.costs.total; scores = diagnostics.scores
            scores = torch.cat((scores[..., :total.shape[-1]] - total, scores[..., total.shape[-1]:]), dim=-1)
        self.robot_ids = tuple(sorted(simulator.robot_states)); self.task_ids = tuple(sorted(simulator.task_states)); self.scores = scores[0]
        return self.scores

    def score_batch(self, assignments):
        if self.pair is None or self.costs is None:
            raise RuntimeError("score provider must run before batch scoring")
        ri = {value: index for index, value in enumerate(self.robot_ids)}; ti = {value: index for index, value in enumerate(self.task_ids)}
        indexed = tuple((ri[r], ti[t]) for r, t in assignments)
        raw = self.policy.joint_head(__import__("experiments.joint_batch_features", fromlist=["pooled_batch_features"]).pooled_batch_features(self.pair, indexed))[0]
        base = -sum((self.scores[r, t] for r, t in indexed), self.scores.new_zeros(()))
        return float(base + torch.tanh(raw) * self.limit)


class JointDecoderAdapter:
    def __init__(self, domain, provider): self.domain = domain; self.provider = provider
    def decode(self, scores, simulator, *, hard_feasibility_mask=None):
        mask = simulator_hard_mask(simulator) if hard_feasibility_mask is None else hard_feasibility_mask
        baseline = LearnedConstrainedDecoder().decode(scores, simulator, hard_feasibility_mask=mask)
        if not eligible_snapshot(simulator) or not baseline.assignments:
            return baseline
        state = residual_state(type("Candidate", (), {"simulator": simulator, "domain": self.domain})())
        task_ids = sorted(task for _, task in baseline.assignments)
        batches = tuple(batch for batch in enumerate_forced_batches(self.domain, state) if len(batch.assignments) == len(baseline.assignments) and sorted(task for _, task in batch.assignments) == task_ids)
        if not batches:
            return baseline
        selected = min(batches, key=lambda batch: self.provider.score_batch(batch.assignments))
        if self.provider.score_batch(selected.assignments) >= self.provider.score_batch(baseline.assignments) - 1e-6:
            return baseline
        repaired = FastAssignmentRepair().repair(selected.assignments, scores, simulator, hard_feasibility_mask=mask)
        return replace(repaired, decoder="frozen_joint_local")


def paired(rows, model_seed):
    edge = {int(x["instance_seed"]): x for x in rows if x["method"] == "residual" and x["model_seed"] == model_seed}
    joint = {int(x["instance_seed"]): x for x in rows if x["method"] == "frozen_joint" and x["model_seed"] == model_seed}
    common = [seed for seed in sorted(edge) if edge[seed]["success"] and joint[seed]["success"]]
    makespan = [float(joint[s]["final_makespan"]) - float(edge[s]["final_makespan"]) for s in common]
    completion = [float(joint[s]["latest_task_completion"]) - float(edge[s]["latest_task_completion"]) for s in common]
    return {"model_seed": model_seed, "matched_instances": len(edge), "common_successes": len(common), "mean_final_makespan_delta": statistics.mean(makespan), "mean_task_completion_delta": statistics.mean(completion), "win_tie_loss": {"win": sum(x < 0 for x in makespan), "tie": sum(x == 0 for x in makespan), "loss": sum(x > 0 for x in makespan)}, "joint_mean_not_worse": statistics.mean(makespan) <= 0, "success_not_decreased": sum(bool(x["success"]) for x in joint.values()) >= sum(bool(x["success"]) for x in edge.values()), "illegal_assignments_zero": all(int(x["illegal_assignment_count"]) == 0 for x in joint.values()), "mean_task_completion_not_worse": statistics.mean(completion) <= 0, "per_instance_task_constraint": all(x <= 1 for x in completion), "deltas": [{"instance_seed": seed, "final_makespan_delta": delta} for seed, delta in zip(common, makespan, strict=True)]}


def run(output_dir, *, c0_checkpoints, residual_checkpoints, joint_checkpoints, seeds=FORMAL_RESIDUAL_SPLIT_PLAN.test_seeds, device="cpu"):
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=False)
    c0 = {s: load_legacy_c0_checkpoint(p, device=device)[0] for s, p in c0_checkpoints.items()}
    residual = {s: load_residual_tail_checkpoint(p, device=device)[0] for s, p in residual_checkpoints.items()}
    joint = {s: load_frozen_joint_checkpoint(p, device=device) for s, p in joint_checkpoints.items()}
    rows = []
    for instance_seed in seeds:
        domain = generate_md_instance(MDGeneratorConfig(seed=instance_seed)).domain; instance_id = f"md-residual-{instance_seed}"
        for model_seed in sorted(c0):
            rows.append(_run_online(domain, instance_id=instance_id, instance_seed=instance_seed, method="legacy_c0", model_seed=model_seed, scorer=OnlineNeuralScoreProvider(c0[model_seed], build_md_policy_inputs_from_simulator, device=device), decoder=LearnedConstrainedDecoder(), regret_time_limit=10, max_steps=10000))
            rows.append(_run_online(domain, instance_id=instance_id, instance_seed=instance_seed, method="residual", model_seed=model_seed, scorer=OnlineNeuralScoreProvider(residual[model_seed], build_md_policy_inputs_from_simulator, device=device), decoder=LearnedConstrainedDecoder(), regret_time_limit=10, max_steps=10000))
            policy, payload = joint[model_seed]; provider = FrozenJointScoreProvider(policy, payload["config"]["correction_limit"], device)
            rows.append(_run_online(domain, instance_id=instance_id, instance_seed=instance_seed, method="frozen_joint", model_seed=model_seed, scorer=provider, decoder=JointDecoderAdapter(domain, provider), regret_time_limit=10, max_steps=10000))
        rows.append(_run_online(domain, instance_id=instance_id, instance_seed=instance_seed, method="masked_greedy", model_seed=None, scorer=lambda sim: torch.zeros_like(simulator_hard_mask(sim), dtype=torch.float32), decoder=MaskedGreedyDecoder(), regret_time_limit=10, max_steps=10000))
        rows.append(_run_milp(domain, instance_id=instance_id, instance_seed=instance_seed, time_limit=60, max_steps=10000))
    _attach_state_regret(rows); aggregates = _aggregate(rows); comparisons = [paired(rows, seed) for seed in sorted(c0)]
    acceptance = {"all_joint_seeds_not_worse": all(x["joint_mean_not_worse"] for x in comparisons), "success_not_decreased": all(x["success_not_decreased"] for x in comparisons), "illegal_assignments_zero": all(x["illegal_assignments_zero"] for x in comparisons), "mean_task_completion_not_worse": all(x["mean_task_completion_not_worse"] for x in comparisons), "per_instance_task_constraint": all(x["per_instance_task_constraint"] for x in comparisons)}; acceptance["passed"] = all(acceptance.values())
    (destination / "rollouts.jsonl").write_text("".join(json.dumps(x, allow_nan=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps({"test_seeds": list(seeds), "aggregates": aggregates, "paired_joint_vs_residual": comparisons, "acceptance": acceptance}, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination / "summary.json"


def mapping(values):
    return {int(value.partition("=")[0]): Path(value.partition("=")[2]) for value in values}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--c0-checkpoint", action="append", required=True); parser.add_argument("--residual-checkpoint", action="append", required=True); parser.add_argument("--joint-checkpoint", action="append", required=True); parser.add_argument("--seeds", nargs="+", type=int); parser.add_argument("--device", default="cpu"); args = parser.parse_args(argv)
    seeds = FORMAL_RESIDUAL_SPLIT_PLAN.test_seeds if args.seeds is None else tuple(args.seeds)
    if any(seed not in FORMAL_RESIDUAL_SPLIT_PLAN.test_seeds for seed in seeds): raise ValueError("all seeds must belong to the fixed test split")
    print(run(args.output_dir, c0_checkpoints=mapping(args.c0_checkpoint), residual_checkpoints=mapping(args.residual_checkpoint), joint_checkpoints=mapping(args.joint_checkpoint), seeds=seeds, device=args.device)); return 0


if __name__ == "__main__": raise SystemExit(main())
