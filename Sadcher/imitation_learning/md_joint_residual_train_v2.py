"""Vectorized joint residual IL training entrypoint."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from data_generation.md_residual_dataset import load_samples
from experiments.md_joint_residual_training import JointResidualTailPolicy, joint_batch_ranking_loss
from experiments.md_residual_tail_training import ResidualCostPredictions
from experiments.protocol import DatasetSplit
from imitation_learning.md_joint_residual_train import JointILConfig, evaluate_joint
from imitation_learning.md_residual_train import (
    _batch_loss,
    _stack_md_inputs,
    load_legacy_c0_checkpoint,
    residual_batch_loss,
    residual_inputs,
    shape_grouped_mixed_batches,
)


def joint_minibatch_loss(policy, samples, device):
    """One shared base-policy forward for a shape-grouped residual minibatch."""
    prepared = [residual_inputs(sample, device) for sample in samples]
    robot_ids, task_ids = prepared[0][4], prepared[0][5]
    robot = torch.cat([item[0] for item in prepared], dim=0)
    task = torch.cat([item[1] for item in prepared], dim=0)
    adjacency = torch.cat([item[2] for item in prepared], dim=0)
    md = _stack_md_inputs([item[3] for item in prepared])
    diagnostics = policy.base_policy.forward_with_diagnostics(robot, task, adjacency, md_inputs=md)
    pair = diagnostics.pair_representation
    costs = ResidualCostPredictions(*(head(pair).squeeze(-1) for head in policy.cost_heads))
    robot_index = {value: index for index, value in enumerate(robot_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    ranks, regressions = [], []
    for sample_index, sample in enumerate(samples):
        features, edge_totals = [], []
        for label in sample.legal_batches:
            assignments = tuple((robot_index[r], task_index[t]) for r, t in label.batch.assignments)
            selected = torch.stack([pair[sample_index, r, t] for r, t in assignments])
            metadata = pair.new_tensor([len(assignments), len({t for _, t in assignments}), len(assignments)])
            features.append(torch.cat((selected.mean(dim=0), pair[sample_index].mean(dim=(0, 1)), metadata)))
            edge_totals.append(sum((costs.total[sample_index, r, t] for r, t in assignments), costs.total.new_zeros(())))
        total = torch.stack(edge_totals) + policy.joint_head(torch.stack(features))[:, 0]
        targets = total.new_tensor([label.remaining_task_horizon + label.terminal_return_tail for label in sample.legal_batches])
        regressions.append(torch.nn.functional.smooth_l1_loss(total, targets / targets.abs().max().clamp_min(1.0)))
        ranks.append(joint_batch_ranking_loss(total, [label.regret for label in sample.legal_batches]))
    return torch.stack(ranks).mean(), torch.stack(regressions).mean()


def train_joint_residual_v2(c0_dataset_root, residual_dataset, legacy_checkpoint, output_dir, *, config, initialization_checkpoint=None, device="cuda:0"):
    runtime = torch.device(device)
    base, _ = load_legacy_c0_checkpoint(legacy_checkpoint, device=runtime)
    root = Path(residual_dataset)
    train = load_samples(root / "train.jsonl")
    development = load_samples(root / "development.jsonl")
    c0 = MDExpertDatasetLoader(c0_dataset_root).load_samples(split=DatasetSplit.TRAIN)
    first = residual_inputs(train[0], runtime)
    width = int(base.forward_with_diagnostics(first[0], first[1], first[2], md_inputs=first[3]).pair_representation.shape[-1])
    policy = JointResidualTailPolicy(base, pair_feature_dim=width, hidden_dim=config.hidden_dim).to(runtime)
    if initialization_checkpoint:
        payload = torch.load(initialization_checkpoint, map_location=runtime, weights_only=False)
        policy.load_state_dict({key: value for key, value in payload["model_state_dict"].items() if key in policy.state_dict()}, strict=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    evaluations, epochs, best_state = [], [], None
    for epoch in range(1, config.epochs + 1):
        policy.train(); started = time.perf_counter(); metrics = []
        for c0_batch, residual_batch in shape_grouped_mixed_batches(train, c0, batch_size=config.batch_size, seed=config.seed + epoch):
            c0_loss = _batch_loss(policy, c0_batch, runtime)
            edge_loss = residual_batch_loss(policy, residual_batch, runtime)
            joint_rank, joint_regression = joint_minibatch_loss(policy, residual_batch, runtime)
            loss = 0.50 * c0_loss + 0.25 * edge_loss + 0.20 * joint_rank + 0.05 * joint_regression
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            metrics.append((float(loss.detach()), float(c0_loss.detach()), float(edge_loss.detach()), float(joint_rank.detach()), float(joint_regression.detach())))
        epochs.append({"epoch": epoch, "seconds": time.perf_counter() - started, "loss": sum(x[0] for x in metrics) / len(metrics), "c0_loss": sum(x[1] for x in metrics) / len(metrics), "edge_loss": sum(x[2] for x in metrics) / len(metrics), "joint_rank_loss": sum(x[3] for x in metrics) / len(metrics), "joint_regression_loss": sum(x[4] for x in metrics) / len(metrics)})
        if epoch % config.evaluation_interval == 0 or epoch == config.epochs:
            row = {"epoch": epoch, **evaluate_joint(policy, development, runtime)}; evaluations.append(row)
            if min(evaluations, key=lambda x: (x["forced_action_regret"], x["terminal_return_tail"], x["epoch"])) is row:
                best_state = copy.deepcopy(policy.state_dict())
    best = min(evaluations, key=lambda x: (x["forced_action_regret"], x["terminal_return_tail"], x["epoch"]))
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=False)
    checkpoint = destination / "best_checkpoint.pt"
    torch.save({"schema_version": "joint-residual-il-2.0", "model_state_dict": best_state, "config": asdict(config), "pair_feature_dim": width, "legacy_checkpoint": str(Path(legacy_checkpoint).resolve()), "best_development": best}, checkpoint)
    (destination / "training_summary.json").write_text(json.dumps({"config": asdict(config), "loss_weights": {"c0": .50, "edge": .25, "joint_rank": .20, "joint_regression": .05}, "evaluations": evaluations, "epoch_metrics": epochs, "best_development": best, "initialization_checkpoint": initialization_checkpoint, "test_split_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--c0-dataset-root", required=True); parser.add_argument("--residual-dataset", required=True); parser.add_argument("--legacy-checkpoint", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--initialization-checkpoint"); parser.add_argument("--seed", type=int, default=3101); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(argv)
    config = JointILConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, initialization="residual" if args.initialization_checkpoint else "c0")
    print(json.dumps({"checkpoint": str(train_joint_residual_v2(args.c0_dataset_root, args.residual_dataset, args.legacy_checkpoint, args.output_dir, config=config, initialization_checkpoint=args.initialization_checkpoint, device=args.device))}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
