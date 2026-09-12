"""Train the optional joint-batch correction head without touching C0."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from data_generation.md_residual_dataset import load_samples
from experiments.md_joint_residual_training import JointResidualTailPolicy, joint_batch_ranking_loss
from experiments.md_residual_tail_training import select_development_checkpoint
from experiments.protocol import DatasetSplit
from imitation_learning.md_residual_train import _batch_loss, load_legacy_c0_checkpoint, residual_inputs, shape_grouped_mixed_batches


@dataclass(frozen=True, slots=True)
class JointILConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    evaluation_interval: int = 10
    seed: int = 3101
    hidden_dim: int = 64
    initialization: str = "c0"


def _sample_loss(policy, sample, device):
    robot, task, adjacency, md, robot_ids, task_ids = residual_inputs(sample, device)
    diagnostics = policy.base_policy.forward_with_diagnostics(robot, task, adjacency, md_inputs=md)
    _, costs = policy.forward_with_costs(robot, task, adjacency, md_inputs=md)
    robot_index = {value: index for index, value in enumerate(robot_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    predictions = [policy.predict_batch(diagnostics.pair_representation, costs, tuple((robot_index[r], task_index[t]) for r, t in label.batch.assignments)) for label in sample.legal_batches]
    total = torch.stack([value.total for value in predictions])
    targets = torch.tensor([label.remaining_task_horizon + label.terminal_return_tail for label in sample.legal_batches], dtype=total.dtype, device=device)
    scale = targets.detach().abs().max().clamp_min(1.0)
    regression = torch.nn.functional.smooth_l1_loss(total, targets / scale)
    ranking = joint_batch_ranking_loss(total, [label.regret for label in sample.legal_batches])
    return ranking + 0.25 * regression, ranking, regression


def evaluate_joint(policy, samples, device):
    policy.eval(); regrets = []; tails = []; hits = []
    with torch.no_grad():
        for sample in samples:
            robot, task, adjacency, md, robot_ids, task_ids = residual_inputs(sample, device)
            diagnostics = policy.base_policy.forward_with_diagnostics(robot, task, adjacency, md_inputs=md)
            _, costs = policy.forward_with_costs(robot, task, adjacency, md_inputs=md)
            ri = {value: index for index, value in enumerate(robot_ids)}; ti = {value: index for index, value in enumerate(task_ids)}
            chosen = min(sample.legal_batches, key=lambda label: float(policy.predict_batch(diagnostics.pair_representation, costs, tuple((ri[r], ti[t]) for r, t in label.batch.assignments)).total))
            regrets.append(chosen.regret); tails.append(chosen.terminal_return_tail); hits.append(float(chosen.tie_optimal))
    return {"forced_action_regret": sum(regrets) / len(regrets), "terminal_return_tail": sum(tails) / len(tails), "tie_aware_accuracy": sum(hits) / len(hits)}


def train_joint_residual(c0_dataset_root, residual_dataset, legacy_checkpoint, output_dir, *, config=JointILConfig(), initialization_checkpoint=None, device="cpu"):
    runtime = torch.device(device); base, _ = load_legacy_c0_checkpoint(legacy_checkpoint, device=runtime)
    root = Path(residual_dataset); train = load_samples(root / "train.jsonl"); development = load_samples(root / "development.jsonl")
    c0 = MDExpertDatasetLoader(c0_dataset_root).load_samples(split=DatasetSplit.TRAIN)
    first = residual_inputs(train[0], runtime); width = int(base.forward_with_diagnostics(first[0], first[1], first[2], md_inputs=first[3]).pair_representation.shape[-1])
    policy = JointResidualTailPolicy(base, pair_feature_dim=width, hidden_dim=config.hidden_dim).to(runtime)
    if initialization_checkpoint:
        payload = torch.load(initialization_checkpoint, map_location=runtime, weights_only=False)
        policy.load_state_dict({key: value for key, value in payload["model_state_dict"].items() if key in policy.state_dict()}, strict=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate); evaluations = []; epochs = []; best_state = None
    for epoch in range(1, config.epochs + 1):
        policy.train(); values = []
        for c0_batch, residual_batch in shape_grouped_mixed_batches(train, c0, batch_size=config.batch_size, seed=config.seed + epoch):
            c0_loss = _batch_loss(policy, c0_batch, runtime)
            components = [_sample_loss(policy, sample, runtime) for sample in residual_batch]
            joint = torch.stack([item[0] for item in components]).mean()
            loss = 0.5 * c0_loss + 0.5 * joint
            optimizer.zero_grad(); loss.backward(); optimizer.step(); values.append(float(loss.detach()))
        epochs.append({"epoch": epoch, "loss": sum(values) / len(values)})
        if epoch % config.evaluation_interval == 0 or epoch == config.epochs:
            metrics = {"epoch": epoch, **evaluate_joint(policy, development, runtime)}; evaluations.append(metrics)
            best = min(evaluations, key=lambda row: (row["forced_action_regret"], row["terminal_return_tail"], row["epoch"]))
            if best is metrics: best_state = copy.deepcopy(policy.state_dict())
    best = min(evaluations, key=lambda row: (row["forced_action_regret"], row["terminal_return_tail"], row["epoch"]))
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=False); checkpoint = destination / "best_checkpoint.pt"
    torch.save({"schema_version": "joint-residual-il-1.0", "model_state_dict": best_state, "config": asdict(config), "pair_feature_dim": width, "legacy_checkpoint": str(Path(legacy_checkpoint).resolve()), "best_development": best}, checkpoint)
    (destination / "training_summary.json").write_text(json.dumps({"config": asdict(config), "evaluations": evaluations, "epoch_metrics": epochs, "best_development": best, "initialization_checkpoint": initialization_checkpoint, "test_split_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--c0-dataset-root", required=True); parser.add_argument("--residual-dataset", required=True); parser.add_argument("--legacy-checkpoint", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--initialization-checkpoint"); parser.add_argument("--seed", type=int, default=3101); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default="cpu"); args = parser.parse_args(argv)
    path = train_joint_residual(args.c0_dataset_root, args.residual_dataset, args.legacy_checkpoint, args.output_dir, config=JointILConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, initialization="residual" if args.initialization_checkpoint else "c0"), initialization_checkpoint=args.initialization_checkpoint, device=args.device)
    print(json.dumps({"checkpoint": str(path)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
