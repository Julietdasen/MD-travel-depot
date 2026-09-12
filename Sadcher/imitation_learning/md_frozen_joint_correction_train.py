"""Train a bounded joint correction while freezing an existing residual policy."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from data_generation.md_residual_dataset import load_samples, robust_scale
from experiments.md_joint_residual_training import JointResidualTailPolicy, joint_batch_ranking_loss
from experiments.md_residual_tail_training import ResidualCostPredictions
from imitation_learning.md_residual_train import _stack_md_inputs, residual_inputs


@dataclass(frozen=True, slots=True)
class FrozenJointConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    evaluation_interval: int = 10
    seed: int = 3101
    hidden_dim: int = 64
    correction_limit: float = 0.25
    ranking_weight: float = 0.8
    regression_weight: float = 0.2


def _load_policy(residual_checkpoint, config, device):
    payload = torch.load(residual_checkpoint, map_location=device, weights_only=False)
    from imitation_learning.md_residual_train import load_legacy_c0_checkpoint
    base, _ = load_legacy_c0_checkpoint(payload["legacy_checkpoint"], device=device)
    policy = JointResidualTailPolicy(base, pair_feature_dim=int(payload["pair_feature_dim"]), hidden_dim=config.hidden_dim).to(device)
    policy.load_state_dict({key: value for key, value in payload["model_state_dict"].items() if key in policy.state_dict()}, strict=False)
    for name, parameter in policy.named_parameters():
        parameter.requires_grad = name.startswith("joint_head.")
    return policy, payload


def _grouped_batches(samples, batch_size, seed):
    groups = collections.defaultdict(list)
    for sample in samples:
        payload = sample.simulator_snapshot
        groups[(tuple(payload["robot_ids"]), tuple(payload["task_ids"]))].append(sample)
    rng = random.Random(seed)
    for key in sorted(groups):
        values = groups[key]; rng.shuffle(values)
        for start in range(0, len(values), batch_size):
            yield tuple(values[start:start + batch_size])


def _forward_frozen(policy, samples, device):
    prepared = [residual_inputs(sample, device) for sample in samples]
    robot = torch.cat([item[0] for item in prepared]); task = torch.cat([item[1] for item in prepared]); adjacency = torch.cat([item[2] for item in prepared])
    md = _stack_md_inputs([item[3] for item in prepared])
    with torch.no_grad():
        diagnostics = policy.base_policy.forward_with_diagnostics(robot, task, adjacency, md_inputs=md)
        pair = diagnostics.pair_representation
        costs = ResidualCostPredictions(*(head(pair).squeeze(-1) for head in policy.cost_heads))
    return pair, costs, prepared[0][4], prepared[0][5]


def _batch_predictions(policy, pair, costs, sample, sample_index, robot_ids, task_ids, limit):
    ri = {value: index for index, value in enumerate(robot_ids)}; ti = {value: index for index, value in enumerate(task_ids)}
    features, edge = [], []
    for label in sample.legal_batches:
        assignments = tuple((ri[r], ti[t]) for r, t in label.batch.assignments)
        selected = torch.stack([pair[sample_index, r, t] for r, t in assignments])
        metadata = pair.new_tensor([len(assignments), len({t for _, t in assignments}), len(assignments)])
        features.append(torch.cat((selected.mean(0), pair[sample_index].mean((0, 1)), metadata)))
        edge.append([sum((head[sample_index, r, t] for r, t in assignments), head.new_zeros(())) for head in (costs.total, costs.task, costs.tail)])
    correction = torch.tanh(policy.joint_head(torch.stack(features))) * limit
    return torch.stack([torch.stack(values) for values in edge]) + correction, correction


def correction_loss(policy, samples, device, config):
    pair, costs, robot_ids, task_ids = _forward_frozen(policy, samples, device)
    ranks, regressions, magnitudes = [], [], []
    for index, sample in enumerate(samples):
        predicted, correction = _batch_predictions(policy, pair, costs, sample, index, robot_ids, task_ids, config.correction_limit)
        targets = predicted.new_tensor(list(zip(
            robust_scale(tuple(x.remaining_task_horizon + x.terminal_return_tail for x in sample.legal_batches)),
            robust_scale(tuple(x.remaining_task_horizon for x in sample.legal_batches)),
            robust_scale(tuple(x.terminal_return_tail for x in sample.legal_batches)),
        )))
        ranks.append(joint_batch_ranking_loss(predicted[:, 0], [x.regret for x in sample.legal_batches]))
        regressions.append(torch.nn.functional.smooth_l1_loss(predicted, targets))
        magnitudes.append(correction.abs().mean())
    rank = torch.stack(ranks).mean(); regression = torch.stack(regressions).mean(); magnitude = torch.stack(magnitudes).mean()
    return config.ranking_weight * rank + config.regression_weight * regression, rank, regression, magnitude


def evaluate(policy, samples, device, config):
    policy.eval(); regrets = []; tails = []; hits = []; corrections = []
    with torch.no_grad():
        for batch in _grouped_batches(samples, config.batch_size, config.seed):
            pair, costs, robot_ids, task_ids = _forward_frozen(policy, batch, device)
            for index, sample in enumerate(batch):
                predicted, correction = _batch_predictions(policy, pair, costs, sample, index, robot_ids, task_ids, config.correction_limit)
                chosen = int(torch.argmin(predicted[:, 0]))
                label = sample.legal_batches[chosen]; regrets.append(label.regret); tails.append(label.terminal_return_tail); hits.append(float(label.tie_optimal)); corrections.extend(correction[:, 0].tolist())
    return {"forced_action_regret": sum(regrets) / len(regrets), "terminal_return_tail": sum(tails) / len(tails), "tie_aware_accuracy": sum(hits) / len(hits), "correction_abs_mean": sum(abs(x) for x in corrections) / len(corrections), "correction_abs_max": max(abs(x) for x in corrections)}


def train(residual_dataset, residual_checkpoint, output_dir, *, config, device="cuda:0"):
    runtime = torch.device(device); root = Path(residual_dataset); train_samples = load_samples(root / "train.jsonl"); development = load_samples(root / "development.jsonl")
    policy, source = _load_policy(residual_checkpoint, config, runtime)
    baseline = evaluate(policy, development, runtime, config)
    optimizer = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=config.learning_rate)
    evaluations = [{"epoch": 0, **baseline}]; epochs = []; best_state = copy.deepcopy(policy.state_dict())
    for epoch in range(1, config.epochs + 1):
        policy.train(); started = time.perf_counter(); rows = []
        for batch in _grouped_batches(train_samples, config.batch_size, config.seed + epoch):
            loss, rank, regression, magnitude = correction_loss(policy, batch, runtime, config)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); rows.append((float(loss.detach()), float(rank.detach()), float(regression.detach()), float(magnitude.detach())))
        epochs.append({"epoch": epoch, "seconds": time.perf_counter() - started, "loss": sum(x[0] for x in rows) / len(rows), "ranking_loss": sum(x[1] for x in rows) / len(rows), "regression_loss": sum(x[2] for x in rows) / len(rows), "correction_abs_mean": sum(x[3] for x in rows) / len(rows)})
        if epoch % config.evaluation_interval == 0 or epoch == config.epochs:
            row = {"epoch": epoch, **evaluate(policy, development, runtime, config)}; evaluations.append(row)
            eligible = [x for x in evaluations if x["forced_action_regret"] <= baseline["forced_action_regret"] and x["terminal_return_tail"] <= baseline["terminal_return_tail"] + 1.0]
            best = min(eligible, key=lambda x: (x["forced_action_regret"], x["terminal_return_tail"], x["epoch"]))
            if best is row: best_state = copy.deepcopy(policy.state_dict())
    eligible = [x for x in evaluations if x["forced_action_regret"] <= baseline["forced_action_regret"] and x["terminal_return_tail"] <= baseline["terminal_return_tail"] + 1.0]
    best = min(eligible, key=lambda x: (x["forced_action_regret"], x["terminal_return_tail"], x["epoch"]))
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=False); checkpoint = destination / "best_checkpoint.pt"
    torch.save({"schema_version": "frozen-joint-correction-1.0", "model_state_dict": best_state, "config": asdict(config), "pair_feature_dim": source["pair_feature_dim"], "legacy_checkpoint": source["legacy_checkpoint"], "residual_checkpoint": str(Path(residual_checkpoint).resolve()), "baseline_development": baseline, "best_development": best}, checkpoint)
    (destination / "training_summary.json").write_text(json.dumps({"config": asdict(config), "baseline_development": baseline, "best_development": best, "evaluations": evaluations, "epoch_metrics": epochs, "trainable_parameters": [name for name, parameter in policy.named_parameters() if parameter.requires_grad], "test_split_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--residual-dataset", required=True); parser.add_argument("--residual-checkpoint", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(argv)
    config = FrozenJointConfig(seed=args.seed, epochs=args.epochs); print(json.dumps({"checkpoint": str(train(args.residual_dataset, args.residual_checkpoint, args.output_dir, config=config, device=args.device))}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
