"""Offline residual-tail IL without modifying the legacy C0 training path."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from data_generation.md_expert_dataset import MDExpertDatasetLoader
from data_generation.md_residual_dataset import ResidualDecisionSample, load_samples, robust_scale
from experiments.md_residual_tail_training import (
    ResidualCostPredictions,
    ResidualTailPolicy,
    batch_margin_ranking_loss,
    residual_tail_loss,
    select_development_checkpoint,
)
from experiments.protocol import DatasetSplit
from imitation_learning.md_train import _batch_loss, load_md_policy_checkpoint
from experiments.md_task_process_context_models import build_context_ablation_model
from models.md_policy import MDPolicyInputs


@dataclass(frozen=True, slots=True)
class ResidualILConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.002
    evaluation_interval: int = 10
    seed: int = 3101
    hidden_dim: int = 64
    margin: float = 0.1
    regression_weight: float = 0.25


@dataclass(frozen=True, slots=True)
class ResidualILResult:
    checkpoint_path: Path
    summary_path: Path
    best_epoch: int


def _tensor(value: object, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device).unsqueeze(0)


def residual_inputs(sample: ResidualDecisionSample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MDPolicyInputs, tuple[int, ...], tuple[int, ...]]:
    payload = sample.simulator_snapshot
    required = {"robot_ids", "task_ids", "robot_features", "task_features", "task_adjacency", "md_inputs"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"residual snapshot is missing training features: {missing}")
    md = payload["md_inputs"]
    if not isinstance(md, Mapping):
        raise ValueError("residual snapshot md_inputs must be an object")
    inputs = MDPolicyInputs(
        robot_metadata=_tensor(md["robot_metadata"], dtype=torch.float32, device=device),
        task_metadata=_tensor(md["task_metadata"], dtype=torch.float32, device=device),
        pair_metadata=_tensor(md["pair_metadata"], dtype=torch.float32, device=device),
        task_is_transport=_tensor(md["task_is_transport"], dtype=torch.bool, device=device),
        typed_adjacency=_tensor(md["typed_adjacency"], dtype=torch.float32, device=device),
        downstream_task_index=_tensor(md["downstream_task_index"], dtype=torch.long, device=device),
        hard_feasibility_mask=_tensor(md["hard_feasibility_mask"], dtype=torch.bool, device=device),
        opportunity_context=None if md.get("opportunity_context") is None else _tensor(md["opportunity_context"], dtype=torch.float32, device=device),
    )
    return (
        _tensor(payload["robot_features"], dtype=torch.float32, device=device),
        _tensor(payload["task_features"], dtype=torch.float32, device=device),
        _tensor(payload["task_adjacency"], dtype=torch.float32, device=device),
        inputs,
        tuple(int(x) for x in payload["robot_ids"]),
        tuple(int(x) for x in payload["task_ids"]),
    )


def _stack_md_inputs(items: Sequence[MDPolicyInputs]) -> MDPolicyInputs:
    def stack(name: str):
        values = [getattr(item, name) for item in items]
        return None if values[0] is None else torch.cat(values, dim=0)

    return MDPolicyInputs(
        robot_metadata=stack("robot_metadata"),
        task_metadata=stack("task_metadata"),
        pair_metadata=stack("pair_metadata"),
        task_is_transport=stack("task_is_transport"),
        typed_adjacency=stack("typed_adjacency"),
        downstream_task_index=stack("downstream_task_index"),
        hard_feasibility_mask=stack("hard_feasibility_mask"),
        opportunity_context=stack("opportunity_context"),
    )


def residual_batch_loss(
    policy: ResidualTailPolicy,
    samples: Sequence[ResidualDecisionSample],
    device: torch.device,
    *,
    margin: float = 0.1,
    regression_weight: float = 0.25,
    return_components: bool = False,
):
    if not samples:
        raise ValueError("residual minibatch must not be empty")
    prepared = [residual_inputs(sample, device) for sample in samples]
    robot_ids, task_ids = prepared[0][4], prepared[0][5]
    if any(item[4] != robot_ids or item[5] != task_ids for item in prepared):
        raise ValueError("residual minibatch must have identical robot/task shapes")
    robot = torch.cat([item[0] for item in prepared], dim=0)
    task = torch.cat([item[1] for item in prepared], dim=0)
    adjacency = torch.cat([item[2] for item in prepared], dim=0)
    md_inputs = _stack_md_inputs([item[3] for item in prepared])
    _, predicted = policy.forward_with_costs(
        robot, task, adjacency, md_inputs=md_inputs
    )
    robot_index = {identifier: index for index, identifier in enumerate(robot_ids)}
    task_index = {identifier: index for index, identifier in enumerate(task_ids)}
    targets = [torch.zeros_like(predicted.total) for _ in range(3)]
    masks = [torch.zeros_like(predicted.total, dtype=torch.bool) for _ in range(3)]
    ranks = []
    for batch_index, sample in enumerate(samples):
        values = (
            robust_scale(tuple(edge.total_cost for edge in sample.edge_labels)),
            robust_scale(tuple(edge.task_cost for edge in sample.edge_labels)),
            robust_scale(tuple(edge.tail_cost for edge in sample.edge_labels)),
        )
        for edge_position, edge in enumerate(sample.edge_labels):
            r, t = robot_index[edge.robot_id], task_index[edge.task_id]
            for head, scaled in enumerate(values):
                targets[head][batch_index, r, t] = scaled[edge_position]
                masks[head][batch_index, r, t] = len(set(scaled)) > 1
        indexed_labels = tuple(
            type(label)(
                type(label.batch)(tuple((robot_index[r], task_index[t]) for r, t in label.batch.assignments)),
                label.makespan, label.remaining_task_horizon, label.terminal_return_tail,
                label.latest_return_robot_id, label.regret, label.tie_optimal,
            ) for label in sample.legal_batches
        )
        ranks.append(batch_margin_ranking_loss(
            -predicted.total[batch_index],
            indexed_labels,
            margin=margin,
        ))
    rank = torch.stack(ranks).mean()
    regression = residual_tail_loss(
        predicted,
        ResidualCostPredictions(*targets),
        regression_masks=ResidualCostPredictions(*masks),
    ) * (regression_weight / 0.25)
    total = rank + regression
    return (total, rank, regression) if return_components else total


def residual_sample_loss(policy: ResidualTailPolicy, sample: ResidualDecisionSample, device: torch.device, *, margin: float = 0.1, regression_weight: float = 0.25) -> torch.Tensor:
    return residual_batch_loss(
        policy,
        (sample,),
        device,
        margin=margin,
        regression_weight=regression_weight,
    )


def shape_grouped_mixed_batches(
    residual_samples: Sequence[ResidualDecisionSample],
    c0_samples: Sequence,
    *,
    batch_size: int,
    seed: int,
):
    if batch_size < 2 or batch_size % 2:
        raise ValueError("residual batch_size must be an even integer >= 2")
    half = batch_size // 2
    residual_groups = collections.defaultdict(list)
    c0_groups = collections.defaultdict(list)
    for sample in residual_samples:
        payload = sample.simulator_snapshot
        residual_groups[(tuple(payload["robot_ids"]), tuple(payload["task_ids"]))].append(sample)
    for sample in c0_samples:
        c0_groups[(tuple(sample.robot_ids), tuple(sample.task_ids))].append(sample)
    rng = random.Random(seed)
    c0_shapes = sorted(c0_groups)
    if not c0_shapes:
        raise ValueError("C0 dataset has no shape groups")
    for group in c0_groups.values():
        rng.shuffle(group)
    mixed_index = 0
    for shape in sorted(residual_groups):
        residual_group = list(residual_groups[shape])
        rng.shuffle(residual_group)
        for start in range(0, len(residual_group), half):
            residual_batch = residual_group[start:start + half]
            while len(residual_batch) < half:
                residual_batch.append(residual_group[len(residual_batch) % len(residual_group)])
            c0_group = c0_groups[c0_shapes[mixed_index % len(c0_shapes)]]
            c0_start = mixed_index * half
            c0_batch = [
                c0_group[(c0_start + index) % len(c0_group)]
                for index in range(half)
            ]
            mixed_index += 1
            yield tuple(c0_batch), tuple(residual_batch)


def evaluate_residual(policy: ResidualTailPolicy, samples: Sequence[ResidualDecisionSample], device: torch.device) -> dict[str, float]:
    regrets, tails, hits = [], [], []
    policy.eval()
    with torch.no_grad():
        for sample in samples:
            robot, task, adjacency, md_inputs, robot_ids, task_ids = residual_inputs(sample, device)
            _, costs = policy.forward_with_costs(robot, task, adjacency, md_inputs=md_inputs)
            ri, ti = {x:i for i,x in enumerate(robot_ids)}, {x:i for i,x in enumerate(task_ids)}
            chosen = min(sample.legal_batches, key=lambda label: sum(float(costs.total[0, ri[r], ti[t]]) for r,t in label.batch.assignments))
            regrets.append(chosen.regret); tails.append(chosen.terminal_return_tail); hits.append(float(chosen.tie_optimal))
    if not regrets:
        raise ValueError("development residual split is empty")
    return {"forced_action_regret": sum(regrets)/len(regrets), "terminal_return_tail": sum(tails)/len(tails), "tie_aware_accuracy": sum(hits)/len(hits)}


def train_residual_tail_policy(c0_dataset_root: str | Path, residual_dataset: str | Path, legacy_checkpoint: str | Path, output_dir: str | Path, *, config: ResidualILConfig = ResidualILConfig(), device: str | torch.device = "cpu") -> ResidualILResult:
    """Fine-tune a new residual checkpoint; legacy data and checkpoint stay read-only."""
    runtime = torch.device(device)
    base, _ = load_legacy_c0_checkpoint(legacy_checkpoint, device=runtime)
    residual_source = Path(residual_dataset)
    if residual_source.is_dir():
        train_residual = load_samples(residual_source / "train.jsonl")
        development = load_samples(residual_source / "development.jsonl")
    else:
        samples = load_samples(residual_source)
        train_residual = tuple(x for x in samples if x.split == "train")
        development = tuple(x for x in samples if x.split == "development")
    if not train_residual or not development:
        raise ValueError("residual dataset requires train and development samples")
    if {x.instance_id for x in train_residual} & {x.instance_id for x in development}:
        raise ValueError("residual train/development instances must be disjoint")
    c0_samples = MDExpertDatasetLoader(c0_dataset_root).load_samples(split=DatasetSplit.TRAIN)
    if not c0_samples:
        raise ValueError("C0 training split is empty")

    first_inputs = residual_inputs(train_residual[0], runtime)
    pair_width = len(train_residual[0].simulator_snapshot["robot_features"][0]) + len(train_residual[0].simulator_snapshot["task_features"][0])
    if hasattr(base, "forward_with_diagnostics"):
        base.eval()
        with torch.no_grad():
            diagnostics = base.forward_with_diagnostics(
                first_inputs[0],
                first_inputs[1],
                first_inputs[2],
                md_inputs=first_inputs[3],
            )
        representation = getattr(diagnostics, "pair_representation", None)
        if representation is not None:
            pair_width = int(representation.shape[-1])

    policy = ResidualTailPolicy(base, pair_feature_dim=pair_width, hidden_dim=config.hidden_dim, cost_to_score_weight=1.0).to(runtime)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    evaluations: list[dict[str, Any]] = []
    epoch_metrics: list[dict[str, float | int]] = []
    best_state = None
    for epoch in range(1, config.epochs + 1):
        policy.train()
        epoch_started = time.perf_counter()
        total_values, c0_values, rank_values, regression_values = [], [], [], []
        processed = 0
        batches = shape_grouped_mixed_batches(
            train_residual,
            c0_samples,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
        )
        for c0_batch, residual_batch in batches:
            c0_loss = _batch_loss(policy, c0_batch, runtime)
            residual_loss, rank_loss, regression_loss = residual_batch_loss(
                policy,
                residual_batch,
                runtime,
                margin=config.margin,
                regression_weight=config.regression_weight,
                return_components=True,
            )
            loss = 0.5 * (c0_loss + residual_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            processed += len(c0_batch) + len(residual_batch)
            total_values.append(float(loss.detach()))
            c0_values.append(float(c0_loss.detach()))
            rank_values.append(float(rank_loss.detach()))
            regression_values.append(float(regression_loss.detach()))
        elapsed = time.perf_counter() - epoch_started
        epoch_metrics.append({
            "epoch": epoch,
            "minibatch_count": len(total_values),
            "samples_processed": processed,
            "samples_per_second": processed / elapsed,
            "loss": sum(total_values) / len(total_values),
            "c0_loss": sum(c0_values) / len(c0_values),
            "ranking_loss": sum(rank_values) / len(rank_values),
            "regression_loss": sum(regression_values) / len(regression_values),
        })
        if epoch % config.evaluation_interval == 0 or epoch == config.epochs:
            metrics = {"epoch": epoch, **evaluate_residual(policy, development, runtime)}
            evaluations.append(metrics)
            if select_development_checkpoint(evaluations) is metrics:
                best_state = copy.deepcopy(policy.state_dict())
    best = select_development_checkpoint(evaluations)
    destination = Path(output_dir) / f"C0_residual_tail_seed{config.seed}"
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint_path = destination / "best_checkpoint.pt"
    projection_error = sum(sample.joint_projection_error for sample in train_residual) / len(train_residual)
    checkpoint_payload = {
        "schema_version": "residual-il-2.0",
        "model_state_dict": best_state,
        "config": asdict(config),
        "cost_to_score_weight": 1.0,
        "pair_feature_dim": pair_width,
        "pair_representation": "md_context_metadata_opportunity",
        "best_development": best,
        "legacy_checkpoint": str(Path(legacy_checkpoint).resolve()),
        "joint_projection_error_mean": projection_error,
        "joint_head_recommended": projection_error > 0.20,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    summary_path = destination / "training_summary.json"
    summary_path.write_text(json.dumps({
        "config": asdict(config),
        "evaluations": evaluations,
        "epoch_metrics": epoch_metrics,
        "best_development": best,
        "pair_feature_dim": pair_width,
        "joint_projection_error_mean": projection_error,
        "joint_head_recommended": projection_error > 0.20,
        "mixing": {"c0_fraction": 0.5, "residual_fraction": 0.5},
        "test_split_read": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ResidualILResult(checkpoint_path, summary_path, int(best["epoch"]))


def load_residual_tail_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[ResidualTailPolicy, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    legacy = Path(payload["legacy_checkpoint"])
    base, _ = load_legacy_c0_checkpoint(legacy, device=device)
    config = payload["config"]
    policy = ResidualTailPolicy(base, pair_feature_dim=int(payload["pair_feature_dim"]), hidden_dim=int(config["hidden_dim"]), cost_to_score_weight=float(payload.get("cost_to_score_weight", 1.0))).to(device)
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    return policy, payload


def load_legacy_c0_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, Mapping[str, Any]]:
    """Load either the versioned C0 checkpoint or diagnostic C0 format."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" in payload and "model_spec" in payload:
        return load_md_policy_checkpoint(path, device=device)
    model = build_context_ablation_model("current_pair_aware", seed=int(payload.get("model_seed", 3101)), device=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def evaluate_residual_test_split(checkpoint: str | Path, residual_dataset: str | Path, output_path: str | Path, *, device: str | torch.device = "cpu") -> Mapping[str, Any]:
    """Read the held-out split only after a checkpoint has been fixed."""
    policy, checkpoint_payload = load_residual_tail_checkpoint(checkpoint, device=device)
    samples = load_samples(Path(residual_dataset) / "test.jsonl")
    metrics = evaluate_residual(policy, samples, torch.device(device))
    result = {"checkpoint": str(Path(checkpoint).resolve()), "selected_development": checkpoint_payload["best_development"], "test_sample_count": len(samples), "test_metrics": metrics}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-dataset-root", required=True)
    parser.add_argument("--residual-dataset", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = train_residual_tail_policy(
        args.c0_dataset_root,
        args.residual_dataset,
        args.legacy_checkpoint,
        args.output_dir,
        config=ResidualILConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=0.002,
            evaluation_interval=10,
            seed=args.seed,
        ),
        device=args.device,
    )
    print(json.dumps({
        "checkpoint": str(result.checkpoint_path),
        "summary": str(result.summary_path),
        "best_epoch": result.best_epoch,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
