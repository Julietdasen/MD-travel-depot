"""Solver-free offline imitation learning for MD expert decision samples."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from data_generation.md_expert_dataset import (
    MDDecisionSample,
    MDExpertDatasetLoader,
)
from experiments.protocol import DatasetSplit, task_level_split
from models.md_enhanced_policy import (
    MDEnhancedPolicyConfig,
    MDEnhancedSchedulerNetwork,
)
from models.md_legacy_features import legacy_policy_inputs_from_samples
from models.md_policy import MDSchedulerNetwork
from models.md_policy_features import md_policy_inputs_from_samples
from models.md_training_enhancements import (
    MDStateValueHead,
    MDTrainingEnhancementConfig,
    compute_optional_training_losses,
)


MD_IL_CHECKPOINT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class MDILTrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 2025
    embed_dim: int = 16
    ff_dim: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 1
    gat_heads: int = 4
    gat_layers: int = 1
    hidden_dim: int = 64
    use_cross_attention: bool = True
    use_pair_aware_attention: bool = False
    cross_attention_heads: int = 4
    residual_bound: float = 0.25
    use_typed_edges: bool = True
    use_downstream_encoding: bool = True
    use_transport_eta: bool = True
    use_capacity_features: bool = True
    legacy_transport_bound: float = 0.25
    physics_scale: float = 1.0
    eta_scale: float = 1.0
    use_critical_path_proxy: bool = True
    use_last_material_blocker: bool = True
    use_coalition_context: bool = True
    use_capacity_scarcity: bool = True
    use_structured_loss: bool = False
    use_value_head: bool = False
    structured_weight: float = 1.0
    value_weight: float = 0.1
    margin_weight: float = 1.0
    max_feasible_assignments: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "seed",
            "embed_dim",
            "ff_dim",
            "transformer_heads",
            "transformer_layers",
            "gat_heads",
            "gat_layers",
            "hidden_dim",
            "cross_attention_heads",
            "max_feasible_assignments",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        if self.embed_dim % self.transformer_heads:
            raise ValueError("embed_dim must be divisible by transformer_heads")
        if self.embed_dim % self.gat_heads:
            raise ValueError("embed_dim must be divisible by gat_heads")
        if self.hidden_dim % self.cross_attention_heads:
            raise ValueError("hidden_dim must be divisible by cross_attention_heads")
        if (
            isinstance(self.residual_bound, bool)
            or not isinstance(self.residual_bound, (int, float))
            or not math.isfinite(self.residual_bound)
            or self.residual_bound < 0
        ):
            raise ValueError("residual_bound must be non-negative and finite")
        for name in ("legacy_transport_bound", "physics_scale", "eta_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "use_cross_attention",
            "use_pair_aware_attention",
            "use_typed_edges",
            "use_downstream_encoding",
            "use_transport_eta",
            "use_capacity_features",
            "use_critical_path_proxy",
            "use_last_material_blocker",
            "use_coalition_context",
            "use_capacity_scarcity",
            "use_structured_loss",
            "use_value_head",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.use_cross_attention and self.use_pair_aware_attention:
            raise ValueError(
                "token cross-attention and pair-aware attention are mutually exclusive"
            )

        self.training_enhancement_config()

    def md_policy_config(self) -> MDEnhancedPolicyConfig:
        return MDEnhancedPolicyConfig(
            enabled=True,
            hidden_dim=self.hidden_dim,
            use_cross_attention=self.use_cross_attention,
            use_pair_aware_attention=self.use_pair_aware_attention,
            cross_attention_heads=self.cross_attention_heads,
            residual_bound=self.residual_bound,
            use_typed_edges=self.use_typed_edges,
            use_downstream_encoding=self.use_downstream_encoding,
            use_transport_eta=self.use_transport_eta,
            use_capacity_features=self.use_capacity_features,
            legacy_transport_bound=self.legacy_transport_bound,
            physics_scale=self.physics_scale,
            eta_scale=self.eta_scale,
            use_critical_path_proxy=self.use_critical_path_proxy,
            use_last_material_blocker=self.use_last_material_blocker,
            use_coalition_context=self.use_coalition_context,
            use_capacity_scarcity=self.use_capacity_scarcity,
        )

    def training_enhancement_config(self) -> MDTrainingEnhancementConfig:
        return MDTrainingEnhancementConfig(
            use_structured_loss=self.use_structured_loss,
            use_value_head=self.use_value_head,
            structured_weight=self.structured_weight,
            value_weight=self.value_weight,
            margin_weight=self.margin_weight,
            max_feasible_assignments=self.max_feasible_assignments,
        )


@dataclass(frozen=True, slots=True)
class MDILTrainingResult:
    checkpoint_path: Path
    summary_path: Path
    train_instance_ids: tuple[str, ...]
    validation_instance_ids: tuple[str, ...]
    train_sample_count: int
    validation_sample_count: int
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]


def train_md_policy(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    config: MDILTrainingConfig,
    device: str | torch.device = "cpu",
) -> MDILTrainingResult:
    """Train only from stored samples; no optimization solver is used here."""

    if not isinstance(config, MDILTrainingConfig):
        raise TypeError("config must be an MDILTrainingConfig")
    loader = MDExpertDatasetLoader(dataset_root)
    train_records = loader.load_records(split=DatasetSplit.TRAIN)
    validation_records = loader.load_records(split=DatasetSplit.VALIDATION)
    if not train_records:
        raise ValueError("offline IL requires at least one train instance")
    if not validation_records:
        raise ValueError("offline IL requires at least one validation instance")
    train_ids = tuple(record.instance_id for record in train_records)
    validation_ids = tuple(record.instance_id for record in validation_records)
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation instance IDs must be disjoint")
    train_groups = {record.task_group_id for record in train_records}
    validation_groups = {record.task_group_id for record in validation_records}
    if train_groups & validation_groups:
        raise ValueError("train and validation task groups must be disjoint")
    for expected_split, records in (
        (DatasetSplit.TRAIN, train_records),
        (DatasetSplit.VALIDATION, validation_records),
    ):
        mismatched = tuple(
            record.instance_id
            for record in records
            if task_level_split(record.task_group_id) is not expected_split
        )
        if mismatched:
            raise ValueError(
                f"{expected_split.value} records disagree with task-level split: "
                f"{mismatched}"
            )
    train_samples = tuple(
        sample for record in train_records for sample in record.samples
    )
    validation_samples = tuple(
        sample for record in validation_records for sample in record.samples
    )
    if not train_samples or not validation_samples:
        raise ValueError("train and validation records must contain decision samples")

    torch.manual_seed(config.seed)
    runtime_device = torch.device(device)
    model = _build_model(config).to(runtime_device)
    enhancement_config = config.training_enhancement_config()
    value_head = (
        MDStateValueHead(hidden_dim=config.hidden_dim).to(runtime_device)
        if enhancement_config.use_value_head
        else None
    )
    parameters = model.parameters()
    if value_head is not None:
        parameters = itertools.chain(parameters, value_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_value_head_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        model.train()
        if value_head is not None:
            value_head.train()
        epoch_losses = []
        for batch in _sample_batches(
            train_samples,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
            shuffle=True,
        ):
            loss = _batch_loss(
                model,
                batch,
                runtime_device,
                enhancement_config=enhancement_config,
                value_head=value_head,
            )
            optimizer.zero_grad()
            loss.backward()
            if not _gradients_finite(model) or (
                value_head is not None and not _gradients_finite(value_head)
            ):
                raise RuntimeError("offline IL produced non-finite gradients")
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_loss = _mean(epoch_losses, "training")
        train_losses.append(train_loss)

        model.eval()
        if value_head is not None:
            value_head.eval()
        with torch.no_grad():
            epoch_validation = [
                float(
                    _batch_loss(
                        model,
                        batch,
                        runtime_device,
                        enhancement_config=enhancement_config,
                        value_head=value_head,
                    ).cpu()
                )
                for batch in _sample_batches(
                    validation_samples,
                    batch_size=config.batch_size,
                    seed=config.seed,
                    shuffle=False,
                )
            ]
        validation_loss = _mean(epoch_validation, "validation")
        validation_losses.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_value_head_state = (
                None
                if value_head is None
                else copy.deepcopy(value_head.state_dict())
            )

    assert best_state is not None
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_checkpoint.pt"
    summary_path = destination / "training_summary.json"
    checkpoint = {
        "schema_version": MD_IL_CHECKPOINT_VERSION,
        "model_kind": "md_enhanced",
        "state_dict": best_state,
        "model_spec": _model_spec(config),
        "md_policy_config": asdict(config.md_policy_config()),
        "training_config": asdict(config),
        "training_enhancements": asdict(enhancement_config),
        "value_head_state_dict": best_value_head_state,
        "dataset": {
            "train_instance_ids": list(train_ids),
            "validation_instance_ids": list(validation_ids),
            "train_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
        },
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "best_validation_loss": best_validation,
        "solver_calls_during_training": 0,
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        key: value
        for key, value in checkpoint.items()
        if key not in ("state_dict", "value_head_state_dict")
    }
    summary["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(runtime_device),
    }
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MDILTrainingResult(
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        train_instance_ids=train_ids,
        validation_instance_ids=validation_ids,
        train_sample_count=len(train_samples),
        validation_sample_count=len(validation_samples),
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
    )


def load_md_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MDSchedulerNetwork, dict[str, Any]]:
    runtime_device = torch.device(device)
    checkpoint = torch.load(path, map_location=runtime_device, weights_only=True)
    if checkpoint.get("schema_version") != MD_IL_CHECKPOINT_VERSION:
        raise ValueError("unsupported MD IL checkpoint schema")
    model_spec = checkpoint["model_spec"]
    if checkpoint.get("model_kind") != "md_enhanced":
        raise ValueError("unsupported MD IL model kind")
    md_config = MDEnhancedPolicyConfig(**checkpoint["md_policy_config"])
    model = MDEnhancedSchedulerNetwork(md_config=md_config, **model_spec).to(runtime_device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_missing = {
        "opportunity_projection.weight",
        "opportunity_raw_magnitudes",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible MD policy checkpoint: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model, checkpoint


def _build_model(config: MDILTrainingConfig) -> MDEnhancedSchedulerNetwork:
    return MDEnhancedSchedulerNetwork(
        md_config=config.md_policy_config(), **_model_spec(config)
    )


def _model_spec(config: MDILTrainingConfig) -> dict[str, Any]:
    return {
        "robot_input_dimensions": 7,
        "task_input_dimension": 9,
        "embed_dim": config.embed_dim,
        "ff_dim": config.ff_dim,
        "n_transformer_heads": config.transformer_heads,
        "n_transformer_layers": config.transformer_layers,
        "n_gatn_heads": config.gat_heads,
        "n_gatn_layers": config.gat_layers,
        "dropout": 0.0,
        "use_idle": False,
    }


def _sample_batches(
    samples: Sequence[MDDecisionSample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterable[tuple[MDDecisionSample, ...]]:
    groups: dict[
        tuple[tuple[int, ...], tuple[int, ...]], list[MDDecisionSample]
    ] = {}
    for sample in samples:
        groups.setdefault((sample.robot_ids, sample.task_ids), []).append(sample)
    generator = random.Random(seed)
    for key in sorted(groups):
        group = list(groups[key])
        if shuffle:
            generator.shuffle(group)
        for start in range(0, len(group), batch_size):
            yield tuple(group[start : start + batch_size])


def _batch_loss(
    model: MDSchedulerNetwork,
    samples: tuple[MDDecisionSample, ...],
    device: torch.device,
    *,
    enhancement_config: MDTrainingEnhancementConfig | None = None,
    value_head: MDStateValueHead | None = None,
) -> torch.Tensor:
    legacy = legacy_policy_inputs_from_samples(samples)
    md_inputs = md_policy_inputs_from_samples(samples)
    robot_features = legacy.robot_features.to(device)
    task_features = legacy.task_features.to(device)
    task_adjacency = legacy.task_adjacency.to(device)
    targets = torch.tensor(
        [sample.expert_assignment for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    feasible = md_inputs.hard_feasibility_mask.to(device)
    if not torch.any(feasible):
        raise ValueError("offline IL batch contains no feasible expert actions")
    scores = model(
        robot_features,
        task_features,
        task_adjacency,
        md_inputs=md_inputs,
    )
    loss = F.binary_cross_entropy_with_logits(scores[feasible], targets[feasible])
    if enhancement_config is not None:
        loss = loss + compute_optional_training_losses(
            scores,
            samples,
            policy=model,
            md_inputs=md_inputs,
            config=enhancement_config,
            value_head=value_head,
        ).total
    if not torch.isfinite(scores).all() or not torch.isfinite(loss):
        raise RuntimeError("offline IL produced non-finite scores or loss")
    return loss


def _gradients_finite(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(
        gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    )


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ValueError(f"{label} produced no batches")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} loss is non-finite")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MD-SADCHER++ offline IL")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--structured-loss", action="store_true")
    parser.add_argument("--value-head", action="store_true")
    parser.add_argument("--structured-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--max-feasible-assignments", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = MDILTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        use_structured_loss=args.structured_loss,
        use_value_head=args.value_head,
        structured_weight=args.structured_weight,
        value_weight=args.value_weight,
        margin_weight=args.margin_weight,
        max_feasible_assignments=args.max_feasible_assignments,
    )
    result = train_md_policy(args.dataset_dir, args.output_dir, config=config)
    print(result.checkpoint_path)


if __name__ == "__main__":
    main()
