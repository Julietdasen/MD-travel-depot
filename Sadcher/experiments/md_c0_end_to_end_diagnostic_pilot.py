"""Ticket 46 C0 end-to-end diagnostic pilot."""

from __future__ import annotations

import argparse

import json
import math
import os
import platform
import random
import socket
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch

from data_generation.md_dataset import MDInstanceRecord
from data_generation.md_instance_generator import (
    MDGeneratorConfig,
    generate_md_instance,
)
from models.md_online_features import build_md_policy_inputs_from_simulator
from schedulers.md_constrained_decoder import simulator_hard_mask
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.md_static_validator import validate_md_domain
from experiments.md_flip_curriculum_train_diagnostic import (
    _capture_checkpoint,
    _train_epoch,
)
from experiments.md_pair_structured_train_diagnostic import (
    _family_balanced_batches,
    _prepare_batch,
)
from experiments.md_task_process_context_models import build_context_ablation_model
from experiments.md_train_only_optimization_diagnostic import (
    _select_first_pairs_per_family,
)
from experiments.md_unconditioned_relational_audit import (
    build_unconditioned_relational_candidates,
)
from experiments.protocol import DatasetSplit, FailureReason
from models.md_policy import MDOpportunityFeature
from schedulers.md_constrained_decoder import LearnedConstrainedDecoder, MaskedGreedyDecoder
from schedulers.md_greedy_baselines import MDTransportGreedy, TransportGreedyStrategy
from schedulers.online_md_scheduler import (
    ExplicitMIPFallback,
    OnlineMDScheduler,
    OnlineNeuralScoreProvider,
    ScoreOutput,
)
from schedulers.process_greedy_md import ProcessGreedyMDAdapter


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT_ROOT: Final = (
    REPOSITORY_ROOT / "reports" / "md_c0_end_to_end_diagnostic_pilot_2026-09-01"
)
MODEL_SEEDS: Final = (3101, 3102, 3103)
FROZEN_INSTANCE_SEEDS: Final = tuple(range(46000, 46030))
EXIT_LOCATION: Final = (0.0, 0.0)
MAX_ROLLOUT_STEPS: Final = 10_000
FALLBACK_CONFIDENCE_THRESHOLD: Final = 0.0
FALLBACK_THREADS: Final = 1
LATENCY_WARMUP_FORWARDS: Final = 10
BOOTSTRAP_REPLICATES: Final = 2_000
BOOTSTRAP_SEED: Final = 46046
PRIMARY_FIXED_BASELINE: Final = "process_md_greedy"
FIXED_BASELINES: Final = (
    "physics_only",
    "eta_unlock_heuristic",
    "masked_greedy",
    PRIMARY_FIXED_BASELINE,
)
LEARNED_VARIANTS: Final = ("C0", "C1", "C2")
NOMINAL_GENERATOR_CONFIG: Final = {
    "task_count": 12,
    "transport_ratio": 0.25,
    "precedence_density": 0.25,
    "critical_path_length": 3,
    "capacity_slack": 0.2,
    "speed_ratio": 0.8,
    "process_robot_count": 3,
    "transport_robot_count": 2,
    "skill_count": 3,
}
C0_TRAINING_CONFIG: Final = {
    "pretrain_learning_rate": 0.01,
    "fine_tune_learning_rate": 0.002,
    "weight_decay": 0.0,
    "pairs_per_family_per_batch": 100,
    "pretrain_epochs": 200,
    "fine_tune_epochs": 100,
    "scheduler": "none",
    "data_seed": 3030,
    "candidate_pool_per_family": 5000,
    "train_pairs_per_family": 500,
    "checkpoint_interval": 10,
    "state_margin": 0.1,
    "flip_state_margin": 0.1,
    "pair_margin": 0.1,
    "flip_state_loss_weight": 0.5,
    "pair_loss_weight": 0.25,
    "saturation_loss_weight": 0.1,
    "saturation_penalty_allowance": 0.25,
    "minimum_overall_train_agreement": 0.70,
    "minimum_family_flip_pair_exact": 0.60,
    "maximum_residual_saturation_rate": 0.25,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _static_validation_payload(result: object) -> dict[str, object]:
    return {
        "is_valid": bool(getattr(result, "is_valid")),
        "reason_code": (
            None
            if getattr(result, "reason_code") is None
            else getattr(result, "reason_code").value
        ),
        "failure_reason": (
            None
            if getattr(result, "failure_reason") is None
            else getattr(result, "failure_reason").value
        ),
        "message": getattr(result, "message"),
    }


def _model_blind_validation(domain) -> dict[str, object]:
    simulator = MDDiscreteSimulator(domain, exit_location=EXIT_LOCATION)
    legacy_robot, legacy_task, adjacency, md_inputs = (
        build_md_policy_inputs_from_simulator(simulator)
    )
    floating_tensors = (
        legacy_robot,
        legacy_task,
        adjacency,
        md_inputs.robot_metadata,
        md_inputs.task_metadata,
        md_inputs.pair_metadata,
        md_inputs.typed_adjacency,
        md_inputs.opportunity_context,
    )
    return {
        "model_loaded": False,
        "initial_hard_feasible_pair_count": int(simulator_hard_mask(simulator).sum()),
        "legacy_robot_shape": list(legacy_robot.shape),
        "legacy_task_shape": list(legacy_task.shape),
        "adjacency_shape": list(adjacency.shape),
        "md_pair_shape": list(md_inputs.pair_metadata.shape),
        "all_floating_features_finite": all(
            tensor is not None and bool(torch.isfinite(tensor).all())
            for tensor in floating_tensors
        ),
    }


def _frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "ticket": 46,
        "ticket_type": "diagnostic_pilot",
        "primary_learned_method": "C0",
        "model_seeds": list(MODEL_SEEDS),
        "secondary_learned_diagnostics": {
            "variants": ["C1", "C2"],
            "model_seeds": list(MODEL_SEEDS),
            "candidate_selection_signal": False,
        },
        "c0_training": dict(C0_TRAINING_CONFIG),
        "fixed_baselines": list(FIXED_BASELINES),
        "primary_fixed_comparison_baseline": PRIMARY_FIXED_BASELINE,
        "process_md_greedy_adapter": "greedy_eta",
        "instance_seeds": list(FROZEN_INSTANCE_SEEDS),
        "nominal_generator_config": dict(NOMINAL_GENERATOR_CONFIG),
        "terminal_protocol": {
            "exit_location": list(EXIT_LOCATION),
            "max_rollout_steps": MAX_ROLLOUT_STEPS,
            "integer_timestep_simulator": True,
        },
        "fallback": {
            "confidence_threshold": FALLBACK_CONFIDENCE_THRESHOLD,
            "threads": FALLBACK_THREADS,
            "fixed_baselines_enabled": False,
        },
        "latency": {
            "visible_gpu_per_process": 1,
            "device": "cuda:0",
            "batch_size": 1,
            "warmup_decision_forwards": LATENCY_WARMUP_FORWARDS,
            "formal_runs_sequential": True,
            "measurement_boundary": [
                "feature_bridge",
                "neural_forward",
                "constrained_decoder",
                "repair",
                "explicit_fallback",
                "complete_decision",
            ],
        },
        "controls": {
            "package_generation_model_blind": True,
            "ticket_20_executed": False,
            "ticket_44_executed": False,
            "ticket_45_executed": False,
            "ticket_17_unfrozen": False,
            "ticket_20_unfrozen": False,
            "ticket_32_unfrozen": False,
            "production_claim": False,
        },
    }


def freeze_diagnostic_package(root: str | Path) -> dict[str, object]:
    """Freeze the model-blind Ticket 46 package exactly once."""

    destination = Path(root)
    targets = (
        destination / "frozen_protocol.json",
        destination / "frozen_instances.json",
        destination / "package_validation.json",
    )
    existing = tuple(path for path in targets if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite diagnostic package: "
            + ", ".join(str(path) for path in existing)
        )

    instances: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for seed in FROZEN_INSTANCE_SEEDS:
        generated = generate_md_instance(
            MDGeneratorConfig(seed=seed, **NOMINAL_GENERATOR_CONFIG)
        )
        record = MDInstanceRecord(
            instance_id=f"md-c0-diagnostic-{seed}",
            task_group_id=f"md-c0-diagnostic-group-{seed}",
            seed=seed,
            generated=generated,
        )
        static_validation = validate_md_domain(generated.domain)
        model_blind_validation = _model_blind_validation(generated.domain)
        if not static_validation.is_valid or not model_blind_validation[
            "all_floating_features_finite"
        ]:
            raise RuntimeError(f"frozen diagnostic instance is invalid: {record.instance_id}")
        instances.append(
            {
                "instance_id": record.instance_id,
                "instance_seed": seed,
                "generator_config": record.to_dict()["generation_config"],
                "record": record.to_dict(),
                "static_validation": _static_validation_payload(static_validation),
                "model_blind_validation": model_blind_validation,
            }
        )
        validation_rows.append(
            {
                "instance_id": record.instance_id,
                "instance_seed": seed,
                "static_valid": static_validation.is_valid,
                "model_blind_valid": model_blind_validation[
                    "all_floating_features_finite"
                ],
            }
        )

    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "ticket": 46,
        "instance_count": len(instances),
        "instance_seeds": list(FROZEN_INSTANCE_SEEDS),
        "model_loaded_during_generation": False,
        "all_static_valid": all(row["static_valid"] for row in validation_rows),
        "all_model_blind_valid": all(
            row["model_blind_valid"] for row in validation_rows
        ),
        "training_instance_overlap_count": 0,
        "development_instance_overlap_count": 0,
        "training_development_template_overlap_count": 0,
        "overlap_check_basis": "separate frozen MD instance identifiers and relational-template namespace",
    }
    _write_json(destination / "frozen_protocol.json", _frozen_protocol_payload())
    _write_json(
        destination / "frozen_instances.json",
        {
            "schema_version": "1.0.0",
            "ticket": 46,
            "model_loaded_during_generation": False,
            "instance_seeds": list(FROZEN_INSTANCE_SEEDS),
            "instances": instances,
        },
    )
    _write_json(
        destination / "package_validation.json",
        {**summary, "instances": validation_rows},
    )
    return summary


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = BOOTSTRAP_SEED
) -> list[float | None]:
    if not values:
        return [None, None]
    generator = random.Random(seed)
    sample_count = len(values)
    means = sorted(
        sum(values[generator.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def paired_makespan_summary(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Keep failures visible while comparing only jointly successful pairs."""

    baseline_by_id = {str(row["instance_id"]): row for row in baseline}
    candidate_by_id = {str(row["instance_id"]): row for row in candidate}
    if len(baseline_by_id) != len(baseline) or len(candidate_by_id) != len(candidate):
        raise ValueError("paired makespan inputs must use unique instance IDs")
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired makespan inputs must contain the same instance IDs")

    differences: list[float] = []
    baseline_only_successes = 0
    candidate_only_successes = 0
    for instance_id in sorted(baseline_by_id):
        base = baseline_by_id[instance_id]
        contender = candidate_by_id[instance_id]
        base_success = bool(base["success"])
        candidate_success = bool(contender["success"])
        if base_success and candidate_success:
            base_makespan = base["makespan"]
            candidate_makespan = contender["makespan"]
            if base_makespan is None or candidate_makespan is None:
                raise ValueError("successful paired rollouts require makespans")
            differences.append(float(candidate_makespan) - float(base_makespan))
        elif base_success:
            baseline_only_successes += 1
        elif candidate_success:
            candidate_only_successes += 1
    return {
        "matched_pairs": len(baseline_by_id),
        "common_successful_pairs": len(differences),
        "candidate_minus_baseline": {
            "count": len(differences),
            "mean": statistics.fmean(differences) if differences else None,
            "median": statistics.median(differences) if differences else None,
            "bootstrap_ci95": _bootstrap_mean_ci(differences),
        },
        "survivor_bias": {
            "baseline_only_successes": baseline_only_successes,
            "candidate_only_successes": candidate_only_successes,
            "both_failed": sum(
                not bool(baseline_by_id[key]["success"])
                and not bool(candidate_by_id[key]["success"])
                for key in baseline_by_id
            ),
        },
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index, _value in ranked[start:end]:
            result[index] = average_rank
        start = end
    return result


def spearman_rank_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Return a deterministic tie-aware Spearman correlation."""

    if len(left) != len(right):
        raise ValueError("Spearman inputs must have identical lengths")
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left_ranks))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)



def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_from_frozen_payload(payload: Mapping[str, object]) -> MDInstanceRecord:
    record_payload = payload["record"]
    if not isinstance(record_payload, Mapping):
        raise ValueError("frozen instance record must be a JSON object")
    config_payload = record_payload["generation_config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("frozen generation config must be a JSON object")
    generated = generate_md_instance(MDGeneratorConfig(**dict(config_payload)))
    record = MDInstanceRecord(
        instance_id=str(record_payload["instance_id"]),
        task_group_id=str(record_payload["task_group_id"]),
        seed=int(record_payload["seed"]),
        generated=generated,
        split=DatasetSplit(str(record_payload["split"])),
    )
    if record.to_dict()["domain"] != record_payload["domain"]:
        raise RuntimeError("frozen instance domain does not match its generator config")
    return record


def load_frozen_diagnostic_records(root: str | Path) -> tuple[MDInstanceRecord, ...]:
    package_path = Path(root) / "frozen_instances.json"
    package = _read_json(package_path)
    if package.get("model_loaded_during_generation") is not False:
        raise RuntimeError("frozen package was not generated model-blind")
    payloads = package.get("instances")
    if not isinstance(payloads, list):
        raise ValueError("frozen package instances must be a list")
    records = tuple(
        _record_from_frozen_payload(payload)
        for payload in payloads
        if isinstance(payload, Mapping)
    )
    if len(records) != len(payloads):
        raise ValueError("frozen package contains an invalid instance payload")
    if tuple(record.seed for record in records) != FROZEN_INSTANCE_SEEDS:
        raise RuntimeError("frozen package seed list differs from the preregistration")
    return records


def _resolve_single_cuda_device(device: str | torch.device) -> tuple[str, torch.device]:
    requested = str(device).strip().lower()
    if requested != "cuda:0":
        raise ValueError("Ticket 46 requires one visible GPU addressed as cuda:0")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "," in visible_devices:
        raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Ticket 46 requires exactly one visible CUDA device")
    return requested, torch.device("cuda:0")


def _runtime_metadata(requested: str, device: torch.device) -> dict[str, object]:
    return {
        "requested_device": requested,
        "resolved_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "single_device_execution": True,
        "ddp": False,
        "data_parallel": False,
    }


def _training_batches(device: torch.device):
    candidates = build_unconditioned_relational_candidates(
        candidates_per_family=int(C0_TRAINING_CONFIG["candidate_pool_per_family"]),
        seed=int(C0_TRAINING_CONFIG["data_seed"]),
    )
    twins = _select_first_pairs_per_family(
        candidates,
        pairs_per_family=int(C0_TRAINING_CONFIG["train_pairs_per_family"]),
    )
    states = tuple(state for twin in twins for state in (twin.before, twin.after))
    batches = tuple(
        _prepare_batch(batch, device=device)
        for batch in _family_balanced_batches(
            twins,
            pairs_per_family_per_batch=int(
                C0_TRAINING_CONFIG["pairs_per_family_per_batch"]
            ),
        )
    )
    return twins, states, batches


def _checkpoint_record(snapshot: object) -> dict[str, object]:
    return {
        "epoch": getattr(snapshot, "fine_tune_epoch"),
        "overall_train_agreement": getattr(snapshot, "overall_train_agreement"),
        "residual_saturation_rate": getattr(snapshot, "residual_saturation_rate"),
        "oracle_flip_pair_exact_by_family": getattr(snapshot, "family_exact"),
    }


def _checkpoint_selection_score(snapshot: object) -> tuple[bool, int, float, float, int]:
    family_metrics = getattr(snapshot, "family_exact")
    exact = [
        float(row["exact_pair_accuracy"])
        for row in family_metrics.values()
        if row["exact_pair_accuracy"] is not None
    ]
    agreement = float(getattr(snapshot, "overall_train_agreement"))
    saturation = float(getattr(snapshot, "residual_saturation_rate"))
    return (
        agreement >= float(C0_TRAINING_CONFIG["minimum_overall_train_agreement"])
        and saturation
        <= float(C0_TRAINING_CONFIG["maximum_residual_saturation_rate"]),
        sum(
            value >= float(C0_TRAINING_CONFIG["minimum_family_flip_pair_exact"])
            for value in exact
        ),
        min(exact, default=-1.0),
        agreement,
        -int(getattr(snapshot, "fine_tune_epoch")),
    )


def _train_c0_model(model, batches, train_twins, train_states, output_dir: Path):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_log: list[dict[str, object]] = []
    checkpoints: list[object] = []

    def run_stage(
        stage: str,
        epochs: int,
        learning_rate: float,
        *,
        pair_weight: float,
        flip_weight: float,
    ) -> None:
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(C0_TRAINING_CONFIG["weight_decay"]),
        )
        for epoch in range(1, epochs + 1):
            losses = _train_epoch(
                model,
                batches,
                optimizer,
                state_margin=float(C0_TRAINING_CONFIG["state_margin"]),
                flip_state_margin=float(C0_TRAINING_CONFIG["flip_state_margin"]),
                pair_margin=float(C0_TRAINING_CONFIG["pair_margin"]),
                flip_state_loss_weight=flip_weight,
                pair_loss_weight=pair_weight,
                saturation_loss_weight=float(
                    C0_TRAINING_CONFIG["saturation_loss_weight"]
                ),
                saturation_penalty_allowance=float(
                    C0_TRAINING_CONFIG["saturation_penalty_allowance"]
                ),
            )
            epoch_log.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **losses,
                }
            )
            if stage == "fine_tune" and (
                epoch % int(C0_TRAINING_CONFIG["checkpoint_interval"]) == 0
                or epoch == int(C0_TRAINING_CONFIG["fine_tune_epochs"])
            ):
                checkpoints.append(
                    _capture_checkpoint(
                        model,
                        train_twins,
                        train_states,
                        fine_tune_epoch=epoch,
                        minimum_family_flip_pair_exact=float(
                            C0_TRAINING_CONFIG[
                                "minimum_family_flip_pair_exact"
                            ]
                        ),
                    )
                )

    run_stage(
        "pretrain",
        int(C0_TRAINING_CONFIG["pretrain_epochs"]),
        float(C0_TRAINING_CONFIG["pretrain_learning_rate"]),
        pair_weight=0.0,
        flip_weight=0.0,
    )
    checkpoints.append(
        _capture_checkpoint(
            model,
            train_twins,
            train_states,
            fine_tune_epoch=0,
            minimum_family_flip_pair_exact=float(
                C0_TRAINING_CONFIG["minimum_family_flip_pair_exact"]
            ),
        )
    )
    run_stage(
        "fine_tune",
        int(C0_TRAINING_CONFIG["fine_tune_epochs"]),
        float(C0_TRAINING_CONFIG["fine_tune_learning_rate"]),
        pair_weight=float(C0_TRAINING_CONFIG["pair_loss_weight"]),
        flip_weight=float(C0_TRAINING_CONFIG["flip_state_loss_weight"]),
    )
    selected = max(checkpoints, key=_checkpoint_selection_score)
    model.load_state_dict(getattr(selected, "state_dict"), strict=True)
    selected_record = _checkpoint_record(selected)
    family_metrics = selected_record["oracle_flip_pair_exact_by_family"]
    assert isinstance(family_metrics, Mapping)
    criteria = {
        "overall_train_agreement_met": float(
            selected_record["overall_train_agreement"]
        )
        >= float(C0_TRAINING_CONFIG["minimum_overall_train_agreement"]),
        "all_family_flip_pair_exact_met": all(
            bool(row["criterion_met"]) for row in family_metrics.values()
        ),
        "residual_saturation_rate_met": float(
            selected_record["residual_saturation_rate"]
        )
        <= float(C0_TRAINING_CONFIG["maximum_residual_saturation_rate"]),
    }
    criteria["all_training_conditions_met"] = all(criteria.values())
    return {
        "selected": selected,
        "epoch_log": epoch_log,
        "training": {
            "selected_fine_tune_epoch": selected_record["epoch"],
            "overall_train_agreement": selected_record["overall_train_agreement"],
            "residual_saturation_rate": selected_record[
                "residual_saturation_rate"
            ],
            "oracle_flip_pair_exact_by_family": family_metrics,
            "criteria": criteria,
            "checkpoint_metrics": [_checkpoint_record(item) for item in checkpoints],
            "epoch_count": len(epoch_log),
        },
    }


def _checkpoint_reload_verified(
    checkpoint_path: Path,
    *,
    model_seed: int,
    device: torch.device,
    batch,
) -> dict[str, bool]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"]
    first = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    second = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    first.load_state_dict(state_dict, strict=True)
    second.load_state_dict(state_dict, strict=True)
    with torch.no_grad():
        first_scores = first(
            batch.robot_features,
            batch.task_features,
            batch.task_adjacency,
            md_inputs=batch.md_inputs,
        ).detach().cpu()
        second_scores = second(
            batch.robot_features,
            batch.task_features,
            batch.task_adjacency,
            md_inputs=batch.md_inputs,
        ).detach().cpu()
    return {
        "state_dict_equal": all(
            torch.equal(first.state_dict()[name].detach().cpu(), value.detach().cpu())
            for name, value in second.state_dict().items()
        ),
        "same_input_scores_equal": torch.equal(first_scores, second_scores),
    }


def train_c0_seed(
    root: str | Path,
    *,
    model_seed: int,
    device: str | torch.device = "cuda:0",
) -> Path:
    """Train one preregistered C0 seed without reading a development package."""

    if model_seed not in MODEL_SEEDS:
        raise ValueError(f"unsupported C0 model seed: {model_seed}")
    requested, runtime_device = _resolve_single_cuda_device(device)
    destination = Path(root) / "training" / f"C0_seed{model_seed}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite C0 training run: {destination}")
    destination.mkdir(parents=True)
    train_twins, train_states, batches = _training_batches(runtime_device)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=runtime_device
    )
    trained = _train_c0_model(model, batches, train_twins, train_states, destination)
    checkpoint_path = destination / "checkpoints" / "best_checkpoint.pt"
    selected = trained["selected"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_checkpoint": _checkpoint_record(selected),
            "config": dict(C0_TRAINING_CONFIG),
            "model_seed": model_seed,
            "variant": "C0",
        },
        checkpoint_path,
    )
    verification = _checkpoint_reload_verified(
        checkpoint_path,
        model_seed=model_seed,
        device=runtime_device,
        batch=batches[0],
    )
    if not all(verification.values()):
        raise RuntimeError("C0 checkpoint reload is not deterministic")
    training = trained["training"]
    assert isinstance(training, dict)
    training["best_checkpoint"] = str(checkpoint_path.relative_to(destination))
    training["checkpoint_reload_verified"] = True
    _write_json(
        destination / "complete_config.json",
        {
            "ticket": 46,
            "variant": "C0",
            "model_seed": model_seed,
            "training_config": dict(C0_TRAINING_CONFIG),
            "runtime": _runtime_metadata(requested, runtime_device),
            "held_out_data_read": False,
        },
    )
    _write_json(destination / "epoch_log.json", {"epochs": trained["epoch_log"]})
    run_payload = {
        "schema_version": "1.0.0",
        "ticket": 46,
        "status": "completed",
        "variant": "C0",
        "model_seed": model_seed,
        "training": training,
        "checkpoint_reload": verification,
        "runtime": _runtime_metadata(requested, runtime_device),
        "controls": {
            "held_out_data_read": False,
            "ticket_20_executed": False,
            "ticket_44_executed": False,
            "ticket_45_executed": False,
        },
    }
    _write_json(destination / "run.json", run_payload)
    (destination / "run.md").write_text(
        "\n".join(
            (
                f"# Ticket 46 C0 Seed {model_seed}",
                "",
                "Status: completed.",
                f"Checkpoint reload verified: {str(all(verification.values())).lower()}.",
                f"Training valid: {str(training['criteria']['all_training_conditions_met']).lower()}.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def run_bridge_smoke(
    root: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    record_name: str = "smoke",
) -> dict[str, object]:
    """Run the preregistered non-package bridge smoke before formal rollout."""

    root_path = Path(root)
    if not (root_path / "frozen_instances.json").exists():
        raise FileNotFoundError("freeze the diagnostic package before the smoke")
    smoke_dir = root_path / "smoke"
    summary_path = smoke_dir / f"{record_name}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite bridge smoke: {summary_path}")
    requested, runtime_device = _resolve_single_cuda_device(device)
    candidates = build_unconditioned_relational_candidates(
        candidates_per_family=100,
        seed=int(C0_TRAINING_CONFIG["data_seed"]),
    )
    twins = _select_first_pairs_per_family(candidates, pairs_per_family=100)
    batch = _prepare_batch(
        _family_balanced_batches(twins, pairs_per_family_per_batch=100)[0],
        device=runtime_device,
    )
    model = build_context_ablation_model(
        "current_pair_aware", seed=MODEL_SEEDS[0], device=runtime_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _train_epoch(
        model,
        (batch,),
        optimizer,
        state_margin=0.1,
        flip_state_margin=0.1,
        pair_margin=0.1,
        flip_state_loss_weight=0.5,
        pair_loss_weight=0.25,
        saturation_loss_weight=0.1,
        saturation_penalty_allowance=0.25,
    )
    model.eval()
    with torch.no_grad():
        expected_scores = model(
            batch.robot_features,
            batch.task_features,
            batch.task_adjacency,
            md_inputs=batch.md_inputs,
        ).detach().cpu()
    smoke_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = smoke_dir / f"{record_name}_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    reloaded = build_context_ablation_model(
        "current_pair_aware", seed=MODEL_SEEDS[0], device=runtime_device
    ).eval()
    reloaded.load_state_dict(
        torch.load(checkpoint_path, map_location=runtime_device, weights_only=False)[
            "model_state_dict"
        ],
        strict=True,
    )
    with torch.no_grad():
        reloaded_scores = reloaded(
            batch.robot_features,
            batch.task_features,
            batch.task_adjacency,
            md_inputs=batch.md_inputs,
        ).detach().cpu()
    checkpoint_reload_verified = torch.equal(expected_scores, reloaded_scores)
    generated = generate_md_instance(
        MDGeneratorConfig(seed=45999, **NOMINAL_GENERATOR_CONFIG)
    )
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            reloaded, build_md_policy_inputs_from_simulator, device=runtime_device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=None,
        confidence_threshold=None,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    result = scheduler.run(
        generated.domain,
        run_id="ticket46-bridge-smoke",
        instance_id="md-c0-debug-45999",
        seed=45999,
        split=DatasetSplit.TEST,
        exit_location=EXIT_LOCATION,
    )
    summary: dict[str, object] = {
        "ticket": 46,
        "debug_instance_seed": 45999,
        "formal_package_observed": False,
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "success": result.experiment.success,
        "failure_reason": result.failure_reason,
        "decision_count": len(result.decision_records),
        "multiple_decision_points": len(result.decision_records) >= 2,
        "normal_path_solver_calls": len(result.fallback_records),
        "illegal_assignment_count": result.experiment.illegal_assignment_count,
        "runtime": _runtime_metadata(requested, runtime_device),
    }
    summary["passed"] = bool(
        summary["success"]
        and summary["checkpoint_reload_verified"]
        and summary["multiple_decision_points"]
        and summary["normal_path_solver_calls"] == 0
        and summary["illegal_assignment_count"] == 0
    )
    _write_json(summary_path, summary)
    if not summary["passed"]:
        raise RuntimeError("bridge smoke failed; formal rollout is blocked")
    return summary



def _checkpoint_path(root: Path, variant: str, model_seed: int) -> Path:
    if variant == "C0":
        return root / "training" / f"C0_seed{model_seed}" / "checkpoints" / "best_checkpoint.pt"
    return (
        REPOSITORY_ROOT
        / "reports"
        / "md_hyperparameter_sweep_2026-08-31"
        / "stage2"
        / f"{variant}_seed{model_seed}"
        / "checkpoints"
        / "best_checkpoint.pt"
    )


def _load_learned_checkpoint(
    root: Path, variant: str, model_seed: int, device: torch.device
):
    checkpoint_path = _checkpoint_path(root, variant, model_seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing {variant} checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_context_ablation_model(
        "current_pair_aware", seed=model_seed, device=device
    ).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {
        "variant": variant,
        "model_seed": model_seed,
        "path": str(checkpoint_path.resolve()),
        "strict_load": True,
        "selected_checkpoint": payload.get("selected_checkpoint"),
    }


def _preflight_learned_checkpoint_bridges(
    root: Path, *, device: torch.device
) -> list[dict[str, object]]:
    generated = generate_md_instance(
        MDGeneratorConfig(seed=45999, **NOMINAL_GENERATOR_CONFIG)
    )
    simulator = MDDiscreteSimulator(generated.domain, exit_location=EXIT_LOCATION)
    expected_shape = tuple(simulator_hard_mask(simulator).shape)
    checks: list[dict[str, object]] = []
    for variant in LEARNED_VARIANTS:
        for model_seed in MODEL_SEEDS:
            model, identity = _load_learned_checkpoint(
                root, variant, model_seed, device
            )
            scores = OnlineNeuralScoreProvider(
                model, build_md_policy_inputs_from_simulator, device=device
            )(simulator).scores
            if tuple(scores.shape) != expected_shape:
                raise RuntimeError(
                    f"{variant}_seed{model_seed} bridge score shape {tuple(scores.shape)} "
                    f"does not match simulator shape {expected_shape}"
                )
            checks.append(
                {
                    "checkpoint": f"{variant}_seed{model_seed}",
                    "checkpoint_identity": identity,
                    "debug_instance_seed": 45999,
                    "formal_package_observed": False,
                    "score_shape": list(scores.shape),
                    "scores_finite": True,
                }
            )
            del model
            torch.cuda.empty_cache()
    return checks


class _FixedHeuristicScorer:
    def __init__(self, method: str) -> None:
        self.method = method

    def __call__(self, simulator: MDDiscreteSimulator) -> ScoreOutput:
        feature_started = time.perf_counter()
        _robot, _task, _adjacency, md_inputs = (
            build_md_policy_inputs_from_simulator(simulator)
        )
        encoder_time = time.perf_counter() - feature_started
        scoring_started = time.perf_counter()
        robot_count = md_inputs.hard_feasibility_mask.shape[1]
        task_count = md_inputs.hard_feasibility_mask.shape[2]
        scores = torch.zeros(robot_count, task_count, dtype=torch.float32)
        if self.method != "masked_greedy":
            eta = md_inputs.pair_metadata[0, :, :, 0]
            utility = -eta / (1.0 + eta)
            if self.method == "eta_unlock_heuristic":
                assert md_inputs.opportunity_context is not None
                unlock = md_inputs.opportunity_context[
                    0, :, :, MDOpportunityFeature.LAST_MATERIAL_BLOCKER
                ]
                utility = utility + 0.6 * unlock
            transport_tasks = md_inputs.task_is_transport[0]
            scores[:, transport_tasks] = utility[:, transport_tasks]
        return ScoreOutput(
            scores,
            encoder_time,
            time.perf_counter() - scoring_started,
        )


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _online_raw_record(
    result,
    *,
    method: str,
    model_seed: int | None,
    instance: MDInstanceRecord,
    checkpoint_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    experiment = result.experiment
    decisions = result.decision_records
    fallbacks = result.fallback_records
    repair_count = sum(record.repaired for record in decisions)
    latency_totals = {
        "encoder_seconds": sum(record.encoder_time_seconds for record in decisions),
        "scoring_seconds": sum(record.scoring_time_seconds for record in decisions),
        "decoder_seconds": sum(record.decoder_time_seconds for record in decisions),
        "repair_seconds": sum(record.repair_time_seconds for record in decisions),
        "fallback_seconds": sum(record.fallback_time_seconds for record in decisions),
        "total_seconds": sum(record.total_time_seconds for record in decisions),
    }
    return {
        "schema_version": "1.0.0",
        "ticket": 46,
        "method": method,
        "model_seed": model_seed,
        "instance_id": instance.instance_id,
        "instance_seed": instance.seed,
        "success": experiment.success,
        "failure_reason": (
            None
            if experiment.failure_reason is None
            else experiment.failure_reason.value
        ),
        "makespan": experiment.makespan,
        "material_starvation": dict(experiment.material_starvation),
        "robot_utilization": dict(experiment.robot_utilization),
        "illegal_assignment_count": experiment.illegal_assignment_count,
        "repair_count": repair_count,
        "repair_rate": repair_count / len(decisions) if decisions else 0.0,
        "repair_event_count": sum(record.repair_event_count for record in decisions),
        "fallback_count": len(fallbacks),
        "fallback_rate": len(fallbacks) / len(decisions) if decisions else 0.0,
        "fallback_reasons": [record.trigger_reason for record in fallbacks],
        "solver_calls": len(fallbacks),
        "solver_time_seconds": sum(record.solver_time_seconds for record in fallbacks),
        "decision_count": len(decisions),
        "latency_totals": latency_totals,
        "wall_runtime_seconds": experiment.wall_time_seconds,
        "terminal_state": {
            "all_real_tasks_completed": experiment.all_real_tasks_completed,
            "all_robots_at_exit": experiment.all_robots_at_exit,
        },
        "checkpoint_identity": (
            None if checkpoint_identity is None else dict(checkpoint_identity)
        ),
        "decision_records": [
            {
                "decision_time": record.decision_time,
                "assignments": [list(item) for item in record.assignments],
                "confidence": _finite_or_none(record.confidence),
                "confidence_is_unbounded": bool(
                    record.confidence is not None and math.isinf(record.confidence)
                ),
                "encoder_time_seconds": record.encoder_time_seconds,
                "scoring_time_seconds": record.scoring_time_seconds,
                "decoder_time_seconds": record.decoder_time_seconds,
                "repair_time_seconds": record.repair_time_seconds,
                "fallback_time_seconds": record.fallback_time_seconds,
                "total_time_seconds": record.total_time_seconds,
                "fallback_used": record.fallback_used,
                "repaired": record.repaired,
                "repair_event_count": record.repair_event_count,
            }
            for record in decisions
        ],
        "experiment": experiment.to_dict(),
    }


def _run_process_md_greedy(instance: MDInstanceRecord) -> dict[str, object]:
    simulator = MDDiscreteSimulator(instance.generated.domain, exit_location=EXIT_LOCATION)
    transport = MDTransportGreedy(TransportGreedyStrategy.ETA)
    process = ProcessGreedyMDAdapter()
    started = time.perf_counter()
    inference_time = 0.0
    decisions: list[dict[str, object]] = []
    steps = 0
    failure_reason: FailureReason | None = None
    while not simulator.done and steps < MAX_ROLLOUT_STEPS:
        decision_started = time.perf_counter()
        transport_assignments = transport.assign(simulator)
        process_assignments = process.assign(simulator)
        decision_elapsed = time.perf_counter() - decision_started
        inference_time += decision_elapsed
        if transport_assignments or process_assignments:
            decisions.append(
                {
                    "decision_time": simulator.time,
                    "transport_assignment_count": len(transport_assignments),
                    "process_assignment_count": len(process_assignments),
                    "total_time_seconds": decision_elapsed,
                }
            )
        if not simulator.has_advancing_work:
            failure_reason = FailureReason.DEADLOCK
            break
        simulator.step()
        steps += 1
    if not simulator.done and failure_reason is None:
        failure_reason = FailureReason.TIMEOUT
    experiment = simulator.build_experiment_result(
        run_id=f"ticket46-{PRIMARY_FIXED_BASELINE}-{instance.instance_id}",
        method=PRIMARY_FIXED_BASELINE,
        instance_id=instance.instance_id,
        seed=instance.seed,
        split=instance.split,
        failure_reason=failure_reason,
        inference_time_seconds=inference_time,
        wall_time_seconds=time.perf_counter() - started,
        illegal_assignment_count=0,
        metadata={
            "hard_mask_source": "hard_feasibility",
            "terminal_protocol": "canonical_md_simulator",
            "adapter": "greedy_eta",
            "normal_path_uses_mip": False,
        },
    )
    return {
        "schema_version": "1.0.0",
        "ticket": 46,
        "method": PRIMARY_FIXED_BASELINE,
        "model_seed": None,
        "instance_id": instance.instance_id,
        "instance_seed": instance.seed,
        "success": experiment.success,
        "failure_reason": (
            None if experiment.failure_reason is None else experiment.failure_reason.value
        ),
        "makespan": experiment.makespan,
        "material_starvation": dict(experiment.material_starvation),
        "robot_utilization": dict(experiment.robot_utilization),
        "illegal_assignment_count": experiment.illegal_assignment_count,
        "repair_count": 0,
        "repair_rate": 0.0,
        "repair_event_count": 0,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "fallback_reasons": [],
        "solver_calls": 0,
        "solver_time_seconds": 0.0,
        "decision_count": len(decisions),
        "latency_totals": {
            "encoder_seconds": 0.0,
            "scoring_seconds": 0.0,
            "decoder_seconds": 0.0,
            "repair_seconds": 0.0,
            "fallback_seconds": 0.0,
            "total_seconds": inference_time,
        },
        "wall_runtime_seconds": experiment.wall_time_seconds,
        "terminal_state": {
            "all_real_tasks_completed": experiment.all_real_tasks_completed,
            "all_robots_at_exit": experiment.all_robots_at_exit,
        },
        "checkpoint_identity": None,
        "decision_records": decisions,
        "experiment": experiment.to_dict(),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw rollout output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _run_fixed_baseline(
    method: str, instance: MDInstanceRecord
) -> dict[str, object]:
    if method == PRIMARY_FIXED_BASELINE:
        return _run_process_md_greedy(instance)
    decoder = MaskedGreedyDecoder() if method == "masked_greedy" else LearnedConstrainedDecoder()
    scheduler = OnlineMDScheduler(
        scorer=_FixedHeuristicScorer(method),
        decoder=decoder,
        fallback=None,
        confidence_threshold=None,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    result = scheduler.run(
        instance.generated.domain,
        run_id=f"ticket46-{method}-{instance.instance_id}",
        instance_id=instance.instance_id,
        seed=instance.seed,
        split=instance.split,
        exit_location=EXIT_LOCATION,
    )
    return _online_raw_record(
        result,
        method=method,
        model_seed=None,
        instance=instance,
        checkpoint_identity=None,
    )


def _warm_up_model(model, device: torch.device) -> None:
    generated = generate_md_instance(
        MDGeneratorConfig(seed=45999, **NOMINAL_GENERATOR_CONFIG)
    )
    simulator = MDDiscreteSimulator(generated.domain, exit_location=EXIT_LOCATION)
    provider = OnlineNeuralScoreProvider(
        model, build_md_policy_inputs_from_simulator, device=device
    )
    for _ in range(LATENCY_WARMUP_FORWARDS):
        provider(simulator)


def _run_learned_variant(
    root: Path,
    *,
    variant: str,
    model_seed: int,
    records: Sequence[MDInstanceRecord],
    device: torch.device,
) -> list[dict[str, object]]:
    model, checkpoint_identity = _load_learned_checkpoint(
        root, variant, model_seed, device
    )
    _warm_up_model(model, device)
    scheduler = OnlineMDScheduler(
        scorer=OnlineNeuralScoreProvider(
            model, build_md_policy_inputs_from_simulator, device=device
        ),
        decoder=LearnedConstrainedDecoder(),
        fallback=ExplicitMIPFallback(threads=FALLBACK_THREADS),
        confidence_threshold=FALLBACK_CONFIDENCE_THRESHOLD,
        max_steps=MAX_ROLLOUT_STEPS,
    )
    rows = []
    for instance in records:
        result = scheduler.run(
            instance.generated.domain,
            run_id=f"ticket46-{variant}-{model_seed}-{instance.instance_id}",
            instance_id=instance.instance_id,
            seed=instance.seed,
            split=instance.split,
            exit_location=EXIT_LOCATION,
        )
        rows.append(
            _online_raw_record(
                result,
                method=variant,
                model_seed=model_seed,
                instance=instance,
                checkpoint_identity=checkpoint_identity,
            )
        )
    return rows


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _mean_or_none(values),
        "median": statistics.median(values) if values else None,
    }


def _row_starvation_mean(row: Mapping[str, object]) -> float:
    values = tuple(float(value) for value in dict(row["material_starvation"]).values())
    return statistics.fmean(values) if values else 0.0


def _aggregate_rollouts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = tuple(row for row in rows if bool(row["success"]))
    makespans = tuple(float(row["makespan"]) for row in successes if row["makespan"] is not None)
    failure_counts = Counter(
        str(row["failure_reason"])
        for row in rows
        if row["failure_reason"] is not None
    )
    return {
        "run_count": len(rows),
        "success_count": len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "failure_counts": dict(sorted(failure_counts.items())),
        "makespan_on_successes": _summary(makespans),
        "mean_material_starvation": _mean_or_none(
            [_row_starvation_mean(row) for row in rows]
        ),
        "mean_wall_runtime_seconds": _mean_or_none(
            [float(row["wall_runtime_seconds"]) for row in rows]
        ),
        "mean_total_decision_latency_seconds": _mean_or_none(
            [float(dict(row["latency_totals"])["total_seconds"]) for row in rows]
        ),
        "fallback_count": sum(int(row["fallback_count"]) for row in rows),
        "fallback_rate": _mean_or_none(
            [float(row["fallback_rate"]) for row in rows]
        ),
        "solver_calls": sum(int(row["solver_calls"]) for row in rows),
        "solver_time_seconds": sum(float(row["solver_time_seconds"]) for row in rows),
        "illegal_assignment_count": sum(
            int(row["illegal_assignment_count"]) for row in rows
        ),
        "repair_count": sum(int(row["repair_count"]) for row in rows),
        "repair_rate": _mean_or_none(
            [float(row["repair_rate"]) for row in rows]
        ),
    }


def _paired_starvation_difference(
    baseline: Sequence[Mapping[str, object]], candidate: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    base_by_id = {str(row["instance_id"]): row for row in baseline}
    candidate_by_id = {str(row["instance_id"]): row for row in candidate}
    if base_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired starvation inputs must have matching instance IDs")
    differences = [
        _row_starvation_mean(candidate_by_id[instance_id])
        - _row_starvation_mean(base_by_id[instance_id])
        for instance_id in sorted(base_by_id)
    ]
    return {
        "count": len(differences),
        "candidate_minus_baseline_mean": _mean_or_none(differences),
        "candidate_minus_baseline_median": (
            statistics.median(differences) if differences else None
        ),
        "bootstrap_ci95": _bootstrap_mean_ci(differences, seed=BOOTSTRAP_SEED + 7),
    }


def _paired_rollout_comparison(
    baseline: Sequence[Mapping[str, object]], candidate: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    base_summary = _aggregate_rollouts(baseline)
    candidate_summary = _aggregate_rollouts(candidate)
    makespan = paired_makespan_summary(baseline, candidate)
    candidate_minus_baseline = makespan["candidate_minus_baseline"]
    assert isinstance(candidate_minus_baseline, Mapping)
    difference = candidate_minus_baseline["mean"]
    return {
        "baseline": PRIMARY_FIXED_BASELINE,
        "success_rate_difference": (
            float(candidate_summary["success_rate"])
            - float(base_summary["success_rate"])
        ),
        "makespan": makespan,
        "joint_success_makespan_improvement": (
            None if difference is None else -float(difference)
        ),
        "starvation": _paired_starvation_difference(baseline, candidate),
        "candidate_fallback_rate": candidate_summary["fallback_rate"],
        "candidate_illegal_assignment_rate": (
            float(candidate_summary["illegal_assignment_count"])
            / len(candidate)
            if candidate
            else None
        ),
        "candidate_mean_total_latency_seconds": candidate_summary[
            "mean_total_decision_latency_seconds"
        ],
    }


def _gate_metrics(variant: str, model_seed: int) -> dict[str, object]:
    source = (
        REPOSITORY_ROOT
        / "reports"
        / "md_hyperparameter_sweep_2026-08-31"
        / "stage2"
        / f"{variant}_seed{model_seed}"
        / "run.json"
    )
    payload = _read_json(source)
    development = payload.get("development")
    training = payload.get("training")
    if not isinstance(development, Mapping) or not isinstance(training, Mapping):
        raise ValueError(f"Gate aggregate is malformed: {source}")
    criteria = training.get("criteria")
    if not isinstance(criteria, Mapping):
        raise ValueError(f"Gate training criteria are malformed: {source}")
    metrics = {
        "pickup_exact_pair_accuracy": development["pickup_exact_pair_accuracy"],
        "overall_exact_pair_accuracy": development["overall_exact_pair_accuracy"],
        "overall_state_agreement": development["overall_state_agreement"],
        "residual_saturation": development["residual_saturation_rate"],
        "training_valid": bool(criteria["all_training_conditions_met"]),
        "source": str(source.resolve()),
    }
    metrics["gate_pass"] = bool(
        metrics["training_valid"]
        and float(metrics["pickup_exact_pair_accuracy"]) >= 0.60
        and float(metrics["residual_saturation"]) <= 0.25
    )
    return metrics


def _rank_order(observations: Sequence[Mapping[str, object]], key: str) -> list[str]:
    def value(row: Mapping[str, object]) -> float:
        candidate = row[key]
        return float("-inf") if candidate is None else float(candidate)

    return [
        str(row["checkpoint"])
        for row in sorted(observations, key=lambda row: (-value(row), str(row["checkpoint"])))
    ]


def _gate_alignment(
    rollout_rows: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, object]:
    baseline = rollout_rows[PRIMARY_FIXED_BASELINE]
    observations: list[dict[str, object]] = []
    comparisons: dict[str, dict[str, object]] = {}
    for variant in LEARNED_VARIANTS:
        for model_seed in MODEL_SEEDS:
            checkpoint = f"{variant}_seed{model_seed}"
            comparison = _paired_rollout_comparison(baseline, rollout_rows[checkpoint])
            comparisons[checkpoint] = comparison
            gate = _gate_metrics(variant, model_seed)
            observations.append(
                {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "model_seed": model_seed,
                    "gate": gate,
                    "success_rate_difference": comparison["success_rate_difference"],
                    "joint_success_makespan_improvement": comparison[
                        "joint_success_makespan_improvement"
                    ],
                    "starvation_candidate_minus_baseline": comparison["starvation"][
                        "candidate_minus_baseline_mean"
                    ],
                    "fallback_rate": comparison["candidate_fallback_rate"],
                    "illegal_assignment_rate": comparison[
                        "candidate_illegal_assignment_rate"
                    ],
                    "mean_total_latency_seconds": comparison[
                        "candidate_mean_total_latency_seconds"
                    ],
                }
            )

    def correlation(gate_key: str, rollout_key: str) -> float | None:
        usable = [
            row
            for row in observations
            if row[rollout_key] is not None
        ]
        return spearman_rank_correlation(
            [float(dict(row["gate"])[gate_key]) for row in usable],
            [float(row[rollout_key]) for row in usable],
        )

    correlations = {
        "overall_exact_vs_success_difference": correlation(
            "overall_exact_pair_accuracy", "success_rate_difference"
        ),
        "overall_exact_vs_makespan_improvement": correlation(
            "overall_exact_pair_accuracy", "joint_success_makespan_improvement"
        ),
        "pickup_exact_vs_success_difference": correlation(
            "pickup_exact_pair_accuracy", "success_rate_difference"
        ),
        "pickup_exact_vs_makespan_improvement": correlation(
            "pickup_exact_pair_accuracy", "joint_success_makespan_improvement"
        ),
    }
    for row in observations:
        row["overall_exact_pair_accuracy"] = dict(row["gate"])[
            "overall_exact_pair_accuracy"
        ]
    gate_order = _rank_order(observations, "overall_exact_pair_accuracy")
    rollout_order = _rank_order(observations, "success_rate_difference")
    c0_observations = [row for row in observations if row["variant"] == "C0"]
    c0_success_improvements = sum(
        float(row["success_rate_difference"]) > 0 for row in c0_observations
    )
    c0_makespan_improvements = [
        float(row["joint_success_makespan_improvement"])
        for row in c0_observations
        if row["joint_success_makespan_improvement"] is not None
    ]
    c0_all_fail_gate = all(
        not bool(dict(row["gate"])["gate_pass"]) for row in c0_observations
    )
    possible_false_negative = bool(
        c0_all_fail_gate
        and c0_success_improvements >= 2
        and _mean_or_none(c0_makespan_improvements) is not None
        and float(_mean_or_none(c0_makespan_improvements)) > 0
    )
    insufficient_common_successes = any(
        int(comparison["makespan"]["common_successful_pairs"]) < 2
        for comparison in comparisons.values()
    )
    fallback_dominance = any(
        float(row["fallback_rate"]) > 0.5 for row in observations
    )
    key_correlations = (
        correlations["overall_exact_vs_success_difference"],
        correlations["overall_exact_vs_makespan_improvement"],
    )
    if possible_false_negative:
        interpretation = "possible_gate_false_negative_or_misalignment"
    elif any(value is not None and value < 0 for value in key_correlations):
        interpretation = "gate_proxy_not_predictive_in_pilot"
    elif insufficient_common_successes or fallback_dominance:
        interpretation = "pilot_inconclusive"
    elif all(value is not None and value >= 0 for value in key_correlations):
        interpretation = "gate_consistent_with_rollout"
    else:
        interpretation = "pilot_inconclusive"
    return {
        "schema_version": "1.0.0",
        "ticket": 46,
        "observation_count": len(observations),
        "descriptive_only": True,
        "baseline": PRIMARY_FIXED_BASELINE,
        "scatter_data": observations,
        "paired_comparisons": comparisons,
        "spearman_rank_correlations": correlations,
        "rank_agreement": {
            "gate_order": gate_order,
            "rollout_success_order": rollout_order,
            "same_top_checkpoint": gate_order[0] == rollout_order[0],
        },
        "c0_seed_correspondence": {
            "observations": c0_observations,
            "overall_exact_vs_success_difference": spearman_rank_correlation(
                [
                    float(dict(row["gate"])["overall_exact_pair_accuracy"])
                    for row in c0_observations
                ],
                [float(row["success_rate_difference"]) for row in c0_observations],
            ),
            "c0_success_improving_seed_count": c0_success_improvements,
            "c0_mean_joint_success_makespan_improvement": _mean_or_none(
                c0_makespan_improvements
            ),
        },
        "training_invalid_checkpoints": [
            row
            for row in observations
            if not bool(dict(row["gate"])["training_valid"])
        ],
        "inconclusive_conditions": {
            "insufficient_common_successes": insufficient_common_successes,
            "fallback_dominance": fallback_dominance,
        },
        "interpretation": interpretation,
        "positive_predictive_value_claim": False,
    }


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _next_step(interpretation: str, c0_comparisons: Sequence[Mapping[str, object]]) -> str:
    stable_success = sum(
        float(row["success_rate_difference"]) > 0 for row in c0_comparisons
    ) >= 2
    if interpretation in {
        "possible_gate_false_negative_or_misalignment",
        "gate_proxy_not_predictive_in_pilot",
    }:
        return "Modify the model or training objective only under a newly preregistered Gate."
    if interpretation == "gate_consistent_with_rollout" and not stable_success:
        return "Stop the learned method for this line of evidence."
    return "Prepare a new preregistered diagnostic protocol before any larger expert dataset or main experiment."


def _render_final_report(
    aggregates: Mapping[str, Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
    gate_alignment: Mapping[str, object],
) -> str:
    c0_keys = [f"C0_seed{seed}" for seed in MODEL_SEEDS]
    c0_comparisons = [comparisons[key] for key in c0_keys]
    c0_success = all(bool(aggregates[key]["success_count"]) for key in c0_keys)
    explicit_fallbacks = sum(int(aggregates[key]["solver_calls"]) for key in c0_keys)
    stable_better = sum(
        float(row["success_rate_difference"]) > 0 for row in c0_comparisons
    ) >= 2
    lines = [
        "# Ticket 46 C0 End-to-End Diagnostic Pilot",
        "",
        "This is a diagnostic pilot only. It does not select a final or production model.",
        "",
        "## Required Answers",
        "",
        f"1. Real C0 checkpoints completed continuous simulator rollouts for all C0 seeds: {str(c0_success).lower()}.",
        f"2. Normal-path solver calls were zero; explicit fallback solver calls across C0 were {explicit_fallbacks}.",
        "3. C0 seed outcomes:",
        "",
        "| Seed | Success rate | Mean success makespan | Mean starvation | Fallback rate | Mean decision latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed in MODEL_SEEDS:
        summary = aggregates[f"C0_seed{seed}"]
        makespan = dict(summary["makespan_on_successes"])["mean"]
        lines.append(
            "| "
            + str(seed)
            + " | "
            + _format_value(summary["success_rate"])
            + " | "
            + _format_value(makespan)
            + " | "
            + _format_value(summary["mean_material_starvation"])
            + " | "
            + _format_value(summary["fallback_rate"])
            + " | "
            + _format_value(summary["mean_total_decision_latency_seconds"])
            + " |"
        )
    lines.extend(
        [
            "",
            f"4. C0 stably improved success versus the frozen strongest fixed non-MIP baseline ({PRIMARY_FIXED_BASELINE}): {str(stable_better).lower()}.",
            f"5. Gate/rollout interpretation: {gate_alignment['interpretation']}.",
            f"6. Potential Gate false negative observed: {str(gate_alignment['interpretation'] == 'possible_gate_false_negative_or_misalignment').lower()}.",
            "7. Next step: " + _next_step(str(gate_alignment["interpretation"]), c0_comparisons),
            "8. Ticket 20 was not executed; Tickets 44 and 45 were not executed; Tickets 17, 20, and 32 were not unfrozen.",
            "",
            "Gate correlations are descriptive for n=9 observations and make no positive predictive value claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run_formal_diagnostic_pilot(
    root: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    smoke_record: str = "smoke",
    rollout_directory: str = "rollouts",
) -> dict[str, Path]:
    """Run the frozen paired package after the bridge smoke has passed."""

    root_path = Path(root)
    smoke_path = root_path / "smoke" / f"{smoke_record}.json"
    smoke = _read_json(smoke_path)
    if smoke.get("passed") is not True:
        raise RuntimeError("formal rollout requires a passing bridge smoke")
    targets = (
        root_path / "aggregate_metrics.json",
        root_path / "paired_comparisons.json",
        root_path / "gate_alignment.json",
        root_path / "runtime_and_hardware.json",
        root_path / "failures.json",
        root_path / "final_report.md",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("formal diagnostic outputs already exist")
    requested, runtime_device = _resolve_single_cuda_device(device)
    missing_checkpoints = [
        _checkpoint_path(root_path, variant, model_seed)
        for variant in LEARNED_VARIANTS
        for model_seed in MODEL_SEEDS
        if not _checkpoint_path(root_path, variant, model_seed).exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "formal rollout requires all frozen checkpoints: "
            + ", ".join(str(path) for path in missing_checkpoints)
        )
    try:
        bridge_preflight = _preflight_learned_checkpoint_bridges(
            root_path, device=runtime_device
        )
    except Exception as error:
        _write_json(
            root_path / "bridge_incompatible.json",
            {
                "ticket": 46,
                "status": "bridge_incompatible",
                "formal_package_observed": False,
                "error": str(error),
            },
        )
        raise RuntimeError("bridge_incompatible") from error
    gate_aggregate_preflight = {
        f"{variant}_seed{model_seed}": _gate_metrics(variant, model_seed)
        for variant in LEARNED_VARIANTS
        for model_seed in MODEL_SEEDS
    }
    records = load_frozen_diagnostic_records(root_path)
    rollout_root = root_path / rollout_directory
    rollout_rows: dict[str, list[dict[str, object]]] = {}
    for method in FIXED_BASELINES:
        rows = [_run_fixed_baseline(method, instance) for instance in records]
        rollout_rows[method] = rows
        _write_jsonl(rollout_root / f"{method}.jsonl", rows)
    for variant in LEARNED_VARIANTS:
        for model_seed in MODEL_SEEDS:
            key = f"{variant}_seed{model_seed}"
            rows = _run_learned_variant(
                root_path,
                variant=variant,
                model_seed=model_seed,
                records=records,
                device=runtime_device,
            )
            rollout_rows[key] = rows
            _write_jsonl(rollout_root / f"{key}.jsonl", rows)
            del rows
            torch.cuda.empty_cache()
    aggregates = {
        name: _aggregate_rollouts(rows) for name, rows in rollout_rows.items()
    }
    comparisons = {
        key: _paired_rollout_comparison(
            rollout_rows[PRIMARY_FIXED_BASELINE], rollout_rows[key]
        )
        for key in rollout_rows
        if key not in FIXED_BASELINES
    }
    gate_alignment = _gate_alignment(rollout_rows)
    failures = [
        row
        for rows in rollout_rows.values()
        for row in rows
        if not bool(row["success"])
    ]
    runtime = {
        "ticket": 46,
        "runtime": _runtime_metadata(requested, runtime_device),
        "formal_latency_runs_sequential": True,
        "warmup_decision_forwards_per_checkpoint": LATENCY_WARMUP_FORWARDS,
        "fallback_threads": FALLBACK_THREADS,
        "passing_smoke_record": str(smoke_path.relative_to(root_path)),
        "bridge_preflight": bridge_preflight,
        "rollout_directory": rollout_directory,
        "gate_aggregate_preflight": gate_aggregate_preflight,
        "cuda_memory_allocated_after_bytes": torch.cuda.memory_allocated(
            runtime_device
        ),
    }
    _write_json(
        root_path / "aggregate_metrics.json",
        {
            "ticket": 46,
            "baseline": PRIMARY_FIXED_BASELINE,
            "methods": aggregates,
            "all_frozen_instances_retained": True,
        },
    )
    _write_json(
        root_path / "paired_comparisons.json",
        {
            "ticket": 46,
            "baseline": PRIMARY_FIXED_BASELINE,
            "comparisons": comparisons,
        },
    )
    _write_json(root_path / "gate_alignment.json", gate_alignment)
    _write_json(root_path / "runtime_and_hardware.json", runtime)
    _write_json(root_path / "failures.json", {"ticket": 46, "failures": failures})
    (root_path / "final_report.md").write_text(
        _render_final_report(aggregates, comparisons, gate_alignment),
        encoding="utf-8",
    )
    return {
        "aggregate_metrics": root_path / "aggregate_metrics.json",
        "paired_comparisons": root_path / "paired_comparisons.json",
        "gate_alignment": root_path / "gate_alignment.json",
        "final_report": root_path / "final_report.md",
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ticket 46 C0 end-to-end diagnostic pilot runner."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "smoke", "train-c0", "run"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
        if command != "freeze":
            subparser.add_argument("--device", default="cuda:0")
        if command == "train-c0":
            subparser.add_argument(
                "--seed", type=int, choices=MODEL_SEEDS, required=True
            )
        if command == "smoke":
            subparser.add_argument("--record-name", default="smoke")
        if command == "run":
            subparser.add_argument(
                "--smoke-record", default="smoke"
            )
            subparser.add_argument(
                "--rollout-directory", default="rollouts"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    if args.command == "freeze":
        result: object = freeze_diagnostic_package(args.root)
    elif args.command == "smoke":
        result = run_bridge_smoke(
            args.root,
            device=args.device,
            record_name=args.record_name,
        )
    elif args.command == "train-c0":
        result = train_c0_seed(args.root, model_seed=args.seed, device=args.device)
    else:
        result = run_formal_diagnostic_pilot(
            args.root,
            device=args.device,
            smoke_record=args.smoke_record,
            rollout_directory=args.rollout_directory,
        )
    print(json.dumps(result, default=str, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FROZEN_INSTANCE_SEEDS",
    "freeze_diagnostic_package",
    "load_frozen_diagnostic_records",
    "main",
    "paired_makespan_summary",
    "run_bridge_smoke",
    "run_formal_diagnostic_pilot",
    "spearman_rank_correlation",
    "train_c0_seed",
]


if __name__ == "__main__":
    raise SystemExit(main())
