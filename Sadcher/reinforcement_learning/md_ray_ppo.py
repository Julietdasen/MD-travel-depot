"""Ray RLlib PPO entry point for the MD Gym adapter."""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
import numpy as np
import torch
from pathlib import Path
import ray
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from models.md_rllib_model import MDRLlibModel, MDAutoregressiveActionDistribution
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from reinforcement_learning.md_gym_environment import MDGymEnvironment
from reinforcement_learning.md_finetune import (
    MDFineTuneConfig,
    MDRewardTracker,
    evaluate_go_no_go,
)

RAY_TEMP_DIR = os.environ.get("MD_RAY_TEMP_DIR", "/data/ZJZ/ray-runtime")


@dataclass(frozen=True, slots=True)
class MDPPOExperimentConfig:
    train_batch_size: int = 256
    minibatch_size: int = 128
    num_epochs: int = 30
    learning_rate: float = 1.0e-5
    il_kl_weight: float = 0.02
    starvation_penalty: float = 0.02
    use_graph_value_head: bool = False

    def __post_init__(self):
        for name in ("train_batch_size", "minibatch_size", "num_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.train_batch_size % self.minibatch_size:
            raise ValueError("train_batch_size must be a multiple of minibatch_size")
        for name in ("learning_rate", "il_kl_weight", "starvation_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")


def phase_a_experiment_configs() -> dict[str, MDPPOExperimentConfig]:
    baseline = MDPPOExperimentConfig()
    larger_batch = replace(
        baseline,
        train_batch_size=2048,
        minibatch_size=256,
        num_epochs=5,
        il_kl_weight=0.005,
    )
    return {
        "A0": baseline,
        "A1": larger_batch,
        "A2": replace(larger_batch, il_kl_weight=0.0),
        "A3": replace(larger_batch, starvation_penalty=0.0),
        "A4": replace(larger_batch, use_graph_value_head=True),
    }


def reward_tracker_for_experiment(
    greedy_makespan: float, experiment: MDPPOExperimentConfig
) -> MDRewardTracker:
    return MDRewardTracker(
        greedy_makespan=greedy_makespan,
        config=MDFineTuneConfig(starvation_penalty=experiment.starvation_penalty),
    )


class MDIllegalAssignmentCallbacks(RLlibCallback):
    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for() or {}
        episode.custom_metrics["illegal_assignment_count"] = float(
            info.get("illegal_assignments", 0)
        )


def instance_seed_splits(model_seed: int) -> dict[str, tuple[int, ...]]:
    base = model_seed * 1_000
    return {
        "train": tuple(range(base, base + 64)),
        "validation": tuple(range(base + 100, base + 116)),
        "held_out": tuple(range(base + 200, base + 216)),
    }


def _number(value):
    return None if value is None else float(value)


def extract_learner_diagnostics(result):
    policy_stats = (
        result.get("info", {})
        .get("learner", {})
        .get("default_policy", {})
    )
    learner_stats = policy_stats.get("learner_stats", {})
    names = {
        "ppo_kl": "kl",
        "entropy": "entropy",
        "value_loss": "vf_loss",
        "value_explained_variance": "vf_explained_var",
        "policy_loss": "policy_loss",
        "ppo_kl_coefficient": "cur_kl_coeff",
        "il_reference_kl": "il_reference_kl",
    }
    diagnostics = {
        output_name: _number(learner_stats.get(source_name))
        for output_name, source_name in names.items()
    }
    if diagnostics["il_reference_kl"] is None:
        diagnostics["il_reference_kl"] = _number(
            policy_stats.get("model", {}).get("il_reference_kl")
        )
    return diagnostics


def is_better_validation(candidate, best, *, il_success_rate):
    if candidate["illegal_assignments"] != 0:
        return False
    if candidate["success_rate"] < il_success_rate:
        return False
    if best is None:
        return True
    ranking_fields = (
        "mean_makespan",
        "mean_material_starvation",
        "p95_latency_seconds",
    )
    return tuple(candidate[field] for field in ranking_fields) < tuple(
        best[field] for field in ranking_fields
    )


def _actor_parameter_snapshot(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("net.") and parameter.requires_grad
    }


def _actor_parameter_drift(model, initial):
    current = dict(model.named_parameters())
    if not initial:
        return {"max_absolute": 0.0, "normalized_l2": 0.0}
    max_absolute = max(
        (current[name].detach().cpu() - before).abs().max().item()
        for name, before in initial.items()
    )
    difference_norm = sum(
        (current[name].detach().cpu() - before).double().square().sum().item()
        for name, before in initial.items()
    ) ** 0.5
    initial_norm = sum(
        before.double().square().sum().item() for before in initial.values()
    ) ** 0.5
    return {
        "max_absolute": float(max_absolute),
        "normalized_l2": float(difference_norm / max(initial_norm, 1.0e-12)),
    }


def _policy_action(algo, observation, *, reference):
    model = algo.get_policy().model
    device = next(model.parameters()).device
    batch = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }
    with torch.no_grad():
        output, _ = model({"obs": batch}, [], None)
        logits = model._last_reference_logits.reshape(1, -1) if reference else output
        action = MDAutoregressiveActionDistribution(logits, model).deterministic_sample()
    return action[0].cpu().numpy()


def rollout_metrics(algo, seeds, *, reference=False, latency_sink=None):
    rows = []
    latencies = []
    action_comparisons = 0
    action_disagreements = 0
    for seed in seeds:
        domain = generate_md_instance(MDGeneratorConfig(seed=seed)).domain
        env = MDGymEnvironment(lambda domain=domain: domain)
        obs, _ = env.reset()
        _policy_action(algo, obs, reference=reference)
        terminated = truncated = False
        while not (terminated or truncated):
            started = time.perf_counter()
            action = _policy_action(algo, obs, reference=reference)
            latencies.append(time.perf_counter() - started)
            if reference:
                rl_action = _policy_action(algo, obs, reference=False)
                action_comparisons += 1
                action_disagreements += int(not np.array_equal(action, rl_action))
            obs, _reward, terminated, truncated, _info = env.step(action)
        rows.append({"seed": seed, "success": bool(terminated), "makespan": float(env.sim.time) if terminated else None, "illegal_assignments": env.illegal_assignment_count, "material_starvation": float(sum(env.sim._material_starvation().values()))})
    if latency_sink is not None:
        latency_sink.extend(latencies)
    successful = [row["makespan"] for row in rows if row["makespan"] is not None]
    return {"rows": rows, "success_rate": sum(row["success"] for row in rows) / len(rows), "mean_makespan": float(np.mean(successful)) if successful else None, "illegal_assignments": sum(row["illegal_assignments"] for row in rows), "mean_material_starvation": float(np.mean([row["material_starvation"] for row in rows])), "p95_latency_seconds": float(np.percentile(latencies, 95)), "latency_warmup_calls_per_episode": 1, "action_comparisons": action_comparisons, "action_disagreements": action_disagreements}


def paired_rollout_metrics(algo, seeds):
    il_rows = []
    rl_rows = []
    il_latencies = []
    rl_latencies = []
    action_comparisons = 0
    action_disagreements = 0
    for index, seed in enumerate(seeds):
        order = ((True, il_rows, il_latencies), (False, rl_rows, rl_latencies))
        if index % 2:
            order = tuple(reversed(order))
        for reference, rows, latency_sink in order:
            metrics = rollout_metrics(
                algo, (seed,), reference=reference, latency_sink=latency_sink
            )
            rows.extend(metrics["rows"])
            if reference:
                action_comparisons += metrics["action_comparisons"]
                action_disagreements += metrics["action_disagreements"]

    def summarize(rows, latencies, *, include_disagreement=False):
        successful = [row["makespan"] for row in rows if row["makespan"] is not None]
        summary = {
            "rows": sorted(rows, key=lambda row: row["seed"]),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_makespan": float(np.mean(successful)) if successful else None,
            "illegal_assignments": sum(row["illegal_assignments"] for row in rows),
            "mean_material_starvation": float(np.mean([row["material_starvation"] for row in rows])),
            "p95_latency_seconds": float(np.percentile(latencies, 95)),
            "latency_warmup_calls_per_episode": 1,
            "evaluation_order": "IL-first on even seed index; RL-first on odd seed index",
        }
        if include_disagreement:
            summary["deterministic_action_disagreement"] = {
                "comparisons": action_comparisons,
                "disagreements": action_disagreements,
                "rate": action_disagreements / action_comparisons if action_comparisons else 0.0,
            }
        return summary
    return summarize(il_rows, il_latencies, include_disagreement=True), summarize(rl_rows, rl_latencies)


def held_out_comparison(il_metrics, rl_metrics):
    result = {"il": il_metrics, "rl": rl_metrics, "go_no_go": {"go": False, "reasons": ["no_successful_rollout"]}}
    if il_metrics["mean_makespan"] is not None and rl_metrics["mean_makespan"] is not None:
        paired = zip(il_metrics["rows"], rl_metrics["rows"], strict=True)
        differences = [il["makespan"] - rl["makespan"] for il, rl in paired]
        result["paired_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        }
        decision = evaluate_go_no_go(il_makespan=il_metrics["mean_makespan"], rl_makespan=rl_metrics["mean_makespan"], il_success_rate=il_metrics["success_rate"], rl_success_rate=rl_metrics["success_rate"], rl_illegal_assignments=rl_metrics["illegal_assignments"], il_material_starvation=il_metrics["mean_material_starvation"], rl_material_starvation=rl_metrics["mean_material_starvation"], rl_p95_latency=rl_metrics["p95_latency_seconds"], il_p95_latency=il_metrics["p95_latency_seconds"])
        result["go_no_go"] = {"go": decision.go, "reasons": list(decision.reasons)}
    return result


def train(
    seed: int = 0,
    *,
    il_checkpoint: str | None = None,
    iterations: int = 1,
    num_env_runners: int = 0,
    num_gpus: float = 1.0,
    freeze_encoders: bool = True,
    kl_weight: float = 0.02,
    output_dir: str = "runs/md_ray_ppo",
    experiment_config: MDPPOExperimentConfig | None = None,
    phase_a_cell: str | None = None,
):
    experiment = experiment_config or replace(
        MDPPOExperimentConfig(), il_kl_weight=kl_weight
    )
    splits = instance_seed_splits(seed)
    output_dir = str(Path(output_dir).resolve())
    il_checkpoint = str(Path(il_checkpoint).resolve()) if il_checkpoint else None

    def env_creator(config):
        seeds = config.get("instance_seeds", splits["train"])
        position = 0

        def domain_factory():
            nonlocal position
            instance_seed = seeds[position % len(seeds)]
            position += 1
            return generate_md_instance(MDGeneratorConfig(seed=instance_seed)).domain

        return MDGymEnvironment(
            domain_factory,
            reward_tracker_factory=lambda greedy: reward_tracker_for_experiment(
                greedy, experiment
            ),
        )

    env_name = "MDGymEnvironment-v0"
    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model("md_il_policy", MDRLlibModel)
    ModelCatalog.register_custom_action_dist(
        "md_autoregressive", MDAutoregressiveActionDistribution
    )
    Path(RAY_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=RAY_TEMP_DIR)
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={"instance_seeds": splits["train"]},
            disable_env_checking=True,
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(num_env_runners=num_env_runners)
        .resources(num_gpus=num_gpus)
        .training(
            lr=experiment.learning_rate,
            gamma=0.99,
            train_batch_size=experiment.train_batch_size,
            num_epochs=experiment.num_epochs,
            minibatch_size=experiment.minibatch_size,
        )
        .callbacks(MDIllegalAssignmentCallbacks)
        .debugging(seed=seed)
    )
    config.model.update(
        {
            "custom_model": "md_il_policy",
            "custom_action_dist": "md_autoregressive",
            "custom_model_config": {
                "il_checkpoint": il_checkpoint,
                "freeze_encoders": freeze_encoders,
                "kl_weight": experiment.il_kl_weight,
                "use_graph_value_head": experiment.use_graph_value_head,
            },
        }
    )
    algo = config.build()
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "seed_splits.json").write_text(json.dumps({key: list(value) for key, value in splits.items()}, indent=2) + "\n")
        metadata = {
            "il_checkpoint": il_checkpoint,
            "ray_version": ray.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "model_seed": seed,
            "split_seeds": {key: list(value) for key, value in splits.items()},
            "kl_weight": experiment.il_kl_weight,
            "experiment_config": asdict(experiment),
            "phase_a_cell": phase_a_cell,
            "freeze_encoders": freeze_encoders,
            "iterations": iterations,
            "ray_temp_dir": RAY_TEMP_DIR,
            "best_iteration": None,
        }
        Path(output_dir, "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        policy_model = algo.get_policy().model
        actor_before = _actor_parameter_snapshot(policy_model)
        encoder_parameter_ids = {
            id(parameter)
            for module in policy_model._encoder_modules()
            for parameter in module.parameters()
        }
        frozen_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in policy_model.named_parameters()
            if id(parameter) in encoder_parameter_ids
        }
        history = []
        best_validation_metrics = None
        best_validation_comparison = None
        best_checkpoint_path = None
        best_iteration = None
        for _ in range(iterations):
            result = algo.train()
            runners = result.get("env_runners", result.get("sampler_results", {}))
            row = {
                "iteration": int(result["training_iteration"]),
                "train_reward_mean": _number(
                    runners.get("episode_return_mean", runners.get("episode_reward_mean"))
                ),
                "train_episode_len_mean": _number(runners.get("episode_len_mean")),
                **extract_learner_diagnostics(result),
                "actor_parameter_drift": _actor_parameter_drift(
                    algo.get_policy().model, actor_before
                ),
                "validation": None,
            }
            should_validate = row["iteration"] % 10 == 0 or row["iteration"] == iterations
            if should_validate:
                il_validation, rl_validation = paired_rollout_metrics(
                    algo, splits["validation"]
                )
                validation_comparison = held_out_comparison(
                    il_validation, rl_validation
                )
                validation_comparison["deterministic_action_disagreement"] = (
                    il_validation["deterministic_action_disagreement"]
                )
                row["validation"] = validation_comparison
                if is_better_validation(
                    rl_validation,
                    best_validation_metrics,
                    il_success_rate=il_validation["success_rate"],
                ):
                    saved = algo.save(
                        str(Path(output_dir, f"best_iter_{row['iteration']:04d}"))
                    )
                    best_checkpoint_path = getattr(
                        getattr(saved, "checkpoint", saved), "path", str(saved)
                    )
                    best_iteration = row["iteration"]
                    best_validation_metrics = rl_validation
                    best_validation_comparison = validation_comparison
            history.append(row)
            if row["iteration"] == 1 or row["iteration"] % 10 == 0:
                print(row, flush=True)
        Path(output_dir, "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        final_checkpoint = algo.save(str(Path(output_dir, "final_checkpoint")))
        if best_checkpoint_path is None:
            best_checkpoint_path = getattr(getattr(final_checkpoint, "checkpoint", final_checkpoint), "path", str(final_checkpoint))
            best_iteration = iterations
        algo.restore(best_checkpoint_path)
        restored_model = algo.get_policy().model
        frozen_after = dict(restored_model.named_parameters())
        frozen_encoder_verified = all(
            torch.equal(before, frozen_after[name].detach().cpu())
            for name, before in frozen_before.items()
        )
        evaluation_split = "validation" if phase_a_cell else "held_out"
        if phase_a_cell and best_validation_comparison is not None and best_iteration == iterations:
            comparison = best_validation_comparison
        else:
            il_metrics, rl_metrics = paired_rollout_metrics(
                algo, splits[evaluation_split]
            )
            comparison = held_out_comparison(il_metrics, rl_metrics)
            comparison["deterministic_action_disagreement"] = (
                il_metrics["deterministic_action_disagreement"]
            )
        comparison.update(
            {
                "model_seed": seed,
                "evaluation_split": evaluation_split,
                "best_validation_iteration": best_iteration,
                "best_checkpoint": best_checkpoint_path,
                "checkpoint_restore_succeeded": True,
                "frozen_encoder_verified": frozen_encoder_verified,
                "actor_parameter_drift": _actor_parameter_drift(
                    restored_model, actor_before
                ),
            }
        )
        metadata["best_iteration"] = best_iteration
        metadata["best_checkpoint"] = best_checkpoint_path
        metadata["checkpoint_restore_succeeded"] = True
        metadata["frozen_encoder_verified"] = frozen_encoder_verified
        metadata["evaluation_split"] = evaluation_split
        metadata["evaluation_order"] = "alternating IL-first/RL-first by seed"
        metadata["latency_warmup_calls_per_episode"] = 1
        Path(output_dir, "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        comparison_name = (
            "validation_comparison.json" if phase_a_cell else "held_out_comparison.json"
        )
        Path(output_dir, comparison_name).write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        print(json.dumps(comparison, indent=2), flush=True)
        return best_checkpoint_path
    finally:
        algo.stop(); ray.shutdown()

def _build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--phase-a-cell", choices=tuple(phase_a_experiment_configs()))
    return parser


def main(argv=None):
    args = _build_cli_parser().parse_args(argv)
    experiment = (
        phase_a_experiment_configs()[args.phase_a_cell]
        if args.phase_a_cell
        else None
    )
    iterations = args.iterations if args.iterations is not None else (5 if experiment else 1)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            f"runs/md_ray_ppo_phase_a_{args.phase_a_cell}_seed{args.seed}"
            if args.phase_a_cell
            else "runs/md_ray_ppo"
        )
    train(
        args.seed,
        il_checkpoint=args.il_checkpoint,
        kl_weight=args.kl_weight,
        iterations=iterations,
        num_env_runners=args.num_env_runners,
        num_gpus=args.num_gpus,
        output_dir=output_dir,
        experiment_config=experiment,
        phase_a_cell=args.phase_a_cell,
    )


if __name__ == "__main__":
    main()
