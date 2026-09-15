"""RL curriculum on scaled process_scarce distribution.

Rationale: 2026-09-14 scaled-target-profile eval confirmed MILP-IL v2 C0 has
strongest advantage-over-greedy on process_scarce (12-42 tasks). We want to
extend that advantage to larger scales (60+) where IL cannot get optimal
labels.

Approach:
- Reuse the existing md_ray_ppo.train pipeline unchanged.
- Monkey-patch its module-level MDGeneratorConfig to build scaled
  process_scarce configs instead of nominal balanced.
- Pass a tuned MDPPOExperimentConfig (larger_batch style: batch=2048,
  minibatch=256, num_epochs=30, il_kl_weight=0.005) to reduce per-iter
  compute at large task_count.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from data_generation.md_instance_generator import MDGeneratorConfig as _RealMDGeneratorConfig
from data_generation.md_instance_profiles import INSTANCE_PROFILES

from reinforcement_learning import md_ray_ppo
from reinforcement_learning.md_ray_ppo import MDPPOExperimentConfig


def _scaled_process_scarce_config(task_count: int, seed: int) -> _RealMDGeneratorConfig:
    base = dict(INSTANCE_PROFILES["process_scarce"].parameters)
    base_task_count = base["task_count"]
    scale = task_count / base_task_count
    process_tasks = int(task_count * (1 - base["transport_ratio"]))
    critical_path = min(
        max(base["critical_path_length"], round(base["critical_path_length"] * scale)),
        process_tasks,
    )
    return _RealMDGeneratorConfig(
        seed=seed,
        task_count=task_count,
        transport_ratio=base["transport_ratio"],
        precedence_density=base["precedence_density"],
        critical_path_length=critical_path,
        capacity_slack=base["capacity_slack"],
        speed_ratio=base["speed_ratio"],
        process_robot_count=max(1, round(base["process_robot_count"] * scale)),
        transport_robot_count=max(1, round(base["transport_robot_count"] * scale)),
        skill_count=base["skill_count"],
    )


def _install_scaled_config_shim(task_count: int) -> None:
    def _shim(*, seed: int) -> _RealMDGeneratorConfig:
        return _scaled_process_scarce_config(task_count, seed)
    md_ray_ppo.MDGeneratorConfig = _shim

def _install_env_runner_gpu_shim(num_gpus_per_env_runner: float) -> None:
    """Wrap PPOConfig.env_runners so it also sets num_gpus_per_env_runner.

    md_ray_ppo.train calls .env_runners(num_env_runners=N) without a GPU
    fraction; RLlib 2.49 accepts num_gpus_per_env_runner as an extra kwarg.
    We monkey-patch the class method so every .env_runners() call adds it.
    """
    if num_gpus_per_env_runner <= 0:
        return
    from ray.rllib.algorithms.ppo import PPOConfig
    original = PPOConfig.env_runners

    def _patched(self, **kwargs):
        kwargs.setdefault("num_gpus_per_env_runner", num_gpus_per_env_runner)
        return original(self, **kwargs)

    PPOConfig.env_runners = _patched



def _experiment_config(kl_weight: float) -> MDPPOExperimentConfig:
    # phase_a A1 defaults but with configurable kl_weight
    return MDPPOExperimentConfig(
        train_batch_size=2048,
        minibatch_size=256,
        num_epochs=5,
        learning_rate=1.0e-5,
        il_kl_weight=kl_weight,
        starvation_penalty=0.02,
        use_graph_value_head=False,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--task-count", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--kl-weight", type=float, default=0.005)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--num-gpus-per-env-runner", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    _install_scaled_config_shim(args.task_count)
    _install_env_runner_gpu_shim(args.num_gpus_per_env_runner)

    md_ray_ppo.train(
        seed=args.seed,
        il_checkpoint=args.il_checkpoint,
        iterations=args.iterations,
        num_env_runners=args.num_env_runners,
        num_gpus=args.num_gpus,
        output_dir=args.output_dir,
        kl_weight=args.kl_weight,
        experiment_config=_experiment_config(args.kl_weight),
    )


if __name__ == "__main__":
    main()
