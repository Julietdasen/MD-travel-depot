"""Gymnasium adapter for the validated material-delivery simulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from reinforcement_learning.md_finetune import MDRewardTracker
from reinforcement_learning.md_joint_action import joint_action_is_legal
from experiments.protocol import DatasetSplit
from schedulers.md_greedy_baselines import run_greedy_eta
from simulation_environment.domain_model import ProcessRobot, ProcessTask, SchedulingDomain, TransportRobot, TransportTask
from simulation_environment.hard_feasibility import TaskStatus
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from models.md_online_features import build_md_policy_inputs_from_simulator


class MDGymEnvironment(gym.Env):
    """Centralized one-decision-per-tick environment for MD policy training."""

    metadata = {"render_modes": []}

    def __init__(self, domain_factory: Callable[[], SchedulingDomain], *, reward_tracker_factory: Callable[[float], MDRewardTracker] | None = None):
        super().__init__()
        self.domain_factory = domain_factory
        domain = domain_factory()
        self._configure_spaces(domain)
        self.reward_tracker_factory = reward_tracker_factory
        self.sim: MDDiscreteSimulator | None = None

    def _configure_spaces(self, domain: SchedulingDomain) -> None:
        self.task_ids = tuple(sorted(task.task_id for task in domain.tasks))
        self.robot_ids = tuple(sorted(robot.robot_id for robot in domain.robots))
        robot_count, task_count = len(self.robot_ids), len(self.task_ids)
        box = gym.spaces.Box
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(robot_count, task_count + 1, dtype=np.int64)
        )
        self.observation_space = gym.spaces.Dict({
            "robot_features": box(-np.inf, np.inf, (robot_count, 7), dtype=np.float32),
            "task_features": box(-np.inf, np.inf, (task_count, 9), dtype=np.float32),
            "task_adjacency": box(0.0, 1.0, (task_count, task_count), dtype=np.float32),
            "robot_metadata": box(-np.inf, np.inf, (robot_count, 4), dtype=np.float32),
            "task_metadata": box(-np.inf, np.inf, (task_count, 8), dtype=np.float32),
            "pair_metadata": box(-np.inf, np.inf, (robot_count, task_count, 5), dtype=np.float32),
            "task_is_transport": box(0.0, 1.0, (task_count,), dtype=np.float32),
            "typed_adjacency": box(0.0, 1.0, (2, task_count, task_count), dtype=np.float32),
            "downstream_task_index": box(-1.0, float(task_count - 1), (task_count,), dtype=np.float32),
            "opportunity_context": box(-np.inf, np.inf, (robot_count, task_count, 5), dtype=np.float32),
            "process_covered_skills": box(0.0, 1.0, (task_count, 3), dtype=np.float32),
            "action_mask": box(0.0, 1.0, (robot_count, task_count + 1), dtype=np.float32),
        })

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        domain = self.domain_factory()
        task_ids = tuple(sorted(task.task_id for task in domain.tasks))
        robot_ids = tuple(sorted(robot.robot_id for robot in domain.robots))
        if task_ids != self.task_ids or robot_ids != self.robot_ids:
            raise ValueError("domain_factory must preserve the fixed robot/task shape")
        self._tasks = {task.task_id: task for task in domain.tasks}
        self._robots = {robot.robot_id: robot for robot in domain.robots}
        self.sim = MDDiscreteSimulator(domain)
        baseline = run_greedy_eta(
            MDDiscreteSimulator(domain), run_id="md-rl-reward-baseline",
            instance_id="md-rl-episode", seed=0, split=DatasetSplit.TRAIN,
            max_steps=10_000,
        )
        greedy = baseline.makespan if baseline.success else None
        if greedy is None or greedy <= 0:
            raise RuntimeError("greedy ETA baseline failed for generated MD domain")
        self._max_episode_steps = max(1_000, int(3 * greedy))
        self.reward_tracker = self.reward_tracker_factory(greedy) if self.reward_tracker_factory else MDRewardTracker(greedy_makespan=greedy)
        self.reward_tracker.reset()
        self.illegal_assignment_count = 0
        return self._observation(), {}

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (len(self.robot_ids),):
            raise ValueError("action must contain one task index per robot")
        observation = self._observation()
        tensor_observation = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in observation.items()
        }
        tensor_action = torch.as_tensor(action).unsqueeze(0)
        assert joint_action_is_legal(tensor_observation, tensor_action), (
            "env.step received an action outside the autoregressive support"
        )
        for robot_id, choice in zip(self.robot_ids, action, strict=True):
            if choice == 0:
                continue
            task_id = self.task_ids[int(choice) - 1]
            result = self.sim.assignment_feasibility(robot_id=robot_id, task_id=task_id)
            assert result.is_feasible, result
            self.sim.assign(robot_id=robot_id, task_id=task_id)
        before = sum(state.status is TaskStatus.COMPLETE for state in self.sim.task_states.values())
        self.sim.step()
        after = sum(state.status is TaskStatus.COMPLETE for state in self.sim.task_states.values())
        starvation = self.sim._material_starvation()
        reward = self.reward_tracker.step(
            time=self.sim.time,
            completed_tasks=after,
            starvation=starvation,
            terminal=self.sim.done,
            makespan=float(self.sim.time) if self.sim.done else None,
            illegal_assignments=0,
        )
        terminated = self.sim.done
        truncated = self.sim.time >= self._max_episode_steps and not terminated
        return self._observation(), reward, terminated, truncated, {
            "illegal_assignments": 0,
            "completed_delta": after - before,
            "executed_action": action.tolist(),
        }

    def _observation(self) -> dict[str, np.ndarray]:
        assert self.sim is not None
        robot, task, adjacency, md = build_md_policy_inputs_from_simulator(self.sim)
        hard_mask = md.hard_feasibility_mask[0]
        # The IL policy was trained without an idle class. Preserve that
        # contract: idle is legal only when a robot has no feasible task.
        idle = ~hard_mask.any(dim=-1, keepdim=True)
        action_mask = torch.cat((idle, hard_mask), dim=-1)
        covered = torch.zeros(len(self.task_ids), 3)
        for task_index, task_id in enumerate(self.task_ids):
            task_entity = self._tasks[task_id]
            if not isinstance(task_entity, ProcessTask):
                continue
            for robot_id in self.sim.task_states[task_id].assigned_robot_ids:
                robot_entity = self._robots[robot_id]
                if isinstance(robot_entity, ProcessRobot):
                    capabilities = torch.tensor(robot_entity.capabilities, dtype=torch.float32)
                    covered[task_index, : len(capabilities)] = torch.maximum(
                        covered[task_index, : len(capabilities)], capabilities
                    )
        return {
            "robot_features": robot[0].numpy(),
            "task_features": task[0].numpy(),
            "task_adjacency": adjacency[0].numpy(),
            "robot_metadata": md.robot_metadata[0].numpy(),
            "task_metadata": md.task_metadata[0].numpy(),
            "pair_metadata": md.pair_metadata[0].numpy(),
            "task_is_transport": md.task_is_transport[0].float().numpy(),
            "typed_adjacency": md.typed_adjacency[0].numpy(),
            "downstream_task_index": md.downstream_task_index[0].float().numpy(),
            "opportunity_context": md.opportunity_context[0].numpy(),
            "process_covered_skills": covered.numpy(),
            "action_mask": action_mask.float().numpy(),
        }
