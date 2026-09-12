"""Legacy-style carry-as-a-skill comparison for MD instances.

This diagnostic baseline deliberately removes material readiness and
downstream unlock semantics. It is not a valid material-delivery scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.protocol import DatasetSplit, ExperimentResult
from schedulers.process_greedy_md import run_process_greedy_md
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator
from simulation_environment.transport_timing import transport_durations


@dataclass(frozen=True, slots=True)
class CarryAsSkillProjection:
    """Process-only projection plus an audit trail of discarded semantics."""

    domain: SchedulingDomain
    carry_skill_index: int
    ignored_material_edges: tuple[tuple[int, int], ...]


def project_carry_as_skill_domain(domain: SchedulingDomain) -> CarryAsSkillProjection:
    """Treat each transport as an ordinary task requiring one carry skill."""

    if not domain.config.enabled:
        raise ValueError("carry-as-skill requires an enabled MD domain")

    skill_count = max(
        (
            len(task.requirements)
            for task in domain.tasks
            if isinstance(task, ProcessTask)
        ),
        default=0,
    )
    tasks: list[ProcessTask] = []
    for task in domain.tasks:
        if isinstance(task, ProcessTask):
            requirements = tuple(task.requirements) + (False,) * (
                skill_count - len(task.requirements)
            )
            tasks.append(
                ProcessTask(
                    task.task_id,
                    task.location,
                    task.duration,
                    requirements + (False,),
                )
            )
        else:
            tasks.append(
                ProcessTask(
                    task.task_id,
                    task.delivery_location,
                    _carry_service_duration(task, domain),
                    (False,) * skill_count + (True,),
                )
            )

    robots: list[ProcessRobot] = []
    for robot in domain.robots:
        if isinstance(robot, ProcessRobot):
            capabilities = tuple(robot.capabilities) + (False,) * (
                skill_count - len(robot.capabilities)
            )
            robots.append(
                ProcessRobot(
                    robot.robot_id,
                    robot.location,
                    capabilities + (False,),
                    robot.speed,
                )
            )
        else:
            robots.append(
                ProcessRobot(
                    robot.robot_id,
                    robot.location,
                    (False,) * skill_count + (robot.transport_capable,),
                    robot.unloaded_speed,
                )
            )

    projection = SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=False),
        tasks=tasks,
        robots=robots,
        normal_edges=domain.normal_edges,
    )
    return CarryAsSkillProjection(
        projection, skill_count, domain.material_edges
    )


def run_carry_as_skill_baseline(
    domain: SchedulingDomain,
    *,
    exit_location: tuple[float, float] = (0.0, 0.0),
    run_id: str,
    instance_id: str,
    seed: int,
    split: DatasetSplit,
    max_steps: int,
) -> ExperimentResult:
    """Run the invalid-semantics comparison under the canonical protocol."""

    projection = project_carry_as_skill_domain(domain)
    simulator = MDDiscreteSimulator(projection.domain, exit_location=exit_location)
    return run_process_greedy_md(
        simulator,
        run_id=run_id,
        instance_id=instance_id,
        seed=seed,
        split=split,
        max_steps=max_steps,
        method="carry_as_skill",
        metadata={
            "uses_material_ready_semantics": False,
            "uses_downstream_unlock_semantics": False,
            "uses_capacity_feasibility": False,
            "ignored_material_edges": projection.ignored_material_edges,
            "carry_skill_index": projection.carry_skill_index,
            "source_domain_is_md": True,
        },
    )


def _carry_service_duration(task: TransportTask, domain: SchedulingDomain) -> int:
    eligible = tuple(
        robot
        for robot in domain.robots
        if isinstance(robot, TransportRobot)
        and robot.transport_capable
        and robot.capacity >= task.load
        and robot.loaded_speed is not None
    )
    if not eligible:
        raise ValueError(f"TRANSPORT task {task.task_id} has no eligible solo robot")
    return min(
        transport_durations(robot.location, robot, task).service
        for robot in eligible
    )


__all__ = [
    "CarryAsSkillProjection",
    "project_carry_as_skill_domain",
    "run_carry_as_skill_baseline",
]
