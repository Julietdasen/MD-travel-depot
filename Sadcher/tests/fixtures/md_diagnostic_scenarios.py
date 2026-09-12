"""Deterministic, CPU-only domains for MD semantic regression tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

from experiments.protocol import FailureReason
from simulation_environment.hard_feasibility import FeasibilityCode
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)


@dataclass(frozen=True, slots=True)
class DiagnosticScenario:
    name: str
    seed: int
    domain: SchedulingDomain
    exit_location: tuple[float, float]
    failure_reason: FailureReason | None
    terminal_metrics_expected: bool


@dataclass(frozen=True, slots=True)
class UnlockScenario(DiagnosticScenario):
    process_task_id: int
    process_robot_id: int
    transport_task_id: int
    transport_robot_id: int
    material_ready_at: int


@dataclass(frozen=True, slots=True)
class LateMaterialScenario(DiagnosticScenario):
    process_task_id: int
    normal_predecessor_id: int
    transport_task_id: int
    normal_robot_id: int
    transport_robot_id: int
    normal_ready_at: int
    material_ready_at: int
    expected_material_starvation: int


@dataclass(frozen=True, slots=True)
class AssignmentFailureScenario(DiagnosticScenario):
    transport_task_id: int
    rejected_robot_id: int
    expected_feasibility_code: FeasibilityCode


@dataclass(frozen=True, slots=True)
class CoalitionScenario(DiagnosticScenario):
    process_task_id: int
    first_robot_id: int
    second_robot_id: int
    expected_started_at: int


@dataclass(frozen=True, slots=True)
class TerminalFailureScenario(DiagnosticScenario):
    initial_assignments: tuple[tuple[int, int], ...]
    max_steps: int


_MD_CONFIG = MaterialDeliveryConfig(enabled=True)
_EXIT = (0.0, 0.0)


def _unlock_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=_MD_CONFIG,
        tasks=(
            ProcessTask(1, (3, 0), 2, (True,)),
            TransportTask(2, (1, 0), (3, 0), 5, 1, 1),
        ),
        robots=(
            ProcessRobot(0, (3, 0), (True,), speed=1),
            TransportRobot(1, (0, 0), True, 5, 1, loaded_speed=1),
        ),
        material_edges=((2, 1),),
    )


def _late_material_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=_MD_CONFIG,
        tasks=(
            ProcessTask(1, (4, 0), 2, (True,)),
            ProcessTask(2, (0, 0), 1, (True,)),
            TransportTask(3, (0, 0), (4, 0), 5, 1, 1),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True,), speed=1),
            TransportRobot(1, (0, 0), True, 5, 1, loaded_speed=1),
        ),
        normal_edges=((2, 1),),
        material_edges=((3, 1),),
    )


def _assignment_failure_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=_MD_CONFIG,
        tasks=(
            ProcessTask(1, (2, 0), 1, (True,)),
            TransportTask(2, (0, 0), (2, 0), 5, 1, 1),
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True,), speed=1),
            TransportRobot(1, (0, 0), True, 2, 1, loaded_speed=1),
            TransportRobot(2, (0, 0), True, 5, 1, loaded_speed=1),
        ),
        material_edges=((2, 1),),
    )


def _coalition_domain(*, task_count: int = 1) -> SchedulingDomain:
    return SchedulingDomain.create(
        config=_MD_CONFIG,
        tasks=tuple(
            ProcessTask(task_id, (0, 0), 2, (True, True))
            for task_id in range(1, task_count + 1)
        ),
        robots=(
            ProcessRobot(0, (0, 0), (True, False), speed=1),
            ProcessRobot(1, (0, 0), (False, True), speed=1),
        ),
    )


def _timeout_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=_MD_CONFIG,
        tasks=(ProcessTask(1, (0, 0), 10, (True,)),),
        robots=(ProcessRobot(0, (0, 0), (True,), speed=1),),
    )


UNLOCK = UnlockScenario(
    name="unlock-after-unloading",
    seed=1009,
    domain=_unlock_domain(),
    exit_location=_EXIT,
    failure_reason=None,
    terminal_metrics_expected=True,
    process_task_id=1,
    process_robot_id=0,
    transport_task_id=2,
    transport_robot_id=1,
    material_ready_at=5,
)

LATE_MATERIAL = LateMaterialScenario(
    name="late-material-after-normal-predecessor",
    seed=1010,
    domain=_late_material_domain(),
    exit_location=_EXIT,
    failure_reason=None,
    terminal_metrics_expected=True,
    process_task_id=1,
    normal_predecessor_id=2,
    transport_task_id=3,
    normal_robot_id=0,
    transport_robot_id=1,
    normal_ready_at=1,
    material_ready_at=7,
    expected_material_starvation=6,
)

CAPACITY_FAILURE = AssignmentFailureScenario(
    name="insufficient-transport-capacity",
    seed=1011,
    domain=_assignment_failure_domain(),
    exit_location=_EXIT,
    failure_reason=None,
    terminal_metrics_expected=False,
    transport_task_id=2,
    rejected_robot_id=1,
    expected_feasibility_code=FeasibilityCode.INSUFFICIENT_CAPACITY,
)

TYPE_MISMATCH = AssignmentFailureScenario(
    name="process-robot-on-transport-task",
    seed=1012,
    domain=_assignment_failure_domain(),
    exit_location=_EXIT,
    failure_reason=FailureReason.ROBOT_TYPE_MISMATCH,
    terminal_metrics_expected=False,
    transport_task_id=2,
    rejected_robot_id=0,
    expected_feasibility_code=FeasibilityCode.ROBOT_TYPE_MISMATCH,
)

_location_domain = _unlock_domain()
_location_process, _location_transport = _location_domain.tasks
assert isinstance(_location_transport, TransportTask)
LOCATION_MISMATCH = DiagnosticScenario(
    name="material-delivery-location-mismatch",
    seed=1013,
    domain=replace(
        _location_domain,
        tasks=(
            _location_process,
            replace(_location_transport, delivery_location=(4, 0)),
        ),
    ),
    exit_location=_EXIT,
    failure_reason=FailureReason.INVALID_GRAPH,
    terminal_metrics_expected=False,
)

COALITION = CoalitionScenario(
    name="two-skill-process-coalition",
    seed=1014,
    domain=_coalition_domain(),
    exit_location=_EXIT,
    failure_reason=None,
    terminal_metrics_expected=True,
    process_task_id=1,
    first_robot_id=0,
    second_robot_id=1,
    expected_started_at=1,
)

DEADLOCK = TerminalFailureScenario(
    name="split-incomplete-process-coalitions",
    seed=1015,
    domain=_coalition_domain(task_count=2),
    exit_location=_EXIT,
    failure_reason=FailureReason.DEADLOCK,
    terminal_metrics_expected=True,
    initial_assignments=((0, 1), (1, 2)),
    max_steps=1,
)

TIMEOUT = TerminalFailureScenario(
    name="long-process-timeout",
    seed=1016,
    domain=_timeout_domain(),
    exit_location=_EXIT,
    failure_reason=FailureReason.TIMEOUT,
    terminal_metrics_expected=True,
    initial_assignments=((0, 1),),
    max_steps=2,
)


__all__ = [
    "AssignmentFailureScenario",
    "CAPACITY_FAILURE",
    "COALITION",
    "CoalitionScenario",
    "DEADLOCK",
    "DiagnosticScenario",
    "LATE_MATERIAL",
    "LOCATION_MISMATCH",
    "LateMaterialScenario",
    "TerminalFailureScenario",
    "TIMEOUT",
    "TYPE_MISMATCH",
    "UNLOCK",
    "UnlockScenario",
]
