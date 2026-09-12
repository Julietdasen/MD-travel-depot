from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_instance_profiles import (
    INSTANCE_PROFILES,
    md_generator_config_for_profile,
)
from simulation_environment.domain_model import ProcessRobot, TransportRobot, TransportTask


def test_balanced_profile_preserves_the_historical_default():
    assert md_generator_config_for_profile("balanced", 31) == MDGeneratorConfig(seed=31)


def test_profiles_generate_deterministic_task_oriented_instances():
    expected = {
        "balanced": (12, 3, 3, 2),
        "process_scarce": (12, 2, 2, 2),
        "transport_bottleneck": (12, 5, 3, 1),
        "dependency_deep": (12, 3, 3, 2),
        "mixed_hard": (15, 5, 3, 2),
        "scale_medium": (18, 6, 4, 3),
    }
    assert set(INSTANCE_PROFILES) == set(expected)
    for name, (tasks, transport_tasks, process_robots, transport_robots) in expected.items():
        config = md_generator_config_for_profile(name, 41)
        first = generate_md_instance(config)
        second = generate_md_instance(config)
        assert first == second
        assert len(first.domain.tasks) == tasks
        assert sum(isinstance(task, TransportTask) for task in first.domain.tasks) == transport_tasks
        assert sum(isinstance(robot, ProcessRobot) for robot in first.domain.robots) == process_robots
        assert sum(isinstance(robot, TransportRobot) for robot in first.domain.robots) == transport_robots
        assert config.skill_count == 3


def test_unknown_profile_is_rejected_with_available_names():
    try:
        md_generator_config_for_profile("missing", 1)
    except ValueError as error:
        assert "balanced" in str(error)
        assert "transport_bottleneck" in str(error)
    else:
        raise AssertionError("unknown profile must be rejected")
