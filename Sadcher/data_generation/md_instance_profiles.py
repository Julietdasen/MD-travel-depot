"""Task-oriented synthetic instance families for MD-SADCHER experiments."""

from __future__ import annotations

from dataclasses import dataclass

from data_generation.md_instance_generator import MDGeneratorConfig


@dataclass(frozen=True, slots=True)
class MDInstanceProfile:
    name: str
    purpose: str
    parameters: tuple[tuple[str, int | float], ...]

    def config(self, seed: int) -> MDGeneratorConfig:
        return MDGeneratorConfig(seed=seed, **dict(self.parameters))


INSTANCE_PROFILES = {
    profile.name: profile
    for profile in (
        MDInstanceProfile(
            "balanced",
            "Historical 12-task baseline used by the current prefix experiments.",
            (
                ("task_count", 12),
                ("transport_ratio", 0.25),
                ("precedence_density", 0.25),
                ("critical_path_length", 3),
                ("capacity_slack", 0.2),
                ("speed_ratio", 0.8),
                ("process_robot_count", 3),
                ("transport_robot_count", 2),
                ("skill_count", 3),
            ),
        ),
        MDInstanceProfile(
            "process_scarce",
            "Stresses process coalition and idle decisions with fewer process robots.",
            (
                ("task_count", 12),
                ("transport_ratio", 1.0 / 6.0),
                ("precedence_density", 0.30),
                ("critical_path_length", 4),
                ("capacity_slack", 0.2),
                ("speed_ratio", 0.8),
                ("process_robot_count", 2),
                ("transport_robot_count", 2),
                ("skill_count", 3),
            ),
        ),
        MDInstanceProfile(
            "transport_bottleneck",
            "Stresses material unlock choices with one slow, capacity-tight carrier.",
            (
                ("task_count", 12),
                ("transport_ratio", 5.0 / 12.0),
                ("precedence_density", 0.25),
                ("critical_path_length", 3),
                ("capacity_slack", 0.0),
                ("speed_ratio", 0.55),
                ("process_robot_count", 3),
                ("transport_robot_count", 1),
                ("skill_count", 3),
            ),
        ),
        MDInstanceProfile(
            "dependency_deep",
            "Stresses normal/material lookahead with a deeper and denser process DAG.",
            (
                ("task_count", 12),
                ("transport_ratio", 0.25),
                ("precedence_density", 0.45),
                ("critical_path_length", 5),
                ("capacity_slack", 0.2),
                ("speed_ratio", 0.8),
                ("process_robot_count", 3),
                ("transport_robot_count", 2),
                ("skill_count", 3),
            ),
        ),
        MDInstanceProfile(
            "mixed_hard",
            "Combines coalition scarcity, transport pressure, and deep dependencies.",
            (
                ("task_count", 15),
                ("transport_ratio", 1.0 / 3.0),
                ("precedence_density", 0.40),
                ("critical_path_length", 5),
                ("capacity_slack", 0.05),
                ("speed_ratio", 0.60),
                ("process_robot_count", 3),
                ("transport_robot_count", 2),
                ("skill_count", 3),
            ),
        ),
        MDInstanceProfile(
            "scale_medium",
            "Measures neural generalization and latency; exhaustive action labeling is optional.",
            (
                ("task_count", 18),
                ("transport_ratio", 1.0 / 3.0),
                ("precedence_density", 0.30),
                ("critical_path_length", 6),
                ("capacity_slack", 0.10),
                ("speed_ratio", 0.70),
                ("process_robot_count", 4),
                ("transport_robot_count", 3),
                ("skill_count", 3),
            ),
        ),
    )
}


def md_generator_config_for_profile(name: str, seed: int) -> MDGeneratorConfig:
    try:
        profile = INSTANCE_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(INSTANCE_PROFILES)
        raise ValueError(f"unknown MD instance profile {name!r}; choose from: {choices}") from error
    return profile.config(seed)


__all__ = [
    "INSTANCE_PROFILES",
    "MDInstanceProfile",
    "md_generator_config_for_profile",
]
