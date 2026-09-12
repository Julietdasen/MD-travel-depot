"""Load and validate isolated MD scaling profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from data_generation.md_instance_generator import MDGeneratorConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPOSITORY_ROOT / "configs" / "md_scale_profiles.yaml"


def load_scale_profiles(path: str | Path = DEFAULT_PROFILE_PATH) -> dict[str, dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("profiles"), Mapping):
        raise ValueError("scale profile file must contain a profiles mapping")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("scale profile defaults must be a mapping")
    profiles: dict[str, dict[str, Any]] = {}
    for name, values in payload["profiles"].items():
        if not isinstance(name, str) or not name.strip() or not isinstance(values, Mapping):
            raise ValueError("each scale profile must have a name and mapping")
        merged = dict(defaults)
        merged.update(values)
        profiles[name] = merged
    if not profiles:
        raise ValueError("scale profile file must define at least one profile")
    return profiles


def generator_config_for_profile(
    profile: str,
    *,
    seed: int | None = None,
    path: str | Path = DEFAULT_PROFILE_PATH,
) -> MDGeneratorConfig:
    profiles = load_scale_profiles(path)
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"unknown profile {profile!r}; available profiles: {available}")
    values = profiles[profile]
    if seed is not None:
        values = {**values, "seed": seed}
    try:
        return MDGeneratorConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid configuration for profile {profile!r}: {error}") from error


def profile_summary(profile: str, config: MDGeneratorConfig) -> dict[str, Any]:
    return {
        "profile": profile,
        "seed": config.seed,
        "task_count": config.task_count,
        "process_robot_count": config.process_robot_count,
        "transport_robot_count": config.transport_robot_count,
        "robot_count": config.process_robot_count + config.transport_robot_count,
        "transport_task_count": config.transport_task_count,
        "process_task_count": config.process_task_count,
        "generator_config": {
            key: getattr(config, key)
            for key in (
                "seed", "task_count", "transport_ratio", "precedence_density",
                "critical_path_length", "capacity_slack", "speed_ratio",
                "process_robot_count", "transport_robot_count", "skill_count",
            )
        },
    }


__all__ = ["DEFAULT_PROFILE_PATH", "generator_config_for_profile", "load_scale_profiles", "profile_summary"]
