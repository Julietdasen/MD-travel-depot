from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_generation.md_instance_generator import generate_md_instance
from experiments.md_scale_profiles import generator_config_for_profile, load_scale_profiles
from simulation_environment.md_static_validator import require_valid_md_domain

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_have_expected_sizes_and_valid_domains():
    profiles = load_scale_profiles()
    assert set(profiles) == {"small", "medium", "large"}
    expected = {"small": (12, 3, 2), "medium": (30, 6, 4), "large": (60, 12, 8)}
    for name, values in expected.items():
        config = generator_config_for_profile(name)
        assert (config.task_count, config.process_robot_count, config.transport_robot_count) == values
        require_valid_md_domain(generate_md_instance(config).domain)


def test_profile_generation_is_deterministic_and_seed_override_changes_instance():
    config = generator_config_for_profile("medium")
    first = generate_md_instance(config)
    assert first.domain == generate_md_instance(config).domain
    assert first.domain != generate_md_instance(generator_config_for_profile("medium", seed=config.seed + 1)).domain


def test_invalid_profile_and_invalid_seed_are_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        generator_config_for_profile("unknown")
    with pytest.raises(ValueError):
        generator_config_for_profile("small", seed=-1)


def test_cli_lists_profiles_without_creating_outputs(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_md_scaled.py"), "--list-profiles", "--output-dir", str(tmp_path)], cwd=ROOT, check=True, capture_output=True, text=True)
    assert "small: tasks=12 robots=5" in completed.stdout
    assert "medium: tasks=30 robots=10" in completed.stdout
    assert not any(tmp_path.iterdir())


def test_cli_greedy_writes_isolated_result_and_refuses_overwrite(tmp_path):
    command = [sys.executable, str(ROOT / "experiments" / "run_md_scaled.py"), "--profile", "small", "--seed", "901", "--scheduler", "greedy", "--output-dir", str(tmp_path)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    run_dir = tmp_path / "small" / "seed-901" / "greedy"
    assert (run_dir / "instance.json").exists()
    assert (run_dir / "config.json").exists()
    result = json.loads((run_dir / "result.json").read_text())
    assert result["profile"]["robot_count"] == 5
    repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
