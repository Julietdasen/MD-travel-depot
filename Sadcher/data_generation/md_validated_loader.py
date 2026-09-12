"""Composition boundary for loading statically validated MD instances."""

from pathlib import Path

from data_generation.md_dataset import MDInstanceRecord, load_md_instance
from simulation_environment.md_static_validator import require_valid_md_domain


def load_validated_md_instance(path: str | Path) -> MDInstanceRecord:
    """Load an MD record and enforce static validation before rollout."""

    record = load_md_instance(path)
    require_valid_md_domain(record.generated.domain)
    return record
