"""Larger independent cohort for the rich-state prefix validation."""

from __future__ import annotations

import json

from experiments import md_rich_prefix_train_validate as experiment


experiment.TRAIN_POOL = range(76300, 76400)
experiment.DEVELOPMENT_POOL = range(76400, 76500)
experiment.TEST_POOL = range(76500, 76600)
experiment.SPLIT_COUNTS = {"train": 16, "development": 12, "test": 12}
experiment.OUTPUT = experiment.Path("reports/md_rich_prefix_train_validate_large_2026-09-10")


if __name__ == "__main__":
    print(json.dumps(experiment.run(), indent=2, sort_keys=True))
