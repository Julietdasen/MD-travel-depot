"""Validate a fixed-order prefix model on the larger rich-state cohort."""

from __future__ import annotations

import json
import shutil

from experiments import md_prefix_conditioned_reranker as reranker
from experiments import md_rich_prefix_train_validate as experiment


experiment.TRAIN_POOL = range(76300, 76400)
experiment.DEVELOPMENT_POOL = range(76400, 76500)
experiment.TEST_POOL = range(76500, 76600)
experiment.SPLIT_COUNTS = {"train": 16, "development": 12, "test": 12}
experiment.OUTPUT = experiment.Path("reports/md_rich_prefix_canonical_validate_2026-09-10")
experiment.MODES = ("prefix_zero", "prefix")
reranker.PREFIX_CANONICAL_ONLY = True


if __name__ == "__main__":
    source = experiment.Path("reports/md_rich_prefix_train_validate_large_2026-09-10/rich_exact_labels.jsonl")
    experiment.OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, experiment.OUTPUT / "rich_exact_labels.jsonl")
    print(json.dumps(experiment.run(), indent=2, sort_keys=True))
