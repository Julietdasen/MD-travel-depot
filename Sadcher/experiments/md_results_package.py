"""One-command, path-portable MD-SADCHER++ results package rebuild."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.protocol import PROTOCOL_VERSION, describe


RESULTS_PACKAGE_VERSION = "1.0.0"
REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "seeds",
    "generation_config",
    "baseline_config",
    "checkpoint",
    "environment",
    "raw_results",
}


def rebuild_results_package(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_mapping(manifest_file)
    _validate_manifest(manifest, manifest_file)
    raw_path = manifest_file.parent / str(manifest["raw_results"])
    raw = _load_sequence(raw_path)
    records = tuple(_validate_result(item) for item in raw)
    summary = _summarize(manifest, records)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": destination / "summary.json",
        "tables": destination / "tables.md",
        "plot": destination / "quality_overview.png",
        "diagnostics": destination / "diagnostics.json",
        "failures": destination / "failures.json",
        "environment": destination / "environment.json",
        "raw_results": destination / "raw_results.json",
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["diagnostics"], _diagnostics(records))
    _write_json(paths["failures"], _failures(records))
    _write_json(paths["environment"], _runtime_environment(manifest))
    _write_json(paths["raw_results"], list(records))
    paths["tables"].write_text(_markdown_tables(summary), encoding="utf-8")
    _plot_summary(summary, paths["plot"])
    return paths


def _validate_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise ValueError(f"results manifest missing keys: {sorted(missing)}")
    if manifest["schema_version"] != RESULTS_PACKAGE_VERSION:
        raise ValueError("unsupported results package schema")
    if manifest["status"] not in {"complete", "formal_experiments_pending"}:
        raise ValueError("results status must be complete or formal_experiments_pending")
    seeds = manifest["seeds"]
    if (
        not isinstance(seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("manifest seeds must be unique non-negative integers")
    for key in ("generation_config", "baseline_config", "environment"):
        if not isinstance(manifest[key], dict):
            raise ValueError(f"{key} must be a mapping")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict) or not {
        "identifier",
        "path",
    }.issubset(checkpoint):
        raise ValueError("checkpoint must record identifier and path")
    for reference in (manifest["raw_results"], checkpoint["path"]):
        if reference is None:
            continue
        candidate = Path(str(reference))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("manifest file references must be portable relative paths")
    raw_path = path.parent / str(manifest["raw_results"])
    if not raw_path.is_file():
        raise ValueError(f"raw results file is missing: {manifest['raw_results']}")


def _validate_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("each raw result must be a mapping")
    for key in ("protocol_version", "run_id", "method", "dataset", "termination", "metrics", "metadata"):
        if key not in value:
            raise ValueError(f"raw result missing {key}")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("raw result protocol version mismatch")
    dataset = value["dataset"]
    termination = value["termination"]
    metrics = value["metrics"]
    if not all(isinstance(item, dict) for item in (dataset, termination, metrics)):
        raise ValueError("raw result dataset, termination, and metrics must be mappings")
    success = termination.get("success")
    makespan = metrics.get("makespan")
    if success is True:
        _finite_non_negative(makespan, "successful makespan")
        if termination.get("failure_reason") is not None:
            raise ValueError("successful raw result cannot have failure_reason")
    elif success is False:
        if makespan is not None or not termination.get("failure_reason"):
            raise ValueError("failed raw result needs null makespan and failure_reason")
    else:
        raise ValueError("raw result success must be boolean")
    return value


def _summarize(manifest: Mapping[str, Any], records: Sequence[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[str(record["method"])].append(record)
    methods = {}
    for method, runs in sorted(by_method.items()):
        successful = [run for run in runs if run["termination"]["success"]]
        failures = Counter(
            run["termination"]["failure_reason"]
            for run in runs
            if not run["termination"]["success"]
        )
        methods[method] = {
            "runs": len(runs),
            "successful_runs": len(successful),
            "success_rate": len(successful) / len(runs),
            "failure_counts": dict(failures),
            "makespan": describe(
                run["metrics"]["makespan"] for run in successful
            ).to_dict(),
            "material_starvation": describe(
                value
                for run in successful
                for value in run["metrics"].get("material_starvation", {}).values()
            ).to_dict(),
            "robot_utilization": describe(
                value
                for run in successful
                for value in run["metrics"].get("robot_utilization", {}).values()
            ).to_dict(),
            "inference_time_seconds": describe(
                run["metrics"].get("inference_time_seconds", 0.0) for run in runs
            ).to_dict(),
            "gurobi_gap": describe(
                run["metadata"]["gurobi_gap"]
                for run in runs
                if run["metadata"].get("gurobi_gap") is not None
            ).to_dict(),
        }
    return {
        "schema_version": RESULTS_PACKAGE_VERSION,
        "status": manifest["status"],
        "formal_experiments_pending": manifest["status"] == "formal_experiments_pending",
        "seeds": manifest["seeds"],
        "generation_config": manifest["generation_config"],
        "baseline_config": manifest["baseline_config"],
        "checkpoint": manifest["checkpoint"],
        "raw_result_count": len(records),
        "methods": methods,
    }


def _failures(records: Sequence[dict]) -> dict:
    failed = [record for record in records if not record["termination"]["success"]]
    timeouts = [record for record in failed if record["termination"]["failure_reason"] == "timeout"]
    infeasible = [
        record
        for record in failed
        if record["termination"]["failure_reason"] in {
            "static_infeasible",
            "no_capable_transport_robot",
            "robot_type_mismatch",
            "invalid_graph",
        }
        or record["metadata"].get("oracle_status") == "infeasible"
    ]
    gaps = [
        {
            "run_id": record["run_id"],
            "method": record["method"],
            "instance_id": record["dataset"]["instance_id"],
            "seed": record["dataset"]["seed"],
            "gurobi_gap": record["metadata"]["gurobi_gap"],
        }
        for record in records
        if record["metadata"].get("gurobi_gap") is not None
    ]
    return {
        "timeouts": timeouts,
        "infeasible": infeasible,
        "gurobi_gaps": gaps,
        "other_failures": [
            record for record in failed if record not in timeouts and record not in infeasible
        ],
    }


def _diagnostics(records: Sequence[dict]) -> dict:
    failures = [record for record in records if not record["termination"]["success"]]
    successes = [record for record in records if record["termination"]["success"]]
    worst = sorted(
        successes,
        key=lambda record: record["metrics"]["makespan"],
        reverse=True,
    )[:10]
    return {"failure_cases": failures, "highest_makespan_cases": worst}


def _markdown_tables(summary: Mapping[str, Any]) -> str:
    lines = [
        "# MD-SADCHER++ Results",
        "",
        f"Status: {summary['status']}",
        "",
        "| Method | Runs | Success | Makespan mean | Starvation mean | Utilization mean | Inference mean (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        lines.append(
            "| "
            + " | ".join(
                (
                    method,
                    str(values["runs"]),
                    _format(values["success_rate"]),
                    _format(values["makespan"]["mean"]),
                    _format(values["material_starvation"]["mean"]),
                    _format(values["robot_utilization"]["mean"]),
                    _format(values["inference_time_seconds"]["mean"]),
                )
            )
            + " |"
        )
    if not summary["methods"]:
        lines.append("| No formal results yet | 0 | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines) + "\n"


def _plot_summary(summary: Mapping[str, Any], path: Path) -> None:
    methods = list(summary["methods"])
    success = [summary["methods"][name]["success_rate"] for name in methods]
    makespan = [
        summary["methods"][name]["makespan"]["mean"] or 0.0 for name in methods
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    if methods:
        axes[0].bar(methods, success, color="#287271")
        axes[1].bar(methods, makespan, color="#D9822B")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "Formal experiments pending", ha="center", va="center")
            axis.set_xticks([])
    axes[0].set_title("Success rate")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Successful makespan mean")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _runtime_environment(manifest: Mapping[str, Any]) -> dict:
    return {
        "declared": manifest["environment"],
        "observed": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable_name": Path(sys.executable).name,
            "matplotlib": matplotlib.__version__,
        },
    }


def _load_mapping(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return payload


def _load_sequence(path: Path) -> list:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite_non_negative(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be non-negative and finite")


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the MD-SADCHER++ results package")
    parser.add_argument("--manifest", default="results/md_sadcher_manifest.json")
    parser.add_argument("--output-dir", default="results/rebuilt")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = rebuild_results_package(args.manifest, args.output_dir)
    print(paths["summary"])


if __name__ == "__main__":
    main()

