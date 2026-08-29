from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def inspect_array(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    array = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "exists": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
    }


def inspect_table(path: Path, required_columns: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "missing_columns": sorted(required_columns)}
    df = pd.read_csv(path, nrows=5)
    columns = set(df.columns)
    return {
        "path": str(path),
        "exists": True,
        "columns": list(df.columns),
        "missing_columns": sorted(required_columns.difference(columns)),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
    }


def inspect_patch_dataset(patch_dir: Path) -> dict[str, Any]:
    required_meta = {"row", "col", "block_id", "center_score", "center_grade"}
    splits = ["train", "val", "test"]
    result: dict[str, Any] = {
        "patch_dir": str(patch_dir),
        "exists": patch_dir.exists(),
        "feature_stats": {"path": str(patch_dir / "feature_stats.json"), "exists": (patch_dir / "feature_stats.json").exists()},
        "splits": {},
    }
    for split in splits:
        result["splits"][split] = {
            "features": inspect_array(patch_dir / f"features_{split}.npy"),
            "metadata": inspect_table(patch_dir / f"meta_{split}.csv", required_meta),
        }
    return result


def collect_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    patch = report["patch_dataset"]
    if not patch["exists"]:
        issues.append(f"Missing patch directory: {patch['patch_dir']}")
    if not patch["feature_stats"]["exists"]:
        issues.append(f"Missing feature statistics: {patch['feature_stats']['path']}")
    for split, split_report in patch["splits"].items():
        if not split_report["features"]["exists"]:
            issues.append(f"Missing {split} feature array: {split_report['features']['path']}")
        missing = split_report["metadata"].get("missing_columns", [])
        if not split_report["metadata"]["exists"]:
            issues.append(f"Missing {split} metadata table: {split_report['metadata']['path']}")
        elif missing:
            issues.append(f"Missing columns in {split} metadata: {missing}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the expected local data layout without loading full datasets into memory.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit code if required files are missing.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    project_root = resolve_path(PROJECT_ROOT, config["project"]["root"])
    patch_dir = resolve_path(project_root, config["data"]["patch_dir"])

    report = {
        "project_root": str(project_root),
        "config": str(args.config),
        "patch_dataset": inspect_patch_dataset(patch_dir),
        "optional_inputs": {
            "boundary": str(project_root / "data" / "raw" / "boundary" / "nanjing_boundary.geojson"),
            "stack_dir": str(project_root / "data" / "processed" / "stacks"),
            "weak_label_dir": str(project_root / "data" / "processed" / "weak_labels"),
            "npp_grouped_cells": str(project_root / "data" / "external" / "npp_grouped_cells.csv"),
        },
    }
    report["issues"] = collect_issues(report)

    out_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_layout_audit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
