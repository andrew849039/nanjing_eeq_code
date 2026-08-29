from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_metric_files(metrics_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "source": str(path),
            "model": payload.get("model", path.parent.name),
            "patch_tag": payload.get("patch_tag", path.parent.parent.name),
            "seed": payload.get("seed"),
        }
        for key in ["test_r2", "test_mae", "test_rmse", "test_oa", "test_kappa", "r2", "mae", "rmse", "oa", "kappa"]:
            if key in payload:
                row[key] = payload[key]
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_pairs = {
        "r2": "test_r2",
        "mae": "test_mae",
        "rmse": "test_rmse",
        "oa": "test_oa",
        "kappa": "test_kappa",
    }
    for src, dst in rename_pairs.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = normalize_metric_columns(df)
    metrics = [col for col in ["test_r2", "test_mae", "test_rmse", "test_oa", "test_kappa"] if col in df.columns]
    rows = []
    for (patch_tag, model), sub in df.groupby(["patch_tag", "model"], sort=False):
        row: dict[str, Any] = {"patch_tag": patch_tag, "model": model, "n_runs": int(len(sub))}
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model metrics into a comparison table.")
    parser.add_argument("--metrics-root", type=Path, default=PROJECT_ROOT / "outputs" / "metrics")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "tables" / "model_comparison_summary.csv")
    args = parser.parse_args()

    df = load_metric_files(args.metrics_root)
    summary = summarize(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8")

    if summary.empty:
        print(f"No metrics.json files found under {args.metrics_root}")
    else:
        print(summary.to_string(index=False))
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
