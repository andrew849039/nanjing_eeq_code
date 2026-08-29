from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanjing_eeq.metrics import compute_all_metrics, score_to_grade  # noqa: E402


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def center_features(patch_dir: Path, split: str) -> np.ndarray:
    patches = np.load(patch_dir / f"features_{split}.npy", mmap_mode="r")
    _, _, height, width = patches.shape
    return np.asarray(patches[:, :, height // 2, width // 2], dtype=np.float32)


def labels(patch_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    meta = pd.read_csv(patch_dir / f"meta_{split}.csv")
    y_score = meta["center_score"].to_numpy(dtype=np.float32)
    y_grade = meta["center_grade"].to_numpy(dtype=np.int32)
    return y_score, y_grade, meta


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_model(name: str, seed: int):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is required for the xgboost baseline.")
        return XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.30,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown baseline model: {name}")


def run_one(model_name: str, seed: int, patch_dir: Path, output_root: Path, patch_tag: str) -> dict[str, float | int | str]:
    x_train = center_features(patch_dir, "train")
    y_train, _, _ = labels(patch_dir, "train")
    x_test = center_features(patch_dir, "test")
    y_test, y_grade_test, meta_test = labels(patch_dir, "test")

    model = build_model(model_name, seed)
    model.fit(x_train, y_train)
    pred_score = np.clip(model.predict(x_test).astype(np.float32), 0.0, 1.0)
    pred_grade = score_to_grade(pred_score)
    metrics = compute_all_metrics(y_test, pred_score, y_grade_test)

    run_dir = output_root / patch_tag / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_df = meta_test.loc[:, ["row", "col", "block_id", "center_score", "center_grade"]].copy()
    pred_df["pred_score"] = pred_score
    pred_df["pred_grade"] = pred_grade
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8")

    payload = {"model": model_name, "patch_tag": patch_tag, "seed": seed, **metrics}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = ["r2", "mae", "rmse", "oa", "kappa"]
    summary_rows = []
    for (patch_tag, model), sub in df.groupby(["patch_tag", "model"], sort=False):
        row: dict[str, float | int | str] = {"patch_tag": patch_tag, "model": model, "n_seeds": int(len(sub))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train center-pixel tabular baselines on the shared patch split.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument("--models", default="rf,xgboost")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48")
    parser.add_argument("--patch-dir", type=Path, default=None)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    project_root = resolve_path(PROJECT_ROOT, config["project"]["root"])
    patch_dir = args.patch_dir or resolve_path(project_root, config["data"]["patch_dir"])
    patch_tag = args.run_tag or config["data"]["patch_tag"]
    output_root = resolve_path(project_root, config["outputs"]["metrics_root"])

    rows: list[dict[str, float | int | str]] = []
    for seed in parse_seeds(args.seeds):
        for model_name in parse_csv_list(args.models):
            print(f"[run] model={model_name} seed={seed}", flush=True)
            rows.append(run_one(model_name, seed, patch_dir, output_root, patch_tag))

    summary = summarize(rows)
    summary_dir = output_root / "tabular_baselines"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "summary_mean_std.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
