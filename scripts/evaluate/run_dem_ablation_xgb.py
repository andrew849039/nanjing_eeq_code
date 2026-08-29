from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class XgbParams:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.30
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    n_jobs: int = -1


def score_to_grade(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 0.0, 1.0)
    grades = np.floor(np.clip(clipped, 0.0, 0.999999) / 0.2).astype(np.int32) + 1
    grades[clipped >= 1.0] = 5
    return np.clip(grades, 1, 5)


def center_features(patch_dir: Path, split: str) -> np.ndarray:
    patches = np.load(patch_dir / f"features_{split}.npy", mmap_mode="r")
    _, _, height, width = patches.shape
    return np.asarray(patches[:, :, height // 2, width // 2], dtype=np.float32)


def labels(patch_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    meta = pd.read_csv(patch_dir / f"meta_{split}.csv")
    y_score = meta["center_score"].to_numpy(dtype=np.float32)
    y_grade = meta["center_grade"].to_numpy(dtype=np.int32)
    return y_score, y_grade, meta


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_grade: np.ndarray) -> dict[str, float]:
    pred = np.clip(y_pred, 0.0, 1.0)
    pred_grade = score_to_grade(pred)
    return {
        "r2": float(r2_score(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "oa": float((pred_grade == y_grade).mean()),
        "kappa": float(cohen_kappa_score(y_grade, pred_grade, labels=[1, 2, 3, 4, 5])),
    }


def fit_one(patch_dir: Path, seed: int, params: XgbParams, output_dir: Path, dataset_name: str) -> dict[str, float | int | str]:
    x_train = center_features(patch_dir, "train")
    y_train, _, _ = labels(patch_dir, "train")
    x_test = center_features(patch_dir, "test")
    y_test, y_grade_test, meta_test = labels(patch_dir, "test")

    model = XGBRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_lambda=params.reg_lambda,
        objective=params.objective,
        tree_method=params.tree_method,
        n_jobs=params.n_jobs,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(np.float32)
    result = metrics(y_test, pred, y_grade_test)

    seed_dir = output_dir / dataset_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    out = meta_test.loc[:, ["row", "col", "block_id", "center_score", "center_grade"]].copy()
    out["pred_score"] = np.clip(pred, 0.0, 1.0)
    out["pred_grade"] = score_to_grade(pred)
    out.to_csv(seed_dir / "test_predictions.csv", index=False)
    (seed_dir / "metrics.json").write_text(
        json.dumps({"seed": seed, "dataset": dataset_name, **result}, indent=2),
        encoding="utf-8",
    )
    return {"seed": seed, "dataset": dataset_name, **result}


def summarize(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = ["r2", "rmse", "mae", "oa", "kappa"]
    grouped = []
    for dataset, sub in df.groupby("dataset", sort=False):
        item: dict[str, float | int | str] = {"dataset": dataset, "n_seeds": int(len(sub))}
        for col in metric_cols:
            item[f"{col}_mean"] = float(sub[col].mean())
            item[f"{col}_std"] = float(sub[col].std(ddof=1))
        grouped.append(item)
    return pd.DataFrame(grouped)


def paired_stats(df: pd.DataFrame, baseline: str, dem: str) -> dict[str, dict[str, float]]:
    metric_cols = ["r2", "rmse", "mae", "oa", "kappa"]
    base = df[df["dataset"] == baseline].sort_values("seed")
    aug = df[df["dataset"] == dem].sort_values("seed")
    if list(base["seed"]) != list(aug["seed"]):
        raise ValueError("Seed lists do not align between paired datasets.")

    stats: dict[str, dict[str, float]] = {}
    for col in metric_cols:
        delta = aug[col].to_numpy(dtype=float) - base[col].to_numpy(dtype=float)
        try:
            p_value = float(wilcoxon(aug[col], base[col], zero_method="wilcox").pvalue)
        except ValueError:
            p_value = 1.0
        stats[col] = {
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(ddof=1)),
            "wilcoxon_p": p_value,
        }
    return stats


def parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run paired XGBoost ablation with and without terrain predictors.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48")
    parser.add_argument("--run-tag", default="dem_ablation_xgb_s7_fast")
    parser.add_argument("--n-estimators", type=int, default=XgbParams.n_estimators)
    parser.add_argument("--max-depth", type=int, default=XgbParams.max_depth)
    parser.add_argument("--learning-rate", type=float, default=XgbParams.learning_rate)
    parser.add_argument("--subsample", type=float, default=XgbParams.subsample)
    parser.add_argument("--colsample-bytree", type=float, default=XgbParams.colsample_bytree)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    output_dir = project_root / "outputs" / "metrics" / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "xgb_rgbn_common": project_root / "data" / "processed" / "patches" / "rgbn_strict_common",
        "xgb_rgbn_dem": project_root / "data" / "processed" / "patches" / "rgbn_dem_strict",
    }
    params = XgbParams(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )
    seeds = parse_seeds(args.seeds)

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for name, patch_dir in datasets.items():
            print(f"[run] dataset={name} seed={seed}", flush=True)
            rows.append(fit_one(patch_dir, seed, params, output_dir, name))

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)
    paired = paired_stats(per_seed, "xgb_rgbn_common", "xgb_rgbn_dem")
    (output_dir / "paired_wilcoxon.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")

    manifest = {
        "run_tag": args.run_tag,
        "project_root": str(project_root),
        "datasets": {key: str(value) for key, value in datasets.items()},
        "seeds": seeds,
        "xgb_params": asdict(params),
        "outputs": {
            "per_seed_metrics": str(output_dir / "per_seed_metrics.csv"),
            "summary_mean_std": str(output_dir / "summary_mean_std.csv"),
            "paired_wilcoxon": str(output_dir / "paired_wilcoxon.json"),
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(paired, indent=2), flush=True)


if __name__ == "__main__":
    main()
