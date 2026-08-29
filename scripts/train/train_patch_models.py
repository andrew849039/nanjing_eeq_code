from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanjing_eeq.data import FeatureStats, PatchCenterDataset
from nanjing_eeq.metrics import compute_all_metrics, score_to_grade
from nanjing_eeq.models import PatchMLP, PlainUNet, center_slice


@dataclass
class RunConfig:
    model: str
    patch_dir: str
    output_dir: str
    device: str
    batch_size: int
    num_workers: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    optimizer: str
    selection_metric: str
    seed: int
    amp: bool
    base_channels: int
    mlp_hidden_dims: list[int]
    patch_size: int
    channels: int
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "project.yaml"))
    parser.add_argument("--model", choices=["patch_mlp", "plain_unet"], required=True)
    parser.add_argument("--patch-dir", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--selection-metric", choices=["score", "total"], default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: RunConfig) -> nn.Module:
    if config.model == "plain_unet":
        return PlainUNet(config.channels, config.base_channels)
    return PatchMLP(config.channels, config.patch_size, config.mlp_hidden_dims)


def build_optimizer(model: nn.Module, config: RunConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def make_loaders(
    patch_dir: Path,
    feature_stats: FeatureStats,
    config: RunConfig,
) -> dict[str, DataLoader]:
    datasets = {
        "train": PatchCenterDataset(patch_dir, "train", feature_stats, config.max_train_samples),
        "val": PatchCenterDataset(patch_dir, "val", feature_stats, config.max_val_samples),
        "test": PatchCenterDataset(patch_dir, "test", feature_stats, config.max_test_samples),
    }
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        )
    return loaders


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: RunConfig,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch_index: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    mse_loss = nn.MSELoss()
    model.train(is_train)

    total_score_loss = 0.0
    total_samples = 0
    y_true_score: list[np.ndarray] = []
    y_pred_score: list[np.ndarray] = []
    y_true_grade: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        score = batch["score"].to(device, non_blocking=True)
        grade = batch["grade"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = bool(config.amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(x)
            score_center = center_slice(outputs["score_map"]).reshape(-1)
            score_loss = mse_loss(score_center, score)
            total_loss = score_loss

        if is_train:
            if scaler is not None and autocast_enabled:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_score_loss += float(score_loss.detach()) * batch_size
        total_samples += batch_size
        y_true_score.append(score.detach().cpu().numpy())
        y_pred_score.append(score_center.detach().cpu().numpy())
        y_true_grade.append((grade.detach().cpu().numpy() + 1).astype(np.int32))

    y_true_score_np = np.concatenate(y_true_score)
    y_pred_score_np = np.clip(np.concatenate(y_pred_score), 0.0, 1.0)
    y_true_grade_np = np.concatenate(y_true_grade)
    metrics = compute_all_metrics(y_true_score_np, y_pred_score_np, y_true_grade_np)
    metrics["score_loss"] = total_score_loss / total_samples
    metrics["total_loss"] = metrics["score_loss"]
    return metrics


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | int]] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        pred_score = center_slice(outputs["score_map"]).detach().cpu().numpy()
        pred_score = np.clip(pred_score, 0.0, 1.0)
        true_score = batch["score"].numpy()
        true_grade = batch["grade"].numpy() + 1
        pred_grade = score_to_grade(pred_score)
        for i in range(len(pred_score)):
            rows.append(
                {
                    "row": int(batch["row"][i]),
                    "col": int(batch["col"][i]),
                    "block_id": int(batch["block_id"][i]),
                    "true_score": float(true_score[i]),
                    "pred_score": float(pred_score[i]),
                    "true_grade": int(true_grade[i]),
                    "pred_grade": int(pred_grade[i]),
                    "residual": float(pred_score[i] - true_score[i]),
                }
            )
    return pd.DataFrame(rows)


def save_training_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_config = load_yaml(Path(args.config))
    project_root = resolve_project_path(PROJECT_ROOT, raw_config["project"]["root"])
    patch_dir = resolve_project_path(project_root, args.patch_dir or raw_config["data"]["patch_dir"])
    feature_stats_path = resolve_project_path(project_root, raw_config["data"]["feature_stats_path"])
    outputs_root = resolve_project_path(project_root, raw_config["outputs"]["metrics_root"])

    patch_tag = args.run_tag or raw_config["data"]["patch_tag"]
    run_dir = outputs_root / patch_tag / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    amp = raw_config["training"]["amp"]
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    config = RunConfig(
        model=args.model,
        patch_dir=str(patch_dir),
        output_dir=str(run_dir),
        device=args.device or raw_config["training"].get("device", "cuda"),
        batch_size=args.batch_size or raw_config["training"]["batch_size"],
        num_workers=args.num_workers if args.num_workers is not None else raw_config["training"]["num_workers"],
        epochs=args.epochs or raw_config["training"]["epochs"],
        patience=args.patience or raw_config["training"]["patience"],
        lr=args.lr or raw_config["training"]["lr"],
        weight_decay=args.weight_decay or raw_config["training"]["weight_decay"],
        optimizer=raw_config["training"]["optimizer"],
        selection_metric=args.selection_metric or raw_config["training"]["selection_metric"],
        seed=args.seed or raw_config["training"]["seed"],
        amp=amp,
        base_channels=raw_config["model"]["base_channels"],
        mlp_hidden_dims=list(raw_config["model"].get("mlp_hidden_dims", [1024, 256])),
        patch_size=raw_config["data"]["patch_size"],
        channels=raw_config["data"]["channels"],
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    requested_device = config.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    set_seed(config.seed)

    feature_stats = FeatureStats.from_json(feature_stats_path)
    loaders = make_loaders(patch_dir, feature_stats, config)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))

    (run_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_state = None
    best_metric = math.inf
    best_epoch = -1
    wait = 0
    training_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, config, device, scaler, epoch - 1)
        val_metrics = run_epoch(model, loaders["val"], None, config, device, scaler, epoch - 1)
        row = {
            "epoch": epoch,
            "train_score_loss": train_metrics["score_loss"],
            "train_total_loss": train_metrics["total_loss"],
            "train_r2": train_metrics["r2"],
            "val_score_loss": val_metrics["score_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_r2": val_metrics["r2"],
            "val_oa": val_metrics["oa"],
            "val_kappa": val_metrics["kappa"],
        }
        training_rows.append(row)
        print(
            f"epoch={epoch} "
            f"train_score={train_metrics['score_loss']:.6f} "
            f"val_score={val_metrics['score_loss']:.6f} "
            f"val_r2={val_metrics['r2']:.6f} "
            f"val_oa={val_metrics['oa']:.6f}",
            flush=True,
        )

        current_metric = val_metrics["score_loss"] if config.selection_metric == "score" else val_metrics["total_loss"]
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "model.pt")

    val_metrics = run_epoch(model, loaders["val"], None, config, device, scaler, best_epoch - 1)
    test_metrics = run_epoch(model, loaders["test"], None, config, device, scaler, best_epoch - 1)
    pred_df = predict_dataset(model, loaders["test"], device)
    conf = confusion_matrix(pred_df["true_grade"], pred_df["pred_grade"], labels=[1, 2, 3, 4, 5])

    metrics_payload = {
        "model": config.model,
        "patch_tag": patch_tag,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "selection_metric": config.selection_metric,
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_oa": val_metrics["oa"],
        "val_kappa": val_metrics["kappa"],
        "test_r2": test_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_oa": test_metrics["oa"],
        "test_kappa": test_metrics["kappa"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(conf, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5]).to_csv(
        run_dir / "confusion_matrix.csv",
        encoding="utf-8",
    )
    save_training_curve(run_dir / "training_curve.csv", training_rows)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
