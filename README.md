# Nanjing EEQ Spatial-Context Modeling

This repository contains the reproducible code workflow for estimating ecological environment quality (EEQ) in Nanjing under a reduced RGBN input setting. The package includes data-layout checks, patch-based deep learning training, tabular baseline training, DEM ablation, and metric summarization.

No remote-sensing rasters, patch arrays, prediction tables, or trained model weights are included in this code release.

## Project Layout

```text
nanjing_eeq_code/
  configs/                 Runtime configuration
  data/                    Empty data placeholders, excluded from version control
  docs/                    Data layout and workflow notes
  outputs/                 Empty output placeholders, excluded from version control
  scripts/
    data/                  Data inventory and layout checks
    train/                 Patch-based and tabular model training
    evaluate/              Ablation and metric summary scripts
  src/nanjing_eeq/         Reusable dataset, model, and metric modules
```

## Expected Data

Place local data under `data/` using the structure described in `docs/DATA_LAYOUT.md`. The main training scripts expect patch datasets with:

```text
data/processed/patches/rgbn_strict/
  features_train.npy
  features_val.npy
  features_test.npy
  meta_train.csv
  meta_val.csv
  meta_test.csv
  feature_stats.json
```

Each `meta_*.csv` file must contain at least `row`, `col`, `block_id`, `center_score`, and `center_grade`.

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches the local CUDA runtime before installing the remaining dependencies.

## Quick Checks

```bash
python scripts/data/audit_data_layout.py --config configs/project.yaml
```

## Main Training

```bash
python scripts/train/train_patch_models.py --model plain_unet --config configs/project.yaml
python scripts/train/train_patch_models.py --model patch_mlp --config configs/project.yaml
```

Trained weights and predictions are written to `outputs/metrics/`. They are intentionally ignored by version control.

## Tabular Baselines

```bash
python scripts/train/train_tabular_baselines.py --config configs/project.yaml --models rf,xgboost
```

This trains center-pixel RF and XGBoost baselines on the same patch metadata and saves per-model metrics under `outputs/metrics/tabular_baselines/`.

## DEM Ablation

```bash
python scripts/evaluate/run_dem_ablation_xgb.py --project-root .
```

The DEM ablation expects paired datasets:

```text
data/processed/patches/rgbn_strict_common/
data/processed/patches/rgbn_dem_strict/
```

## Metric Summary

```bash
python scripts/evaluate/summarize_model_comparison.py --metrics-root outputs/metrics
```
