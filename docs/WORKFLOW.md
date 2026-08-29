# Reproducible Workflow

## 1. Check Data Layout

```bash
python scripts/data/audit_data_layout.py --config configs/project.yaml
```

The script checks required patch arrays, metadata tables, feature statistics, and optional raster inputs.

## 2. Train Main Deep Models

```bash
python scripts/train/train_patch_models.py --model patch_mlp --config configs/project.yaml
python scripts/train/train_patch_models.py --model plain_unet --config configs/project.yaml
```

`patch_mlp` provides a non-spatial neural baseline. `plain_unet` provides the spatial-context baseline used in the manuscript.

## 3. Train Tabular Baselines

```bash
python scripts/train/train_tabular_baselines.py --config configs/project.yaml --models rf,xgboost
```

Tabular baselines use the center pixel of each patch to keep the comparison aligned with the same spatial split and target labels.

## 4. Run DEM Ablation

```bash
python scripts/evaluate/run_dem_ablation_xgb.py --project-root .
```

This compares paired XGBoost models trained on RGBN and RGBN+DEM patch datasets.

## 5. Summarize Metrics

```bash
python scripts/evaluate/summarize_model_comparison.py --metrics-root outputs/metrics
```

The summary table is written to `outputs/tables/model_comparison_summary.csv`.
