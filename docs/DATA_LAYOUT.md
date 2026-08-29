# Data Layout

This code release does not include data. Use the following directory structure when restoring local project data.

## Raw Data

```text
data/raw/
  boundary/
    nanjing_boundary.geojson
  dem/
  sentinel2/
  landsat_lst/
  worldcover/
```

## Processed Raster Inputs

```text
data/processed/
  stacks/
    s2_B2.tif
    s2_B3.tif
    s2_B4.tif
    s2_B8.tif
  weak_labels/
    weak_score.tif
    weak_grade.tif
```

## Patch Datasets

The main RGBN dataset should be stored as:

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

Paired DEM ablation datasets should be stored as:

```text
data/processed/patches/rgbn_strict_common/
data/processed/patches/rgbn_dem_strict/
```

Each patch array should have shape `(n_samples, n_channels, patch_size, patch_size)`. Metadata CSV files must contain:

```text
row,col,block_id,center_score,center_grade
```

`center_score` is the continuous weak EEQ target in `[0, 1]`. `center_grade` is the five-grade EEQ label encoded as integers from 1 to 5.

## External Validation Data

Optional external validation tables can be placed under:

```text
data/external/
```
