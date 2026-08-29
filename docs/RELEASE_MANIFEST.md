# Release Manifest

## Included

```text
README.md
requirements.txt
pyproject.toml
configs/project.yaml
docs/
src/nanjing_eeq/
scripts/data/
scripts/train/
scripts/evaluate/
data/*/.gitkeep
outputs/checkpoints/.gitkeep
outputs/maps/.gitkeep
outputs/metrics/.gitkeep
outputs/tables/.gitkeep
```

## Excluded

```text
raw satellite and GIS data
processed raster stacks
NumPy patch arrays
trained neural-network weights
serialized tree models
prediction tables
temporary logs and cache files
```

The package is intended for code review and reproducibility. To rerun the study, restore the local data according to `docs/DATA_LAYOUT.md`.
