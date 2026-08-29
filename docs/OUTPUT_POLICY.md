# Output and Data Policy

The release package intentionally excludes:

```text
remote-sensing rasters
processed patch arrays
intermediate feature stacks
prediction tables
trained model weights
temporary notebooks or logs
```

Generated results should remain under `outputs/`. The `.gitignore` file excludes model files, raster files, NumPy arrays, and generated outputs by default.
