from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_json(cls, path: Path) -> "FeatureStats":
        payload = json.loads(path.read_text(encoding="utf-8"))
        mean = np.asarray(payload["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(payload["std"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.where(std == 0, 1.0, std)
        return cls(mean=mean, std=std)


class PatchCenterDataset(Dataset):
    """Loads patch tensors and supervises only the center pixel."""

    def __init__(
        self,
        patch_dir: Path,
        split: str,
        feature_stats: FeatureStats,
        max_samples: int | None = None,
    ) -> None:
        self.patch_dir = patch_dir
        self.split = split
        self.feature_stats = feature_stats
        self.features = np.load(patch_dir / f"features_{split}.npy", mmap_mode="r")
        self.meta = pd.read_csv(patch_dir / f"meta_{split}.csv")
        if max_samples is not None:
            self.features = self.features[:max_samples]
            self.meta = self.meta.iloc[:max_samples].reset_index(drop=True)
        self.center_score = self.meta["center_score"].to_numpy(dtype=np.float32)
        self.center_grade = self.meta["center_grade"].to_numpy(dtype=np.int64)
        self.rows = self.meta["row"].to_numpy(dtype=np.int32)
        self.cols = self.meta["col"].to_numpy(dtype=np.int32)
        self.block_ids = self.meta["block_id"].to_numpy(dtype=np.int32)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        x = np.asarray(self.features[index], dtype=np.float32)
        x = np.where(np.isnan(x), self.feature_stats.mean, x)
        x = (x - self.feature_stats.mean) / self.feature_stats.std
        return {
            "x": torch.from_numpy(x),
            "score": torch.tensor(self.center_score[index], dtype=torch.float32),
            "grade": torch.tensor(self.center_grade[index] - 1, dtype=torch.long),
            "row": int(self.rows[index]),
            "col": int(self.cols[index]),
            "block_id": int(self.block_ids[index]),
        }
