from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.types import FeatureStats

OP_SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLS = OP_SETTING_COLS + SENSOR_COLS
COLUMN_NAMES = ["unit_number", "time_cycles", *FEATURE_COLS]


class CMAPSSDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        fd: str,
        split: str = "train",
        window_size: int = 30,
        r_early: int = 125,
        feature_stats: FeatureStats | None = None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.root_dir = Path(root_dir)
        self.fd = fd
        self.split = split
        self.window_size = window_size
        self.r_early = r_early
        file_path = self.root_dir / f"{split}_{fd}.txt"
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
        df["RUL"] = self._compute_rul(df)

        self.feature_stats = feature_stats or self._fit_feature_stats(df)
        df[FEATURE_COLS] = self._normalize(df[FEATURE_COLS].to_numpy())

        self.windows, self.targets, self.unit_numbers = self._build_windows(df)

    def _compute_rul(self, df: pd.DataFrame) -> pd.Series:
        max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
        cycles_from_end = max_cycle - df["time_cycles"]

        if self.split == "train":
            return np.minimum(cycles_from_end, self.r_early)

        final_rul = pd.read_csv(
            self.root_dir / f"RUL_{self.fd}.txt", header=None, names=["RUL"]
        )
        final_rul.index = final_rul.index + 1  # unit numbers are 1-indexed
        end_rul = df["unit_number"].map(final_rul["RUL"])
        return np.minimum(cycles_from_end + end_rul, self.r_early)

    def _fit_feature_stats(self, df: pd.DataFrame) -> FeatureStats:
        values = df[FEATURE_COLS].to_numpy()
        return FeatureStats(min=values.min(axis=0), max=values.max(axis=0))

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        span = self.feature_stats.max - self.feature_stats.min
        span = np.where(span == 0, 1.0, span)
        return (values - self.feature_stats.min) / span

    def _build_windows(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.window_size
        windows, targets, unit_numbers = [], [], []

        for unit_number, group in df.groupby("unit_number", sort=True):
            group = group.sort_values("time_cycles")
            features = group[FEATURE_COLS].to_numpy(dtype=np.float32)
            rul = group["RUL"].to_numpy(dtype=np.float32)

            if len(features) <= w:
                pad_len = w + 1 - len(features)
                features = np.vstack(
                    [np.zeros((pad_len, features.shape[1]), dtype=np.float32), features]
                )
                rul = np.concatenate([np.full(pad_len, rul[0], dtype=np.float32), rul])

            n_steps = len(features) - w
            for t in range(w, w + n_steps):
                windows.append(features[t - w : t])
                targets.append(rul[t])
                unit_numbers.append(unit_number)

        return (
            np.stack(windows),
            np.array(targets, dtype=np.float32),
            np.array(unit_numbers, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[index])
        y = torch.tensor(self.targets[index])
        return x, y
