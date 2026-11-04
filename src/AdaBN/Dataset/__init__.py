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
        max_rul: int | None = None,
        feature_stats: FeatureStats | None = None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.root_dir = Path(root_dir)
        self.fd = fd
        self.split = split
        self.window_size = window_size
        self.r_early = max_rul if max_rul is not None else r_early
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

        rul_path = self.root_dir / f"RUL_{self.fd}.txt"
        rul_series = pd.read_csv(rul_path, header=None).iloc[:, 0]
        actual_rul_map = dict(zip(range(1, len(rul_series) + 1), rul_series))
        return df["unit_number"].map(actual_rul_map) + cycles_from_end

    def _fit_feature_stats(self, df: pd.DataFrame) -> FeatureStats:
        features = df[FEATURE_COLS].to_numpy()
        return FeatureStats(min=features.min(axis=0), max=features.max(axis=0))

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        span = self.feature_stats.max - self.feature_stats.min
        span[span == 0] = 1.0
        return (features - self.feature_stats.min) / span

    def _build_windows(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        windows: list[torch.Tensor] = []
        targets: list[float] = []
        unit_numbers: list[int] = []

        for unit_number, group in df.groupby("unit_number"):
            features = group[FEATURE_COLS].to_numpy()
            ruls = group["RUL"].to_numpy()

            if len(features) < self.window_size:
                continue

            for start in range(len(features) - self.window_size + 1):
                end = start + self.window_size
                windows.append(
                    torch.from_numpy(features[start:end]).float()
                )
                targets.append(float(ruls[end - 1]))
                unit_numbers.append(int(unit_number))

        return torch.stack(windows), torch.tensor(targets, dtype=torch.float32), unit_numbers

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx]
