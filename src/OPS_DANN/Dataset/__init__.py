from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import decimate
from torch.utils.data import Dataset

W_COLS = ["alt", "Mach", "TRA", "T2"]
X_S_COLS = [
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
]
FEATURE_COLS = W_COLS + X_S_COLS

PHASE_ASCENDING = 0
PHASE_STEADY = 1
PHASE_DESCENDING = 2
NUM_PHASES = 3


class FeatureStats(NamedTuple):
    min: np.ndarray
    max: np.ndarray


class NCMAPSSDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        flight_class: int,
        split: str = "all",
        window_size: int = 50,
        stride: int = 1,
        downsample_factor: int = 10,
        phase_threshold: float = 0.5,
        median_filter_length: int = 51,
        feature_stats: FeatureStats | None = None,
    ) -> None:
        if split not in ("dev", "test", "all"):
            raise ValueError(f"split must be 'dev', 'test' or 'all', got {split!r}")

        self.h5_path = Path(h5_path)
        self.flight_class = flight_class
        self.window_size = window_size
        self.stride = stride
        self.downsample_factor = downsample_factor
        self.phase_threshold = phase_threshold
        self.median_filter_length = median_filter_length

        raw = self._select_flight_class(self._load_raw(split))

        windows, targets, phases, unit_numbers, rul_scales = [], [], [], [], []
        for unit in np.unique(raw["unit"]):
            unit_data = self._select_unit(raw, unit)
            unit_data = self._downsample_unit(unit_data)
            unit_data = self._label_phases(unit_data)
            unit_data = self._restrict_to_post_onset(unit_data)
            if unit_data is None:
                continue
            u_windows, u_targets, u_phases = self._build_unit_windows(unit_data)
            if len(u_targets) == 0:
                continue
            windows.append(u_windows)
            targets.append(u_targets)
            phases.append(u_phases)
            unit_numbers.append(np.full(len(u_targets), unit, dtype=np.int64))
            rul_scales.append(
                np.full(len(u_targets), unit_data["rul_onset"], dtype=np.float32)
            )

        self.windows = np.concatenate(windows, axis=0)
        self.targets = np.concatenate(targets, axis=0)
        self.phases = np.concatenate(phases, axis=0)
        self.unit_numbers = np.concatenate(unit_numbers, axis=0)
        self.rul_scale = np.concatenate(rul_scales, axis=0)

        self.feature_stats = feature_stats or self._fit_feature_stats()
        self._normalize_features()

    # ---- loading ---------------------------------------------------------

    def _load_raw(self, split: str) -> dict:
        with h5py.File(self.h5_path, "r") as f:

            def read(prefix):
                w = f[f"W_{prefix}"][:]
                xs = f[f"X_s_{prefix}"][:]
                a = f[f"A_{prefix}"][:]
                y = f[f"Y_{prefix}"][:, 0]
                return w, xs, a, y

            if split == "all":
                w_dev, xs_dev, a_dev, y_dev = read("dev")
                w_test, xs_test, a_test, y_test = read("test")
                w = np.concatenate([w_dev, w_test])
                xs = np.concatenate([xs_dev, xs_test])
                a = np.concatenate([a_dev, a_test])
                y = np.concatenate([y_dev, y_test])
            else:
                w, xs, a, y = read(split)

            a_var = list(np.array(f["A_var"][:], dtype="U20"))

        features = np.concatenate([w, xs], axis=1).astype(
            np.float32
        )  # FEATURE_COLS order
        idx = {name: a_var.index(name) for name in ("unit", "cycle", "Fc", "hs")}
        return {
            "features": features,
            "unit": a[:, idx["unit"]],
            "cycle": a[:, idx["cycle"]],
            "fc": a[:, idx["Fc"]],
            "hs": a[:, idx["hs"]],
            "rul": y.astype(np.float64),
        }

    def _select_flight_class(self, raw: dict) -> dict:
        mask = raw["fc"] == self.flight_class
        return {key: value[mask] for key, value in raw.items()}

    def _select_unit(self, raw: dict, unit: float) -> dict:
        mask = raw["unit"] == unit
        return {key: value[mask] for key, value in raw.items()}

    # ---- per-unit preprocessing ------------------------------------------

    def _downsample_unit(self, data: dict) -> dict:
        q = self.downsample_factor
        if len(data["features"]) < 3 * q:
            out = {key: value[::q] for key, value in data.items()}
            return out

        features = decimate(data["features"], q, n=8, ftype="iir", axis=0)
        n = len(features)
        return {
            "features": features.astype(np.float32),
            "unit": data["unit"][::q][:n],
            "cycle": data["cycle"][::q][:n],
            "hs": data["hs"][::q][:n],
            "rul": data["rul"][::q][:n],
        }

    def _label_phases(self, data: dict) -> dict:
        alt = data["features"][:, FEATURE_COLS.index("alt")]
        cycle = data["cycle"]
        dt = self.downsample_factor

        phase = np.full(len(alt), PHASE_STEADY, dtype=np.int64)
        for c in np.unique(cycle):
            idx = np.nonzero(cycle == c)[0]
            if len(idx) < 2:
                continue
            alt_c = alt[idx]
            d_alt = np.diff(alt_c) / dt

            phase_c = np.full(len(alt_c), PHASE_STEADY, dtype=np.int64)
            phase_c[:-1][d_alt >= self.phase_threshold] = PHASE_ASCENDING
            phase_c[:-1][d_alt <= -self.phase_threshold] = PHASE_DESCENDING
            phase_c[-1] = phase_c[-2]

            if len(phase_c) >= self.median_filter_length:
                phase_c = median_filter(
                    phase_c, size=self.median_filter_length, mode="nearest"
                )
            phase[idx] = phase_c

        out = dict(data)
        out["phase"] = phase
        return out

    def _restrict_to_post_onset(self, data: dict) -> dict | None:
        mask = data["hs"] == 0
        if not np.any(mask):
            return None

        out = {key: value[mask] for key, value in data.items()}
        rul_onset = out["rul"][0]
        if rul_onset <= 0:
            return None
        out["rul"] = out["rul"] / rul_onset
        out["rul_onset"] = rul_onset
        return out

    def _build_unit_windows(
        self, data: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.window_size
        features = data["features"]
        rul = data["rul"]
        phase = data["phase"]
        n = len(features)

        if n <= w:
            pad_len = w + 1 - n
            features = np.vstack(
                [np.zeros((pad_len, features.shape[1]), dtype=np.float32), features]
            )
            rul = np.concatenate([np.full(pad_len, rul[0], dtype=rul.dtype), rul])
            phase = np.concatenate(
                [np.full(pad_len, phase[0], dtype=phase.dtype), phase]
            )
            n = len(features)

        n_steps = n - w
        windows, targets, phases = [], [], []
        for t in range(w, w + n_steps, self.stride):
            windows.append(features[t - w : t])
            targets.append(rul[t])
            phases.append(phase[t])

        return (
            np.stack(windows).astype(np.float32)
            if windows
            else np.empty((0, w, features.shape[1]), dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(phases, dtype=np.int64),
        )

    # ---- normalisation -----------------------------------------------------

    def _fit_feature_stats(self) -> FeatureStats:
        flat = self.windows.reshape(-1, self.windows.shape[-1])
        return FeatureStats(min=flat.min(axis=0), max=flat.max(axis=0))

    def _normalize_features(self) -> None:
        span = self.feature_stats.max - self.feature_stats.min
        span = np.where(span == 0, 1.0, span)
        self.windows = (2 * (self.windows - self.feature_stats.min) / span - 1).astype(
            np.float32
        )

    # ---- torch Dataset API --------------------------------------------------

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.targets[idx])
        z = torch.tensor(self.phases[idx])
        scale = torch.tensor(self.rul_scale[idx])
        return x, y, z, scale
