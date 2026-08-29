from __future__ import annotations

from typing import NamedTuple

import numpy as np


class FeatureStats(NamedTuple):
    min: np.ndarray
    max: np.ndarray
