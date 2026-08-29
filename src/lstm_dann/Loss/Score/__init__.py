from __future__ import annotations

import torch
from torch import nn

from utils.metrics import saxena_cmapss_score


class Score(nn.Module):
    def __init__(self, a_1: float = 13.0, a_2: float = 10.0) -> None:
        super().__init__()
        self.a1 = a_1
        self.a2 = a_2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return saxena_cmapss_score(y_pred, y_true, a1=self.a1, a2=self.a2)
