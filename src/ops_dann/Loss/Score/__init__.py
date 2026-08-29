from __future__ import annotations

import torch
from torch import nn

from utils.metrics import nejjar_cmapss_score


class Score(nn.Module):
    def __init__(
        self,
        alpha_over: float = 1.0 / 10.0,
        alpha_under: float = 1.0 / 13.0,
    ) -> None:
        super().__init__()
        self.alpha_over = alpha_over
        self.alpha_under = alpha_under

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return nejjar_cmapss_score(
            y_pred, y_true, alpha_over=self.alpha_over, alpha_under=self.alpha_under
        )
