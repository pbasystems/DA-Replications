import torch
from torch import nn


class Score(nn.Module):
    def __init__(self, a_1: float, a_2: float):
        super().__init__()
        self.a1 = a_1
        self.a2 = a_2

    def _calculate_score(self, residue):
        """
        Calculate the score based on the residue.
        Equation : e^(-residue/a1) - 1 if residue < 0 else e^(residue/a2) - 1
        """
        value = torch.where(
            residue < 0,
            torch.exp(-residue / self.a1),
            torch.exp(residue / self.a2),
        )
        return value - 1

    def forward(self, y_pred, y_true):
        residue = y_pred - y_true
        score = self._calculate_score(residue)
        return torch.sum(score)
