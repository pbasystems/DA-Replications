import torch
from torch import nn


class Score(nn.Module):
    def __init__(self, alpha_over: float = 1 / 10, alpha_under: float = 1 / 13):
        super().__init__()
        self.alpha_over = alpha_over
        self.alpha_under = alpha_under

    def forward(self, y_pred, y_true):
        residue = y_pred - y_true
        alpha = torch.where(residue >= 0, self.alpha_over, self.alpha_under)
        return torch.sum(torch.exp(alpha * torch.abs(residue)))
