import torch
from torch import nn


class RegressionLoss(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true) ** self.p)


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        y_pred = self.sigmoid(y_pred)
        return torch.mean(
            -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
        )
