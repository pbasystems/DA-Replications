import torch
import torch.nn as nn


class RULLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))


class DomainLoss(nn.Module):
    def forward(self, d_pred, d_true):
        eps = 1e-7
        d_pred = d_pred.clamp(eps, 1 - eps)
        return torch.mean(-d_true * torch.log(d_pred) - (1 - d_true) * torch.log(1 - d_pred))
