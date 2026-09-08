import torch
from torch import nn

from common.grl import GradientReversalLayer


class Classifier(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.grl = GradientReversalLayer()
        self.fc = nn.Linear(feature_dim, 1)  # Two classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grl(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
