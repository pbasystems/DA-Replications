import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(11, 1), stride=1, padding=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
