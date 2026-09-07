import torch
from torch import nn

from ssda.Model.Encoder.ConvBlock import ConvBlock


class Encoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        input_height: int = 14,
        input_width: int = 30,
        input_channels: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(
                input_channels, 10, kernel_size=(11, 1), padding=(5, 0)
            ),  # Input shape : (B,1,14,30) -> (B,10,14,30) | p = (k-1)/2 = 5
            ConvBlock(
                10, 10, kernel_size=(11, 1), padding=(5, 0)
            ),  # Input shape : (B,10,14,30) -> (B,10,14,30) | p = (k-1)/2 = 5
            ConvBlock(
                10, 10, kernel_size=(11, 1), padding=(5, 0)
            ),  # Input shape : (B,10,14,30) -> (B,10,14,30) | p = (k-1)/2 = 5
            ConvBlock(
                10, 10, kernel_size=(11, 1), padding=(5, 0)
            ),  # Input shape : (B,10,14,30) -> (B,10,14,30) | p = (k-1)/2 = 5
            ConvBlock(
                10, 1, kernel_size=(3, 1), padding=(1, 0)
            ),  # Input shape : (B,10,14,30) -> (B,1,14,30) | p = (k-1)/2 = 1
        )
        self.fc = nn.Linear(
            input_height * input_width, embedding_dim
        )  # Input shape : (B,1,14,30) -> (B,embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional Blocks
        x = self.conv(x)
        # Fully Connected
        x = x.view(x.size(0), -1)  # B,C,H,W -> B, 1, 14, 30 -> B, 1*14*30 = 420
        x = self.fc(x)
        x = self.activation(x)
        # No dropout after FC
        return x
