import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 18, hidden_channels: int = 10, kernel_size: int = 10):
        super().__init__()
        pad_total = kernel_size - 1
        pad_left, pad_right = pad_total // 2, pad_total - pad_total // 2

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConstantPad1d((pad_left, pad_right), 0),
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=0),
            )

        self.conv1 = conv_block(input_channels, hidden_channels)
        self.conv2 = conv_block(hidden_channels, hidden_channels)
        self.conv3 = conv_block(hidden_channels, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x.flatten(start_dim=1)
