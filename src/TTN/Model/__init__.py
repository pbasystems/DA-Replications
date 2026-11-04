from __future__ import annotations
import torch
from torch import nn

from TTN.Model.FeatureExtractor import FeatureExtractor
from TTN.Model.Regressor import Regressor
from TTN.Model.TTNLayer import TTN1d, convert_bn_to_ttn


class TTN(nn.Module):
    def __init__(
        self,
        input_size: int = 24,
        lstm_hidden_size: int = 128,
        f_size: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.5,
        regressor_dropout: float = 0.3,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            f_size=f_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        self.regressor = Regressor(
            feature_size=f_size,
            regressor_first_hidden_size=regressor_first_hidden_size,
            regressor_second_hidden_size=regressor_second_hidden_size,
            dropout=regressor_dropout,
        )

    def get_ttn_layers(self) -> list[TTN1d]:
        return [m for m in self.modules() if isinstance(m, TTN1d)]

    def set_ttn_mode(self, mode: str) -> None:
        for layer in self.get_ttn_layers():
            layer.set_mode(mode)

    def set_alpha(self, alpha: float | torch.Tensor) -> None:
        for layer in self.get_ttn_layers():
            layer.set_alpha(alpha)

    def get_alpha_parameters(self) -> list[nn.Parameter]:
        return [layer.alpha for layer in self.get_ttn_layers()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        regression_output = self.regressor(features)
        return regression_output


LSTM_TTN = TTN
