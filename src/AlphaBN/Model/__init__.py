from __future__ import annotations
import torch
from torch import nn

from AlphaBN.Model.AlphaBNLayer import AlphaBN1d, convert_bn_to_alpha_bn
from AlphaBN.Model.FeatureExtractor import FeatureExtractor
from AlphaBN.Model.Regressor import Regressor


class AlphaBN(nn.Module):
    """
    Alpha-BN Model for RUL prediction.
    Equipped with AlphaBN1d layers for test-time batch statistics calibration.
    """

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
        default_alpha: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            f_size=f_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            default_alpha=default_alpha,
        )
        self.regressor = Regressor(
            feature_size=f_size,
            regressor_first_hidden_size=regressor_first_hidden_size,
            regressor_second_hidden_size=regressor_second_hidden_size,
            dropout=regressor_dropout,
            default_alpha=default_alpha,
        )

    def get_alpha_bn_layers(self) -> list[AlphaBN1d]:
        return [m for m in self.modules() if isinstance(m, AlphaBN1d)]

    def set_alpha(self, alpha: float) -> None:
        """Sets mixing parameter alpha in [0, 1] for all AlphaBN1d layers."""
        for layer in self.get_alpha_bn_layers():
            layer.set_alpha(alpha)

    def set_mode(self, mode: str) -> None:
        """Sets mode: 'source' (CBN), 'tbn' (AdaBN), or 'alpha_bn'."""
        for layer in self.get_alpha_bn_layers():
            layer.set_mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        regression_output = self.regressor(features)
        return regression_output


LSTM_AlphaBN = AlphaBN
