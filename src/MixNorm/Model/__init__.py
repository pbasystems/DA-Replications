from __future__ import annotations
import torch
from torch import nn

from MixNorm.Model.FeatureExtractor import FeatureExtractor
from MixNorm.Model.MixNormLayer import MixNorm1d, convert_bn_to_mixnorm
from MixNorm.Model.Regressor import Regressor


class MixNorm(nn.Module):
    """
    Mix-Norm Model for RUL prediction.
    Equipped with MixNorm1d layers for test-time adaptation through online normalization estimation.
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
        mixing_scale: float = 0.05,
        moving_speed: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            f_size=f_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            mixing_scale=mixing_scale,
            moving_speed=moving_speed,
        )
        self.regressor = Regressor(
            feature_size=f_size,
            regressor_first_hidden_size=regressor_first_hidden_size,
            regressor_second_hidden_size=regressor_second_hidden_size,
            dropout=regressor_dropout,
            mixing_scale=mixing_scale,
            moving_speed=moving_speed,
        )

    def get_mixnorm_layers(self) -> list[MixNorm1d]:
        return [m for m in self.modules() if isinstance(m, MixNorm1d)]

    def set_mixing_scale(self, m: float) -> None:
        for layer in self.get_mixnorm_layers():
            layer.set_mixing_scale(m)

    def set_online_update(self, enabled: bool) -> None:
        for layer in self.get_mixnorm_layers():
            layer.set_online_update(enabled)

    def reset_online_stats(self) -> None:
        for layer in self.get_mixnorm_layers():
            layer.reset_online_stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        regression_output = self.regressor(features)
        return regression_output


LSTM_MixNorm = MixNorm
