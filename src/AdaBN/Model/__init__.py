import torch
from torch import nn

from AdaBN.Model.FeatureExtractor import FeatureExtractor
from AdaBN.Model.Regressor import Regressor


class AdaBN(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        regression_output = self.regressor(features)
        return regression_output


LSTM_AdaBN = AdaBN
