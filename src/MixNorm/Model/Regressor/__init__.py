import torch
from torch import nn

from MixNorm.Model.MixNormLayer import MixNorm1d


class Regressor(nn.Module):
    def __init__(
        self,
        feature_size: int,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
        dropout: float = 0.3,
        mixing_scale: float = 0.05,
        moving_speed: float = 0.001,
    ):
        super().__init__()
        if regressor_second_hidden_size == 0:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                MixNorm1d(
                    regressor_first_hidden_size,
                    mixing_scale=mixing_scale,
                    moving_speed=moving_speed,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                MixNorm1d(
                    regressor_first_hidden_size,
                    mixing_scale=mixing_scale,
                    moving_speed=moving_speed,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, regressor_second_hidden_size),
                MixNorm1d(
                    regressor_second_hidden_size,
                    mixing_scale=mixing_scale,
                    moving_speed=moving_speed,
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_second_hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
