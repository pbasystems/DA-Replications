import torch
from torch import nn

from AlphaBN.Model.AlphaBNLayer import AlphaBN1d


class Regressor(nn.Module):
    def __init__(
        self,
        feature_size: int,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
        dropout: float = 0.3,
        default_alpha: float = 0.7,
    ):
        super().__init__()
        if regressor_second_hidden_size == 0:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                AlphaBN1d(regressor_first_hidden_size, default_alpha=default_alpha),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                AlphaBN1d(regressor_first_hidden_size, default_alpha=default_alpha),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, regressor_second_hidden_size),
                AlphaBN1d(regressor_second_hidden_size, default_alpha=default_alpha),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_second_hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
