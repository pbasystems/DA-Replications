import torch
from torch import nn

from TTN.Model.TTNLayer import TTN1d


class Regressor(nn.Module):
    def __init__(
        self,
        feature_size,
        regressor_first_hidden_size=32,
        regressor_second_hidden_size: int = 0,
        dropout=0.3,
    ):
        super().__init__()
        if regressor_second_hidden_size == 0:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                TTN1d(regressor_first_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                TTN1d(regressor_first_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, regressor_second_hidden_size),
                TTN1d(regressor_second_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_second_hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
