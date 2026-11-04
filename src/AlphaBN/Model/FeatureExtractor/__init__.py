import torch
from torch import nn

from AlphaBN.Model.AlphaBNLayer import AlphaBN1d


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        f_size: int,
        num_layers: int,
        dropout: float = 0.5,
        default_alpha: float = 0.7,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.bn_lstm = AlphaBN1d(lstm_hidden_size, default_alpha=default_alpha)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(lstm_hidden_size, f_size)
        self.bn_proj = AlphaBN1d(f_size, default_alpha=default_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length(T_w), input_size(Feature size))
        _, (hn, _) = self.lstm(x)
        # hn[-1] shape: (batch_size, lstm_hidden_size)
        out = self.bn_lstm(hn[-1])
        out = self.dropout(self.activation(out))
        out = self.projection(out)
        out = self.bn_proj(out)
        return out
