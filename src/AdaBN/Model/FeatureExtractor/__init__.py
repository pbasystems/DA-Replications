import torch
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, f_size, num_layers, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(lstm_hidden_size, f_size)
        self.bn_proj = nn.BatchNorm1d(f_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length(T_w), input_size(Feature size))
        _, (hn, _) = self.lstm(x)
        # hn[-1] shape: (batch_size, lstm_hidden_size)
        out = self.bn_lstm(hn[-1])
        out = self.dropout(self.activation(out))
        out = self.projection(out)
        out = self.bn_proj(out)
        return out
