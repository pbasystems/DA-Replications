import torch
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, f_size, num_layers, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(hidden_size, f_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length(T_w), input_size(Feature size))
        _, (hn, _) = self.lstm(x)
        # Return the last hidden state as the feature representation
        return self.projection(
            self.dropout(self.activation(hn[-1]))
        )  # shape: (batch_size, f_size)
