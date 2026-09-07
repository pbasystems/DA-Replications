from torch import nn


class Regressor(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout: float = 0.1,
        expanding_size: int = 100,
        hidden_size: int = 100,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, expanding_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(expanding_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return self.net(x)
