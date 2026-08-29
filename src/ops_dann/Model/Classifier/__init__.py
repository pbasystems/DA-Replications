import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int = 50, second_hidden_size: int = 30
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, second_hidden_size)
        self.fc3 = nn.Linear(second_hidden_size, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.output_activation(x)
