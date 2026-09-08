import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(
        self,
        feature_size,
        classifier_first_hidden_size=32,
        classifier_second_hidden_size=0,
        dropout=0.3,
    ):
        super().__init__()
        if classifier_second_hidden_size == 0:
            self.net = nn.Sequential(
                nn.Linear(feature_size, classifier_first_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_first_hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_size, classifier_first_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_first_hidden_size, classifier_second_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_second_hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
