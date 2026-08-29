import torch
from torch import nn

from lstm_dann.Model.Classifier import Classifier
from lstm_dann.Model.FeatureExtractor import FeatureExtractor
from lstm_dann.Model.GRL import GradientReversal
from lstm_dann.Model.Regressor import Regressor


class LSTM_DANN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        f_size,
        num_layers,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        classifier_dropout=0.3,
        alpha=0.8,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_size, hidden_size, f_size, num_layers, lstm_dropout
        )
        self.regressor = Regressor(f_size, dropout=regressor_dropout)
        self.classifier = Classifier(f_size, dropout=classifier_dropout)
        self.grl = GradientReversal(alpha)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.feature_extractor(x)
        regression_output = self.regressor(features)
        reversed_features = self.grl(features)
        classification_output = self.classifier(reversed_features)
        return regression_output, classification_output
