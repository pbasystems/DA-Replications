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
        lstm_hidden_size,
        f_size,
        num_layers,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        classifier_dropout=0.3,
        alpha=0.8,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
        classifier_first_hidden_size: int = 32,
        classifier_second_hidden_size: int = 0,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_size, lstm_hidden_size, f_size, num_layers, lstm_dropout
        )
        self.regressor = Regressor(
            f_size,
            regressor_first_hidden_size,
            regressor_second_hidden_size,
            dropout=regressor_dropout,
        )
        self.classifier = Classifier(
            f_size,
            classifier_first_hidden_size=classifier_first_hidden_size,
            classifier_second_hidden_size=classifier_second_hidden_size,
            dropout=classifier_dropout,
        )
        self.grl = GradientReversal(alpha)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.feature_extractor(x)
        regression_output = self.regressor(features)
        reversed_features = self.grl(features)
        classification_output = self.classifier(reversed_features)
        return regression_output, classification_output
