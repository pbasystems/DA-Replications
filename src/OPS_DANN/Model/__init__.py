import torch
import torch.nn as nn

from OPS_DANN.Model.Classifier import Classifier
from OPS_DANN.Model.FeatureExtractor import FeatureExtractor
from OPS_DANN.Model.GRL import GradientReversal
from OPS_DANN.Model.Regressor import Regressor


class OPSDANNHard(nn.Module):
    def __init__(self, input_channels: int = 18, num_phases: int = 3):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_channels)
        self.regressor = Regressor(input_size=50)
        self.classifiers = nn.ModuleList([Classifier(input_size=50) for _ in range(num_phases)])
        self.grl = GradientReversal(alpha=0.0)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        rul = self.regressor(features)

        reversed_features = self.grl(features)
        domain_pred = torch.zeros(x.size(0), 1, device=x.device)
        for phase_id, classifier in enumerate(self.classifiers):
            mask = phase == phase_id
            if mask.any():
                domain_pred[mask] = classifier(reversed_features[mask])

        return rul, domain_pred
