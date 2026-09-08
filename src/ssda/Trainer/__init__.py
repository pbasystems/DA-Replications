import lightning as L
import torch
from torch import nn

from ssda.Model.Classifier import Classifier
from ssda.Model.Encoder import Encoder
from ssda.Model.Regressor import Regressor


class SSDA_Model(L.LightningModule):
    def __init__(
        self,
        source_encoder: Encoder,
        target_encoder: Encoder,
        regressor: Regressor,
        domain_classifier: Classifier,
        rul_loss: nn.Module,
        self_supervised_loss: nn.Module,
        unsupervised_loss: nn.Module,
        alpha: float,
        beta: float,
    ):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.regressor = regressor
        self.rul_loss = rul_loss
        self.self_supervised_loss = self_supervised_loss
        self.unsupervised_loss = unsupervised_loss
        self.domain_classifier = domain_classifier

    def training_step(self, batch, batch_idx):
        (
            source_t_a,
            source_t_b,
            source_samples_a,
            source_samples_b,
            source_labels_a,
            source_labels_b,
        ) = batch["source"]
        (
            target_t_a,
            target_t_b,
            target_samples_a,
            target_samples_b,
        ) = batch["target"]
        source_features_a = self.source_encoder(source_samples_a)
        source_features_b = self.source_encoder(source_samples_b)
        target_features_a = self.target_encoder(target_samples_a)
        target_features_b = self.target_encoder(target_samples_b)

        source_label_pred_a = self.regressor(source_features_a)
        source_label_pred_b = self.regressor(source_features_b)
        target_label_pred_a = self.regressor(target_features_a)
        target_label_pred_b = self.regressor(target_features_b)

        # Absolute RUL loss
        loss_a = self.rul_loss(source_label_pred_a, source_labels_a)
        loss_b = self.rul_loss(source_label_pred_b, source_labels_b)
        absolute_loss = loss_a + loss_b

        # Self Supervised Loss - Relative RUL loss
        sourse_t_diff = source_t_a - source_t_b
        predicted_source_t_diff = source_label_pred_a - source_label_pred_b

        source_loss = self.self_supervised_loss(sourse_t_diff, predicted_source_t_diff)

        target_t_diff = target_t_a - target_t_b
        predicted_target_t_diff = target_label_pred_a - target_label_pred_b

        target_loss = self.self_supervised_loss(target_t_diff, predicted_target_t_diff)

        self_supervised_loss = source_loss + target_loss

        # Unsupervised Domain Adaptation
        source_a_class_pred = self.classifier(source_features_a)
        source_b_class_pred = self.classifier(source_features_b)
        target_a_class_pred = self.classifier(target_features_a)
        target_b_class_pred = self.classifier(target_features_b)
        source_classifier_target_labels = torch.zeros_like(source_a_class_pred)
        target_classifier_target_labels = torch.ones_like(target_a_class_pred)
        classfier_target_labels = torch.cat(
            [source_classifier_target_labels, target_classifier_target_labels], dim=0
        )
        classifier_predicted_labels = torch.cat(
            [
                source_a_class_pred,
                source_b_class_pred,
                target_a_class_pred,
                target_b_class_pred,
            ]
        )
        unsupervised_loss = self.unsupervised_loss(
            classifier_predicted_labels, classfier_target_labels
        )
        total_loss = (
            absolute_loss
            + self.alpha * self_supervised_loss
            + self.beta * unsupervised_loss
        )
        return total_loss
