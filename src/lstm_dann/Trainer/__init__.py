import torch
from torch import nn
from tqdm import tqdm

from lstm_dann.EarlyStopper import EarlyStopping
from utils.reporter import Reporter


def _clip_optimizer_grads(optimizer, max_norm=1.0):
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, max_norm)


class Trainer:
    def __init__(
        self,
        model,
        source_train_dataloader,
        source_val_dataloader,
        target_train_dataloader,
        target_val_dataloader,
        regression_optimizer,
        domain_optimizer,
        regression_scheduler,
        domain_scheduler,
        regression_loss,
        classification_loss,
        score_loss,
        device,
        max_grad_norm=1.0,
        early_stopper: EarlyStopping | None = None,
        reporter: Reporter | None = None,
    ):
        self.model = model
        self.source_train_dataloader = source_train_dataloader
        self.source_val_dataloader = source_val_dataloader
        self.target_train_dataloader = target_train_dataloader
        self.target_val_dataloader = target_val_dataloader
        self.regression_optimizer = regression_optimizer
        self.domain_optimizer = domain_optimizer
        self.regression_scheduler = regression_scheduler
        self.domain_scheduler = domain_scheduler
        self.regression_loss = regression_loss
        self.classification_loss = classification_loss
        self.score_loss = score_loss
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.reporter = reporter
        self.mse_loss = nn.MSELoss()
        self.early_stopper = early_stopper

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_regression_loss = 0.0
            total_classification_loss = 0.0

            regression_bar = tqdm(
                self.source_train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} [regression]",
                leave=False,
            )
            for batch in regression_bar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                self.regression_optimizer.zero_grad()
                regression_outputs, _ = self.model(inputs)
                regression_loss_value = self.regression_loss(regression_outputs, labels)
                regression_loss_value.backward()
                _clip_optimizer_grads(self.regression_optimizer, self.max_grad_norm)
                self.regression_optimizer.step()

                total_regression_loss += regression_loss_value.item()
                regression_bar.set_postfix(loss=f"{regression_loss_value.item():.4f}")

            total_regression_loss /= len(self.source_train_dataloader)

            n_batches = 0
            classification_bar = tqdm(
                zip(self.source_train_dataloader, self.target_train_dataloader),
                total=min(
                    len(self.source_train_dataloader), len(self.target_train_dataloader)
                ),
                desc=f"Epoch {epoch + 1}/{epochs} [domain]",
                leave=False,
            )
            for source_batch, target_batch in classification_bar:
                source_inputs, _ = source_batch
                target_inputs, _ = target_batch
                source_inputs, target_inputs = (
                    source_inputs.to(self.device),
                    target_inputs.to(self.device),
                )

                self.domain_optimizer.zero_grad()
                _, source_classification_outputs = self.model(source_inputs)
                _, target_classification_outputs = self.model(target_inputs)

                source_labels = torch.zeros(
                    source_classification_outputs.size(0), 1
                ).to(self.device)
                target_labels = torch.ones(target_classification_outputs.size(0), 1).to(
                    self.device
                )

                source_classification_loss = self.classification_loss(
                    source_classification_outputs, source_labels
                )
                target_classification_loss = self.classification_loss(
                    target_classification_outputs, target_labels
                )
                classification_loss_value = (
                    source_classification_loss + target_classification_loss
                )

                classification_loss_value.backward()
                _clip_optimizer_grads(self.domain_optimizer, self.max_grad_norm)
                self.domain_optimizer.step()

                total_classification_loss += classification_loss_value.item()
                n_batches += 1
                classification_bar.set_postfix(
                    loss=f"{classification_loss_value.item():.4f}"
                )

            total_classification_loss /= n_batches

            if self.regression_scheduler is not None:
                self.regression_scheduler.step()
            if self.domain_scheduler is not None:
                self.domain_scheduler.step()
            if self.reporter is not None:
                self.reporter.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "regression_loss": total_regression_loss,
                        "classification_loss": total_classification_loss,
                    }
                )

            validation_loss = self.validate()
            if self.early_stopper is not None and self.early_stopper.step(
                validation_loss, self.model
            ):
                if self.reporter is not None:
                    self.reporter.info(f"Early stopping at epoch {epoch + 1}")
                self.early_stopper.restore_best_weights(self.model)
                break
        # Restore best weights if the training loop completes without early stopping
        if self.early_stopper is not None:
            self.early_stopper.restore_best_weights(self.model)

    def validate(self) -> None:
        self.model.eval()
        total_regression_loss = 0.0
        regression_bar = tqdm(self.source_val_dataloader, leave=False)
        with torch.no_grad():
            for batch in regression_bar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                regression_outputs, _ = self.model(inputs)
                regression_loss_value = torch.sqrt(
                    self.mse_loss(regression_outputs, labels)
                )

                total_regression_loss += regression_loss_value.item()
                regression_bar.set_postfix(loss=f"{regression_loss_value.item():.4f}")

            total_regression_loss /= len(self.source_val_dataloader)
            if self.reporter is not None:
                self.reporter.log_metrics({"val_rmse": total_regression_loss})
            return total_regression_loss
