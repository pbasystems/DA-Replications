import torch
from torch import nn
from tqdm import tqdm

from AdaBN.EarlyStopper import EarlyStopping
from utils.reporter import Reporter


def _clip_optimizer_grads(optimizer, max_norm=1.0):
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, max_norm)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        source_train_dataloader,
        source_val_dataloader,
        optimizer=None,
        scheduler=None,
        regression_optimizer=None,
        domain_optimizer=None,
        regression_scheduler=None,
        domain_scheduler=None,
        regression_loss=None,
        classification_loss=None,
        score_loss=None,
        device=None,
        max_grad_norm=1.0,
        early_stopper: EarlyStopping | None = None,
        reporter: Reporter | None = None,
        **kwargs,
    ):
        self.model = model
        self.source_train_dataloader = source_train_dataloader
        self.source_val_dataloader = source_val_dataloader
        self.optimizer = optimizer or regression_optimizer
        self.scheduler = scheduler or regression_scheduler
        self.regression_loss = regression_loss or nn.L1Loss()
        self.score_loss = score_loss
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = max_grad_norm
        self.reporter = reporter
        self.mse_loss = nn.MSELoss()
        self.early_stopper = early_stopper

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_regression_loss = 0.0

            regression_bar = tqdm(
                self.source_train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} [AdaBN Source Train]",
                leave=False,
            )
            for batch in regression_bar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                self.optimizer.zero_grad()
                regression_outputs = self.model(inputs)
                loss_val = self.regression_loss(regression_outputs, labels)
                loss_val.backward()
                _clip_optimizer_grads(self.optimizer, self.max_grad_norm)
                self.optimizer.step()

                total_regression_loss += loss_val.item()
                regression_bar.set_postfix(loss=f"{loss_val.item():.4f}")

            total_regression_loss /= len(self.source_train_dataloader)
            train_score = (
                self.score_loss(regression_outputs, labels).item()
                if self.score_loss is not None
                else 0.0
            )

            validation_loss, val_score = self.validate()

            if self.reporter is not None:
                self.reporter.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train/regression_loss": total_regression_loss,
                        "train/score": train_score,
                        "val/regression_loss": validation_loss,
                        "val/score": val_score,
                    }
                )

            if self.early_stopper is not None and self.early_stopper.step(
                validation_loss, self.model
            ):
                if self.reporter is not None:
                    self.reporter.info(f"Early stopping at epoch {epoch + 1}")
                self.early_stopper.restore_best_weights(self.model)
                break

            if self.scheduler is not None:
                self.scheduler.step()

        if self.early_stopper is not None:
            self.early_stopper.restore_best_weights(self.model)

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_sq_error = 0.0
        total_score = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.source_val_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                regression_outputs = self.model(inputs)
                total_sq_error += ((regression_outputs - labels) ** 2).sum().item()
                if self.score_loss is not None:
                    total_score += self.score_loss(regression_outputs, labels).item()
                n_samples += labels.size(0)

        rmse = (total_sq_error / max(n_samples, 1)) ** 0.5
        return rmse, total_score
