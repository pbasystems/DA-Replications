import copy
import torch
from torch import nn

from utils.reporter import Reporter


class Tester:
    __test__ = False  # Prevent pytest from treating Tester as a test case

    def __init__(self, model: nn.Module, score_loss, device, reporter: Reporter | None = None):
        self.model = model
        self.score_loss = score_loss
        self.device = device
        self.reporter = reporter

    @torch.no_grad()
    def adapt(self, target_dataloader) -> None:
        """
        Adaptive Batch Normalization (AdaBN) test-time adaptation.
        Passes target domain data through the network to re-estimate
        mean and variance for all BatchNorm layers, with model weights frozen.
        """
        self.model.eval()

        bn_modules = [
            m
            for m in self.model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        ]

        if not bn_modules:
            return

        for bn in bn_modules:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()  # only BN layers in training mode to update running stats

        for batch in target_dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get("inputs", batch.get("target", None))
            else:
                inputs = batch

            inputs = inputs.to(self.device)
            _ = self.model(inputs)

        # Set back to eval mode
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader, adapt_dataloader=None) -> dict[str, float]:
        """
        Evaluates the model on dataloader.
        If adapt_dataloader is provided, applies AdaBN test-time domain adaptation first.
        """
        if adapt_dataloader is not None:
            self.adapt(adapt_dataloader)

        self.model.eval()

        total_sq_error = 0.0
        total_abs_error = 0.0
        total_score = 0.0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.unsqueeze(1)

            regression_outputs = self.model(inputs)

            total_sq_error += ((regression_outputs - labels) ** 2).sum().item()
            total_abs_error += (regression_outputs - labels).abs().sum().item()
            if self.score_loss is not None:
                total_score += self.score_loss(regression_outputs, labels).item()
            n_samples += labels.size(0)

        rmse = (total_sq_error / max(n_samples, 1)) ** 0.5
        mae = total_abs_error / max(n_samples, 1)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "score": total_score,
            "n_samples": n_samples,
        }

        if self.reporter:
            self.reporter.log_metrics(
                {
                    "test/rmse": rmse,
                    "test/mae": mae,
                    "test/score": total_score,
                }
            )

        return metrics

    def evaluate_with_and_without_adaptation(
        self, test_dataloader, adapt_dataloader=None
    ) -> dict[str, dict[str, float]]:
        """
        Compares target performance before AdaBN (source BN statistics)
        and after AdaBN (target BN statistics).
        """
        adapt_loader = adapt_dataloader if adapt_dataloader is not None else test_dataloader

        # Save initial state dict (including source BN running stats)
        source_state = copy.deepcopy(self.model.state_dict())

        # Evaluate before adaptation
        before_metrics = self.evaluate(test_dataloader, adapt_dataloader=None)

        # Adapt using target domain data and evaluate
        after_metrics = self.evaluate(test_dataloader, adapt_dataloader=adapt_loader)

        # Restore source state
        self.model.load_state_dict(source_state)

        return {
            "before_adaptation": before_metrics,
            "after_adaptation": after_metrics,
        }
