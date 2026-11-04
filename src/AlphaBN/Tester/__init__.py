from __future__ import annotations
import torch
from torch import nn

from utils.reporter import Reporter


class Tester:
    __test__ = False  # Suppress pytest collection

    def __init__(self, model: nn.Module, score_loss, device, reporter: Reporter | None = None):
        self.model = model
        self.score_loss = score_loss
        self.device = device
        self.reporter = reporter

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        alpha: float | None = None,
        mode: str = "alpha_bn",
    ) -> dict[str, float]:
        """
        Evaluates the model on dataloader using Alpha-BN calibration (Eq. 3 & 4).
        """
        self.model.eval()

        if hasattr(self.model, "set_mode"):
            self.model.set_mode(mode)
        if alpha is not None and hasattr(self.model, "set_alpha"):
            self.model.set_alpha(alpha)

        total_sq_error = 0.0
        total_abs_error = 0.0
        total_score = 0.0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.unsqueeze(1)

            outputs = self.model(inputs)

            total_sq_error += ((outputs - labels) ** 2).sum().item()
            total_abs_error += (outputs - labels).abs().sum().item()
            if self.score_loss is not None:
                total_score += self.score_loss(outputs, labels).item()
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
            tag = f"test_{mode}" if alpha is None else f"test_alpha_{alpha:.2f}"
            self.reporter.log_metrics(
                {
                    f"{tag}/rmse": rmse,
                    f"{tag}/mae": mae,
                    f"{tag}/score": total_score,
                }
            )

        return metrics

    def compare_alphas(
        self,
        test_dataloader,
        alphas: list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluates a sweep of alpha values to compare Source (alpha=1),
        T-BN / AdaBN (alpha=0), and various Alpha-BN blends.
        """
        if alphas is None:
            alphas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]

        results = {}
        for a in alphas:
            if a == 1.0:
                name = "Source (alpha=1.0 / CBN)"
            elif a == 0.0:
                name = "T-BN (alpha=0.0 / AdaBN)"
            else:
                name = f"Alpha-BN (alpha={a:.1f})"

            metrics = self.evaluate(test_dataloader, alpha=a, mode="alpha_bn")
            results[name] = metrics

        return results
