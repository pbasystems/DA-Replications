from __future__ import annotations
import copy
import torch
from torch import nn

from TTN.Model.TTNLayer import TTN1d
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
        mode: str = "ttn",
        alpha: float | torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Algorithm 3: Test-time Inference using TTN.
        Supports:
          mode='cbn': Conventional Batch Normalization (Source running stats only)
          mode='tbn': Transductive Batch Normalization (Current batch stats only / AdaBN)
          mode='ttn': Test-Time Normalization (Interpolated stats using alpha)
        """
        self.model.eval()

        if hasattr(self.model, "set_ttn_mode"):
            self.model.set_ttn_mode(mode)
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
            prefix = f"test_{mode}"
            self.reporter.log_metrics(
                {
                    f"{prefix}/rmse": rmse,
                    f"{prefix}/mae": mae,
                    f"{prefix}/score": total_score,
                }
            )

        return metrics

    def compare_cbn_tbn_ttn(
        self,
        test_dataloader,
        constant_alphas: list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Comprehensive comparison of:
        - CBN (alpha=0, source running stats)
        - TBN (alpha=1, AdaBN test batch stats)
        - TTN (optimized alpha)
        - TTN (prior A)
        - TTN with constant alphas (e.g. 0.1, 0.2, 0.3, 0.5)
        """
        if constant_alphas is None:
            constant_alphas = [0.1, 0.2, 0.3, 0.5]

        results = {}

        # 1. CBN: Conventional Batch Normalization (No Adaptation)
        results["CBN (alpha=0)"] = self.evaluate(test_dataloader, mode="cbn")

        # 2. TBN: Transductive Batch Normalization (AdaBN)
        results["TBN (alpha=1)"] = self.evaluate(test_dataloader, mode="tbn")

        # 3. TTN (Current / Optimized alpha)
        results["TTN (Optimized alpha)"] = self.evaluate(test_dataloader, mode="ttn")

        # 4. Constant alpha sweep
        for a in constant_alphas:
            results[f"TTN (alpha={a:.1f})"] = self.evaluate(test_dataloader, mode="ttn", alpha=a)

        return results
