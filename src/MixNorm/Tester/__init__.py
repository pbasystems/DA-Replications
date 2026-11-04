from __future__ import annotations
import copy
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
        mixing_scale: float | None = None,
        online_update: bool = True,
        reset_stats_before: bool = True,
    ) -> dict[str, float]:
        """
        Evaluates the model on dataloader using MixNorm test-time adaptation (Algorithms 1 & 2).
        """
        self.model.eval()

        if reset_stats_before and hasattr(self.model, "reset_online_stats"):
            self.model.reset_online_stats()

        if hasattr(self.model, "set_online_update"):
            self.model.set_online_update(online_update)

        if mixing_scale is not None and hasattr(self.model, "set_mixing_scale"):
            self.model.set_mixing_scale(mixing_scale)

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
            tag = "test_mixnorm"
            self.reporter.log_metrics(
                {
                    f"{tag}/rmse": rmse,
                    f"{tag}/mae": mae,
                    f"{tag}/score": total_score,
                }
            )

        return metrics

    def compare_settings(
        self,
        test_dataloader,
        mixing_scales: list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compares:
        - Source Only (No online update, m=0)
        - Pure Local (m=1.0, T-BN)
        - MixNorm with default and varying mixing scales m
        """
        if mixing_scales is None:
            mixing_scales = [0.01, 0.05, 0.10, 0.20, 0.50]

        results = {}

        # 1. Source Only (no online update, static source stats)
        results["Source Only (Static Source Stats)"] = self.evaluate(
            test_dataloader, mixing_scale=0.0, online_update=False, reset_stats_before=True
        )

        # 2. Pure Local / T-BN (m=1.0)
        results["Pure Local (m=1.0 / T-BN)"] = self.evaluate(
            test_dataloader, mixing_scale=1.0, online_update=False, reset_stats_before=True
        )

        # 3. MixNorm with varying mixing scale m
        for m in mixing_scales:
            results[f"MixNorm (m={m:.2f}, Online EMA)"] = self.evaluate(
                test_dataloader, mixing_scale=m, online_update=True, reset_stats_before=True
            )

        return results
