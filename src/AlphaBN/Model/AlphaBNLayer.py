from __future__ import annotations
import torch
from torch import nn


class AlphaBN1d(nn.Module):
    """
    Alpha-Batch Normalization (Alpha-BN) 1D Layer.
    Reference: You, Li, Zhao, "Test-Time Batch Statistics Calibration for Covariate Shift", 2021.
    arXiv: 2110.04065.

    Mixes source statistics and target batch statistics during inference via:
      mu = alpha * mu_s + (1 - alpha) * mu_t
      sigma = alpha * sigma_s + (1 - alpha) * sigma_t
      x_hat = (x - mu) / (sigma + eps)
      y = gamma * x_hat + beta
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        default_alpha: float = 0.7,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Alpha hyperparameter in [0, 1]: weight for source statistics
        # alpha = 1.0 -> Source (CBN)
        # alpha = 0.0 -> T-BN (AdaBN)
        self.alpha = float(default_alpha)
        self.mode = "alpha_bn"  # 'source', 'tbn', 'alpha_bn'

    def set_alpha(self, alpha: float) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)

    def set_mode(self, mode: str) -> None:
        if mode not in ("source", "tbn", "alpha_bn"):
            raise ValueError(f"Unknown mode {mode!r}. Must be 'source', 'tbn', or 'alpha_bn'.")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_features)
        if self.training:
            # Standard Batch Normalization training pass on source domain
            b_mean = x.mean(dim=0)
            b_var = x.var(dim=0, unbiased=False)

            with torch.no_grad():
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
                self.running_mean = (
                    1.0 - exponential_average_factor
                ) * self.running_mean + exponential_average_factor * b_mean
                b_var_unbiased = x.var(dim=0, unbiased=True) if x.size(0) > 1 else b_var
                self.running_var = (
                    1.0 - exponential_average_factor
                ) * self.running_var + exponential_average_factor * b_var_unbiased

            mean = b_mean
            std = torch.sqrt(b_var + self.eps)

        else:
            # Inference / Test time
            mu_s = self.running_mean
            sigma_s = torch.sqrt(torch.clamp(self.running_var, min=0.0) + self.eps)

            if self.mode == "source" or self.alpha == 1.0:
                # Conventional Batch Normalization (Source statistics only)
                mean = mu_s
                std = sigma_s

            elif self.mode == "tbn" or self.alpha == 0.0:
                # Transductive Batch Normalization (Current batch statistics only)
                mu_t = x.mean(dim=0)
                var_t = x.var(dim=0, unbiased=False) if x.size(0) > 1 else torch.zeros_like(mu_t)
                mean = mu_t
                std = torch.sqrt(var_t + self.eps)

            elif self.mode == "alpha_bn":
                # Alpha-BN: Equations 3 & 4
                mu_t = x.mean(dim=0)
                var_t = x.var(dim=0, unbiased=False) if x.size(0) > 1 else torch.zeros_like(mu_t)
                sigma_t = torch.sqrt(var_t + self.eps)

                a = self.alpha
                mean = a * mu_s + (1.0 - a) * mu_t
                std = a * sigma_s + (1.0 - a) * sigma_t
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        x_hat = (x - mean) / std

        if self.affine:
            return self.weight * x_hat + self.bias
        return x_hat


def convert_bn_to_alpha_bn(module: nn.Module, default_alpha: float = 0.7) -> nn.Module:
    """
    Recursively replaces all nn.BatchNorm1d layers with AlphaBN1d layers,
    preserving learned weights, biases, and source running statistics.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            abn = AlphaBN1d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                default_alpha=default_alpha,
            )
            if child.affine:
                abn.weight.data.copy_(child.weight.data)
                abn.bias.data.copy_(child.bias.data)
            abn.running_mean.copy_(child.running_mean)
            abn.running_var.copy_(child.running_var)
            abn.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, name, abn)
        else:
            convert_bn_to_alpha_bn(child, default_alpha=default_alpha)
    return module
