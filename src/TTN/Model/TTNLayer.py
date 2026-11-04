from __future__ import annotations
import torch
from torch import nn


class TTN1d(nn.Module):
    """
    Test-Time Normalization (TTN) 1D Layer.
    Reference: Lim et al., "TTN: A Domain-Shift Aware Batch Normalization Parameter-Estimation
    Method for Test-Time Adaptation", ICLR 2023.

    Calculates mixed batch statistics between source running statistics and test batch statistics
    via channel-wise alpha in [0, 1]:
      mu_tilde = alpha * mu_t + (1 - alpha) * mu_s
      sigma_tilde^2 = alpha * sigma_t^2 + (1 - alpha) * sigma_s^2 + alpha * (1 - alpha) * (mu_t - mu_s)^2
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
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

        # Source running statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Channel-wise alpha parameter for test-time adaptation
        self.register_buffer("alpha", torch.zeros(num_features))
        self.mode = "ttn"  # 'cbn', 'tbn', or 'ttn'

    def set_alpha(self, alpha: float | torch.Tensor) -> None:
        if isinstance(alpha, (int, float)):
            val = float(alpha)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"alpha must be in [0, 1], got {val}")
            self.alpha.fill_(val)
        elif isinstance(alpha, torch.Tensor):
            if alpha.numel() == 1:
                self.alpha.fill_(alpha.item())
            elif alpha.numel() == self.num_features:
                self.alpha.copy_(alpha.view_as(self.alpha))
            else:
                raise ValueError(
                    f"alpha tensor must have 1 or {self.num_features} elements, got {alpha.numel()}"
                )
        else:
            raise TypeError(f"alpha must be float or Tensor, got {type(alpha)}")

    def set_ttn_mode(self, mode: str) -> None:
        if mode not in ("cbn", "tbn", "ttn"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'cbn', 'tbn', or 'ttn'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_features)
        if self.training:
            # Source training mode: standard Batch Normalization
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
            var = b_var

        else:
            # Test-time adaptation mode
            mu_s = self.running_mean
            var_s = self.running_var

            if self.mode == "cbn":
                # Conventional Batch Normalization (Source statistics only)
                mean = mu_s
                var = var_s

            elif self.mode == "tbn":
                # Transductive Batch Normalization (Current batch statistics only)
                mu_t = x.mean(dim=0)
                var_t = x.var(dim=0, unbiased=False) if x.size(0) > 1 else torch.zeros_like(mu_t)
                mean = mu_t
                var = var_t

            elif self.mode == "ttn":
                # Test-Time Normalization (Equation 2)
                mu_t = x.mean(dim=0)
                var_t = x.var(dim=0, unbiased=False) if x.size(0) > 1 else torch.zeros_like(mu_t)

                a = self.alpha
                mean = a * mu_t + (1.0 - a) * mu_s
                var = a * var_t + (1.0 - a) * var_s + a * (1.0 - a) * ((mu_t - mu_s) ** 2)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            return self.weight * x_hat + self.bias
        return x_hat


def convert_bn_to_ttn(module: nn.Module) -> nn.Module:
    """
    Recursively replaces all nn.BatchNorm1d layers with TTN1d layers,
    preserving learned weights, biases, and source running statistics.
    Ensures newly created TTN1d layers reside on the same device as the original layer.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            device = child.running_mean.device
            ttn = TTN1d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
            ).to(device)
            if child.affine:
                ttn.weight.data.copy_(child.weight.data)
                ttn.bias.data.copy_(child.bias.data)
            ttn.running_mean.copy_(child.running_mean)
            ttn.running_var.copy_(child.running_var)
            ttn.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, name, ttn)
        else:
            convert_bn_to_ttn(child)
    return module
