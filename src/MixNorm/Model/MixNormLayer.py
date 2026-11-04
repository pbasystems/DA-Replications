from __future__ import annotations
import torch
from torch import nn


class MixNorm1d(nn.Module):
    """
    Mix-Norm 1D Layer.
    Reference: Hu et al., "MIXNORM: Test-Time Adaptation through Online Normalization Estimation", 2021.
    arXiv: 2110.11478.

    Estimates test-time normalization statistics by mixing global statistics (exponentially updated
    from online test stream, anchored by source training statistics) and local batch statistics:
      tau = tau_max * 10^(-3 / B)
      mu_global <- (1 - tau) * mu_global + tau * mu_batch
      sigma_global <- (1 - tau) * sigma_global + tau * var_batch
      mu_mixed = (1 - m) * mu_global + m * mu_local
      sigma_mixed = (1 - m) * sigma_global + m * sigma_local
      x_hat = (x - mu_mixed) / sqrt(sigma_mixed + eps)
      y = gamma * x_hat + beta
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        mixing_scale: float = 0.05,
        moving_speed: float = 0.001,
        tau_max: float = 0.9,
        adaptive_tau: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mixing_scale = float(mixing_scale)  # m in paper
        self.moving_speed = float(moving_speed)  # tau in paper
        self.tau_max = float(tau_max)
        self.adaptive_tau = adaptive_tau
        self.online_update = True

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Source training statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Online global statistics for test time (anchored at source stats)
        self.register_buffer("global_mean", torch.zeros(num_features))
        self.register_buffer("global_var", torch.ones(num_features))

    def reset_online_stats(self) -> None:
        """Resets online global statistics back to source running statistics."""
        self.global_mean.copy_(self.running_mean)
        self.global_var.copy_(self.running_var)

    def set_mixing_scale(self, m: float) -> None:
        if not (0.0 <= m <= 1.0):
            raise ValueError(f"mixing_scale m must be in [0, 1], got {m}")
        self.mixing_scale = float(m)

    def set_online_update(self, enabled: bool) -> None:
        self.online_update = bool(enabled)

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor | None = None) -> torch.Tensor:
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

                # Sync global stats with running stats
                self.global_mean.copy_(self.running_mean)
                self.global_var.copy_(self.running_var)

            mean = b_mean
            var = b_var

        else:
            # Test-time adaptation with MixNorm (Algorithms 1 & 2)
            b_size = x.size(0)
            b_mean = x.mean(dim=0)
            b_var = x.var(dim=0, unbiased=False) if b_size > 1 else torch.zeros_like(b_mean)

            # Local statistics: pool current batch and augmented view if provided
            if x_aug is not None:
                combined = torch.cat([x, x_aug], dim=0)
                mu_local = combined.mean(dim=0)
                sigma_local = combined.var(dim=0, unbiased=False)
            else:
                mu_local = b_mean
                sigma_local = b_var

            # Online EMA update of global statistics
            if self.online_update:
                if self.adaptive_tau and b_size > 0:
                    # Algorithm 2: tau = tau_max * 10^(-3 / B)
                    tau = float(self.tau_max * (10.0 ** (-3.0 / float(b_size))))
                    tau = max(1e-4, min(tau, 0.99))
                else:
                    tau = self.moving_speed

                with torch.no_grad():
                    self.global_mean = (1.0 - tau) * self.global_mean + tau * b_mean
                    self.global_var = (1.0 - tau) * self.global_var + tau * b_var

            mu_global = self.global_mean
            sigma_global = self.global_var

            # Mix global and local statistics
            m = self.mixing_scale
            mean = (1.0 - m) * mu_global + m * mu_local
            var = (1.0 - m) * sigma_global + m * sigma_local

        x_hat = (x - mean) / torch.sqrt(torch.clamp(var, min=0.0) + self.eps)

        if self.affine:
            return self.weight * x_hat + self.bias
        return x_hat


def convert_bn_to_mixnorm(
    module: nn.Module,
    mixing_scale: float = 0.05,
    moving_speed: float = 0.001,
) -> nn.Module:
    """
    Recursively replaces all nn.BatchNorm1d layers with MixNorm1d layers,
    preserving weights, biases, and source running statistics.
    Ensures newly created MixNorm1d layers reside on the same device as the original layer.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            device = child.running_mean.device
            mn = MixNorm1d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                mixing_scale=mixing_scale,
                moving_speed=moving_speed,
            ).to(device)
            if child.affine:
                mn.weight.data.copy_(child.weight.data)
                mn.bias.data.copy_(child.bias.data)
            mn.running_mean.copy_(child.running_mean)
            mn.running_var.copy_(child.running_var)
            mn.global_mean.copy_(child.running_mean)
            mn.global_var.copy_(child.running_var)
            mn.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, name, mn)
        else:
            convert_bn_to_mixnorm(child, mixing_scale=mixing_scale, moving_speed=moving_speed)
    return module
