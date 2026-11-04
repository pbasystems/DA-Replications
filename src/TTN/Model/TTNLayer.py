from __future__ import annotations
import torch
from torch import nn


class TTN1d(nn.Module):
    """
    Test-Time Normalization (TTN) 1D Layer.
    Reference: Lim et al., "TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation", ICLR 2023.
    arXiv: 2302.05155.

    Interpolates between source running statistics (CBN) and current batch statistics (TBN)
    using channel-wise mixing weight alpha in [0, 1]:
      tilde_mu = alpha * mu + (1 - alpha) * mu_s
      tilde_sigma^2 = alpha * sigma^2 + (1 - alpha) * sigma_s^2 + alpha * (1 - alpha) * (mu - mu_s)^2
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        initial_alpha: float = 0.5,
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

        # Learnable / adaptive interpolating weight alpha in [0, 1]
        self.alpha = nn.Parameter(torch.full((num_features,), float(initial_alpha)))
        self.register_buffer("prior_A", torch.full((num_features,), float(initial_alpha)))

        # Mode: 'cbn' (source only), 'tbn' (test batch only), 'ttn' (interpolated)
        self.mode = "cbn"

    def set_mode(self, mode: str) -> None:
        if mode not in ("cbn", "tbn", "ttn"):
            raise ValueError(f"Unknown mode {mode!r}. Must be 'cbn', 'tbn', or 'ttn'.")
        self.mode = mode

    def set_alpha(self, alpha_val: float | torch.Tensor) -> None:
        if isinstance(alpha_val, (int, float)):
            self.alpha.data.fill_(float(alpha_val))
        elif isinstance(alpha_val, torch.Tensor):
            self.alpha.data.copy_(alpha_val)

    def set_prior(self, prior: torch.Tensor) -> None:
        self.prior_A.copy_(prior)
        self.alpha.data.copy_(prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_features)
        if self.training and self.mode == "cbn":
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
                # unbiased variance for tracking
                b_var_unbiased = x.var(dim=0, unbiased=True) if x.size(0) > 1 else b_var
                self.running_var = (
                    1.0 - exponential_average_factor
                ) * self.running_var + exponential_average_factor * b_var_unbiased

            mean = b_mean
            var = b_var

        elif self.mode == "cbn":
            # Conventional Batch Normalization (using source statistics)
            mean = self.running_mean
            var = self.running_var

        elif self.mode == "tbn":
            # Transductive Batch Normalization (using current batch statistics)
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False) if x.size(0) > 1 else torch.zeros_like(mean)

        elif self.mode == "ttn":
            # Test-Time Normalization: Eq. 2 from paper
            alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
            b_mean = x.mean(dim=0)
            b_var = x.var(dim=0, unbiased=False) if x.size(0) > 1 else torch.zeros_like(b_mean)

            # tilde_mu = alpha * mu + (1 - alpha) * mu_s
            mean = alpha_clamped * b_mean + (1.0 - alpha_clamped) * self.running_mean

            # tilde_sigma^2 = alpha * sigma^2 + (1 - alpha) * sigma_s^2 + alpha * (1 - alpha) * (mu - mu_s)^2
            diff_sq = (b_mean - self.running_mean) ** 2
            var = (
                alpha_clamped * b_var
                + (1.0 - alpha_clamped) * self.running_var
                + alpha_clamped * (1.0 - alpha_clamped) * diff_sq
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            return self.weight * x_hat + self.bias
        return x_hat


def convert_bn_to_ttn(module: nn.Module) -> nn.Module:
    """
    Recursively replaces all nn.BatchNorm1d layers with TTN1d layers,
    copying weights, biases, and running stats.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            ttn = TTN1d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
            )
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
