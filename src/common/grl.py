from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

"""
Gradient Reversal Layer (GRL) implementation in PyTorch.
Adapted from Ganin et al. (2016) / Ta Duc Huy.
"""


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # ty: ignore[invalid-method-override]
        alpha = ctx.alpha
        grad_input = grad_output.neg() * alpha
        return grad_input, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.alpha)
