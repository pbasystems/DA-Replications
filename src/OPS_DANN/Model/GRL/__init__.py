import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # ty: ignore[invalid-method-override]
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha  # updated externally by the training loop each step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.alpha)
