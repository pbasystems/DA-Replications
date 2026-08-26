import torch
import torch.nn as nn
from torch.autograd import Function

"""
Gradient Reversal Layer (GRL) implementation in PyTorch.
This is adapted version of the original implementation by Ta Duc Huy (ted).
https://github.com/tadeephuy/GradientReversal

"""

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        grad_input = grad_output.neg() * alpha
        return grad_input, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)
