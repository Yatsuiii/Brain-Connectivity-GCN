"""
Gradient Reversal Layer (Ganin et al. 2016, DANN).

During the forward pass the GRL acts as an identity.
During the backward pass it multiplies the incoming gradient by -alpha,
which forces the upstream encoder to *maximise* whatever loss is downstream
of the GRL — i.e. to learn features that confuse the site classifier.

Alpha is set externally (typically annealed from 0 → 1 using the Ganin
schedule: alpha = 2/(1+exp(-10*p)) - 1, where p ∈ [0,1] is training progress).
"""

from __future__ import annotations

import math

import torch
from torch.autograd import Function


class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Flip and scale gradients; None for the alpha grad (not a tensor)
        return -ctx.alpha * grad_output, None


class GradientReversal(torch.nn.Module):
    """Wraps _GRLFunction as a stateful nn.Module so alpha can be updated
    between epochs without re-building the model."""

    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha          # updated externally by the Lightning task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)

    def __repr__(self) -> str:
        return f"GradientReversal(alpha={self.alpha:.4f})"


def ganin_alpha(epoch: int, max_epochs: int) -> float:
    """Ganin et al. (2016) annealing schedule.

    Starts at 0 (GRL has no effect) and saturates towards 1.
    Using 10× steeper ramp than the original paper so alpha reaches
    ~0.9 by the midpoint of training.
    """
    p = epoch / max(max_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
