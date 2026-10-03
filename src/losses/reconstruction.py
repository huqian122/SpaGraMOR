"""Reconstruction loss for MGCMVC-original view autoencoders."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    """Compute ``sum_v ||X_v - D_v(E_v(X_v))||²``."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(
        self,
        inputs: Sequence[Tensor],
        reconstructions: Sequence[Tensor],
    ) -> Tensor:
        if len(inputs) != len(reconstructions):
            raise ValueError("inputs and reconstructions must have the same length")
        if not inputs:
            raise ValueError("at least one view is required")
        losses = []
        for index, (x, reconstruction) in enumerate(zip(inputs, reconstructions)):
            if x.shape != reconstruction.shape:
                raise ValueError(
                    f"view {index} shape mismatch: {tuple(x.shape)} vs "
                    f"{tuple(reconstruction.shape)}"
                )
            losses.append(torch.nn.functional.mse_loss(x, reconstruction, reduction=self.reduction))
        return torch.stack(losses).sum()


def reconstruction_loss(
    inputs: Sequence[Tensor],
    reconstructions: Sequence[Tensor],
    reduction: str = "mean",
) -> Tensor:
    """Functional wrapper for :class:`ReconstructionLoss`."""

    return ReconstructionLoss(reduction=reduction)(inputs, reconstructions)
