"""Sample-level cross-view contrastive loss with optional spatial filtering."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy import sparse


def _to_sparse_coo(adjacency: object) -> sparse.coo_matrix:
    if adjacency is None:
        raise ValueError("adjacency cannot be None")
    if sparse.issparse(adjacency):
        return adjacency.tocoo()
    if isinstance(adjacency, torch.Tensor):
        if adjacency.is_sparse:
            tensor = adjacency.coalesce().cpu()
            indices = tensor.indices().numpy()
            values = tensor.values().numpy()
            return sparse.coo_matrix((values, (indices[0], indices[1])), shape=tensor.shape)
        adjacency = adjacency.detach().cpu().numpy()
    return sparse.coo_matrix(np.asarray(adjacency))


class SpatialNegativeMask:
    """Mark spatial neighbors that should be ignored as negatives."""

    def __call__(self, adjacency: object) -> Tensor:
        coo = _to_sparse_coo(adjacency)
        mask = torch.zeros(coo.shape, dtype=torch.bool)
        if coo.nnz == 0:
            return mask
        row = torch.as_tensor(coo.row, dtype=torch.long)
        col = torch.as_tensor(coo.col, dtype=torch.long)
        off_diag = row != col
        mask[row[off_diag], col[off_diag]] = True
        return mask


class AdaptiveSampleContrastiveLoss(nn.Module):
    """Weighted pairwise InfoNCE over same-spot cross-view positives."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def pair_loss(self, left: Tensor, right: Tensor, ignore_mask: Optional[Tensor] = None) -> Tensor:
        _validate_pair(left, right)
        left = F.normalize(left, dim=1)
        right = F.normalize(right, dim=1)
        logits = left @ right.T / self.temperature
        if ignore_mask is not None:
            if ignore_mask.shape != logits.shape or ignore_mask.dtype != torch.bool:
                raise ValueError("ignore_mask must be a bool tensor with logits shape")
            logits = logits.masked_fill(ignore_mask.to(logits.device), -1e9)
        labels = torch.arange(left.shape[0], device=left.device)
        forward = F.cross_entropy(logits, labels)
        reverse = F.cross_entropy(logits.T, labels)
        return 0.5 * (forward + reverse)

    def forward(
        self,
        representations: Sequence[Tensor],
        weights: Optional[Tensor] = None,
        spatial_adjacency: Optional[object] = None,
        return_debug: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        if len(representations) < 2:
            raise ValueError("at least two views are required")
        if weights is None:
            weights = torch.ones(
                len(representations),
                dtype=representations[0].dtype,
                device=representations[0].device,
            ) / len(representations)
        if weights.ndim != 1 or weights.numel() != len(representations):
            raise ValueError("weights must have one value per view")
        if torch.any(weights < 0):
            raise ValueError("weights must be nonnegative")

        ignore_mask = None
        if spatial_adjacency is not None:
            ignore_mask = SpatialNegativeMask()(spatial_adjacency)
            if ignore_mask.shape[0] != representations[0].shape[0]:
                raise ValueError("spatial mask size must match the number of spots")

        total = representations[0].new_zeros(())
        normalizer = representations[0].new_zeros(())
        masked_positions = (
            int(ignore_mask.sum().item()) if ignore_mask is not None else 0
        )
        # Count the off-diagonal candidates that remain in each pairwise
        # denominator. The mask itself is unchanged; this is diagnostics only.
        off_diagonal = ~torch.eye(
            representations[0].shape[0],
            dtype=torch.bool,
            device=representations[0].device,
        )
        unfiltered_neg_count = int(off_diagonal.sum().item())
        if ignore_mask is None:
            valid_negative_mask = off_diagonal
        else:
            valid_negative_mask = off_diagonal & ~ignore_mask.to(
                representations[0].device
            )
        debug: Dict[str, Any] = {
            "ignore_mask": ignore_mask,
            "snf_enabled": ignore_mask is not None,
            "masked_positions": masked_positions,
            "unfiltered_neg_count": unfiltered_neg_count,
            "neg_count": int(valid_negative_mask.sum().item()),
        }
        for left_index in range(len(representations)):
            for right_index in range(left_index + 1, len(representations)):
                pair_weight = weights[left_index] + weights[right_index]
                total = total + pair_weight * self.pair_loss(
                    representations[left_index], representations[right_index], ignore_mask=ignore_mask
                )
                normalizer = normalizer + pair_weight
        loss = total / normalizer.clamp_min(torch.finfo(total.dtype).eps)
        if return_debug:
            return loss, debug
        return loss


def sample_contrastive_loss(
    representations: Sequence[Tensor],
    weights: Optional[Tensor] = None,
    temperature: float = 0.5,
    spatial_adjacency: Optional[object] = None,
) -> Tensor:
    return AdaptiveSampleContrastiveLoss(temperature)(
        representations,
        weights,
        spatial_adjacency=spatial_adjacency,
    )


def _validate_pair(left: Tensor, right: Tensor) -> None:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("contrastive representations must be 2D")
    if left.shape != right.shape:
        raise ValueError(
            f"cross-view representations must have equal shape; "
            f"received {tuple(left.shape)} and {tuple(right.shape)}"
        )
