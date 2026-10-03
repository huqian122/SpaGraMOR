"""MGCMVC global fusion and WD-based adaptive view weighting."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn


def wasserstein_distance_1d(x: Tensor, y: Tensor) -> Tensor:
    """Compute the empirical 1-D Wasserstein-1 distance.

    MGCMVC's reference implementation applies SciPy's 1-D Wasserstein
    distance to flattened view/global representations. This torch equivalent
    preserves that empirical definition for equal-sized tensors and keeps the
    operation differentiable with respect to sorted values.
    """

    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.numel() != y.numel():
        raise ValueError(
            "This empirical implementation requires equal flattened sizes; "
            f"received {x.numel()} and {y.numel()}"
        )
    return torch.mean(torch.abs(torch.sort(x).values - torch.sort(y).values))


class GlobalFusion(nn.Module):
    """Fuse view-level ``G_v`` tensors into global representation ``U``."""

    def __init__(
        self,
        num_views: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(num_views * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.normalize_output = normalize_output

    def forward(self, representations: Sequence[Tensor]) -> Tensor:
        _validate_views(representations, self.num_views, self.input_dim)
        fused = self.network(torch.cat(list(representations), dim=1))
        if self.normalize_output:
            fused = torch.nn.functional.normalize(fused, dim=1)
        return fused


class AdaptiveWeightModule(nn.Module):
    """Compute WD-based adaptive weights."""

    def __init__(
        self,
        num_views: int,
        representation_dim: int,
        global_dim: int,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.global_fusion = GlobalFusion(
            num_views=num_views,
            input_dim=representation_dim,
            output_dim=global_dim,
            hidden_dim=fusion_hidden_dim,
        )

    def forward(
        self,
        representations: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(global_u, wd_distances, softmax(-wd_distances))``."""

        global_u = self.global_fusion(representations)
        wd_distances = torch.stack(
            [wasserstein_distance_1d(view, global_u) for view in representations]
        )
        weights = torch.softmax(-wd_distances, dim=0)
        return global_u, wd_distances, weights


def _validate_views(
    representations: Sequence[Tensor],
    num_views: int,
    representation_dim: int,
) -> None:
    if len(representations) != num_views:
        raise ValueError(f"expected {num_views} views; received {len(representations)}")
    if not representations:
        raise ValueError("at least one view is required")
    n_spots = representations[0].shape[0]
    for index, representation in enumerate(representations):
        if representation.ndim != 2:
            raise ValueError(f"view {index} must be 2D")
        if representation.shape[0] != n_spots:
            raise ValueError("all views must have the same number of spots")
        if representation.shape[1] != representation_dim:
            raise ValueError(
                f"view {index} has dim {representation.shape[1]}, "
                f"expected {representation_dim}"
            )
