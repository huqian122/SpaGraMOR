"""MGCMVC-style fine/coarse/multigranularity representations.

This module contains the representation path only. It has no spatial
coordinates, spatial consistency term, spatial negative mask, or cluster
refinement logic, so it can serve as the MGCMVC-original migration.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FineGrainedEncoder(nn.Module):
    """Shallow view-specific autoencoder path producing ``Z_v``.

    ``X_v`` has shape ``(N, input_dim)``. The encoder returns ``Z_v`` with
    shape ``(N, latent_dim)`` and the decoder reconstructs ``X_v``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(latent_dim, min(128, input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation if activation is not None else nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be 2D; received shape {tuple(z.shape)}")
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D; received shape {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point tensor")


class CoarseGrainedMLP(nn.Module):
    """View-specific MLP mapping ``Z_v`` to coarse representation ``H_v``."""

    def __init__(
        self,
        input_dim: int,
        coarse_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(coarse_dim, min(128, input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coarse_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be 2D; received shape {tuple(z.shape)}")
        return self.network(z)


class MultigranularityFusion(nn.Module):
    """Fuse ``Z_v`` and ``H_v`` into the per-view ``G_v`` representation."""

    def __init__(
        self,
        fine_dim: int,
        coarse_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(output_dim, min(256, fine_dim + coarse_dim))
        self.network = nn.Sequential(
            nn.Linear(fine_dim + coarse_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.normalize_output = normalize_output

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        if z.ndim != 2 or h.ndim != 2:
            raise ValueError("z and h must both be 2D tensors")
        if z.shape[0] != h.shape[0]:
            raise ValueError("z and h must have the same number of spots")
        g = self.network(torch.cat([z, h], dim=1))
        return F.normalize(g, dim=1) if self.normalize_output else g


class FineOnlyFusion(nn.Module):
    """The historical fine-only ablation: ``Z -> Linear -> L2``."""

    def __init__(self, fine_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(fine_dim, output_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        del h
        return F.normalize(self.linear(z), dim=1)


class CoarseOnlyFusion(nn.Module):
    """The historical coarse-only ablation: ``H -> Linear -> L2``."""

    def __init__(self, coarse_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(coarse_dim, output_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        del z
        return F.normalize(self.linear(h), dim=1)


class LinearFusion(nn.Module):
    """The historical linear fusion: ``concat(Z,H) -> Linear -> L2``."""

    def __init__(self, fine_dim: int, coarse_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(fine_dim + coarse_dim, output_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        return F.normalize(self.linear(torch.cat([z, h], dim=1)), dim=1)


class ViewMultiGranularityEncoder(nn.Module):
    """Convenience wrapper for one view's ``X -> Z/H/G`` path."""

    def __init__(
        self,
        input_dim: int,
        fine_dim: int,
        coarse_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        fusion_mode: str = "full",
    ) -> None:
        super().__init__()
        fusion_mode = str(fusion_mode).lower()
        if fusion_mode not in {"full", "fine_only", "coarse_only", "linear_fusion"}:
            raise ValueError(
                "fusion_mode must be one of full, fine_only, coarse_only, linear_fusion"
            )
        self.fine_grained = FineGrainedEncoder(input_dim, fine_dim, hidden_dim)
        self.coarse_grained = CoarseGrainedMLP(fine_dim, coarse_dim, hidden_dim)
        self.fusion_mode = fusion_mode
        if fusion_mode == "full":
            # Keep the original default hidden-dimension logic. The global
            # fusion_hidden_dim belongs to GlobalFusion, not this module.
            self.fusion = MultigranularityFusion(
                fine_dim, coarse_dim, output_dim, hidden_dim
            )
        elif fusion_mode == "fine_only":
            self.fusion = FineOnlyFusion(fine_dim, output_dim)
        elif fusion_mode == "coarse_only":
            self.fusion = CoarseOnlyFusion(coarse_dim, output_dim)
        else:
            self.fusion = LinearFusion(fine_dim, coarse_dim, output_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        z, reconstruction = self.fine_grained(x)
        h = self.coarse_grained(z)
        g = self.fusion(z, h)
        return {"z": z, "h": h, "g": g, "reconstruction": reconstruction}


def encode_views(
    views: Sequence[Tensor],
    encoders: Sequence[ViewMultiGranularityEncoder],
) -> List[Dict[str, Tensor]]:
    """Encode an ordered collection of graph views independently."""

    if len(views) != len(encoders):
        raise ValueError("views and encoders must have the same length")
    return [encoder(view) for view, encoder in zip(views, encoders)]
