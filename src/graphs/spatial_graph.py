"""Spatial graph construction for SpaGraMOR.

The builder accepts either explicit coordinate matrices or AnnData objects.
It supports the two verified source patterns in the project:

- ``obsm['spatial']``
- ``obs['array_col']`` + ``obs['array_row']``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SpatialCoordinates:
    coordinates: np.ndarray
    source: str


def _as_numpy_coordinates(coordinates: Any) -> np.ndarray:
    if coordinates is None:
        raise ValueError("coordinates cannot be None")
    if hasattr(coordinates, "to_numpy"):
        coordinates = coordinates.to_numpy()
    elif hasattr(coordinates, "values") and not isinstance(coordinates, np.ndarray):
        coordinates = coordinates.values
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"coordinates must have shape (N, 2); received {coordinates.shape}"
        )
    if not np.issubdtype(coordinates.dtype, np.number):
        coordinates = coordinates.astype(np.float32)
    return coordinates.astype(np.float32, copy=False)


def extract_spatial_coordinates(
    adata: Any,
    spatial_key: str = "spatial",
    row_key: str = "array_row",
    col_key: str = "array_col",
) -> SpatialCoordinates:
    """Extract coordinates from verified AnnData sources.

    Preference order:
    1. ``obsm[spatial_key]``
    2. ``obs[col_key]`` + ``obs[row_key]``
    """

    if hasattr(adata, "obsm") and spatial_key in adata.obsm:
        return SpatialCoordinates(_as_numpy_coordinates(adata.obsm[spatial_key]), f"obsm[{spatial_key!r}]")

    if hasattr(adata, "obs") and col_key in adata.obs and row_key in adata.obs:
        coords = np.column_stack([adata.obs[col_key].to_numpy(), adata.obs[row_key].to_numpy()])
        return SpatialCoordinates(_as_numpy_coordinates(coords), f"obs[{col_key!r}]+obs[{row_key!r}]")

    raise KeyError(
        f"Could not find coordinates in obsm[{spatial_key!r}] or obs[{col_key!r}]/obs[{row_key!r}]"
    )


def build_spatial_graph(
    coordinates: Any,
    k: int,
    include_self: bool = False,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """Build a sparse spatial KNN graph.

    Parameters
    ----------
    coordinates:
        Coordinate matrix with shape ``(N, 2)``.
    k:
        Number of nearest neighbors.
    include_self:
        Whether to add diagonal self-loops.
    symmetrize:
        Whether to symmetrize the adjacency with ``max(A, A.T)``.
    """

    coords = _as_numpy_coordinates(coordinates)
    n_samples = coords.shape[0]
    if n_samples == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_samples == 1:
        return sparse.eye(1, dtype=np.float32, format="csr") if include_self else sparse.csr_matrix((1, 1), dtype=np.float32)

    k = int(max(1, min(k, n_samples - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(n_samples), k)
    cols = indices.reshape(-1)
    # Use inverse distance so closer spots receive stronger edges.
    weights = 1.0 / (distances.reshape(-1) + 1e-8)
    adjacency = sparse.coo_matrix((weights, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)

    if symmetrize:
        adjacency = adjacency.maximum(adjacency.T)
    if include_self:
        adjacency = adjacency + sparse.eye(n_samples, dtype=np.float32, format="coo")

    return adjacency.tocsr()
