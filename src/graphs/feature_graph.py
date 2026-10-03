"""Modality-isolated feature graph construction for SpaGraMOR."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _as_feature_matrix(features: Any) -> np.ndarray:
    if features is None:
        raise ValueError("features cannot be None")
    if hasattr(features, "toarray"):
        features = features.toarray()
    elif hasattr(features, "to_numpy"):
        features = features.to_numpy()
    elif hasattr(features, "values") and not isinstance(features, np.ndarray):
        features = features.values
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must be a 2D matrix; received shape {features.shape}")
    if not np.issubdtype(features.dtype, np.number):
        features = features.astype(np.float32)
    return features.astype(np.float32, copy=False)


def _pearson_row_vectors(features: np.ndarray) -> np.ndarray:
    """Convert rows to zero-mean, unit-norm vectors for Pearson similarity."""

    centered = features - features.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return centered / denom


def build_feature_graph(
    features: Any,
    k: int,
    symmetrize: bool = True,
    clip_negative: bool = True,
) -> sparse.csr_matrix:
    """Build a sparse feature graph using Pearson correlation similarity.

    The graph is modality-isolated by construction: callers provide one
    modality's feature matrix only.
    """

    matrix = _as_feature_matrix(features)
    n_samples = matrix.shape[0]
    if n_samples == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_samples == 1:
        return sparse.csr_matrix((1, 1), dtype=np.float32)

    k = int(max(1, min(k, n_samples - 1)))
    normalized = _pearson_row_vectors(matrix)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(normalized)
    distances, indices = nn.kneighbors(normalized)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(n_samples), k)
    cols = indices.reshape(-1)
    similarity = 1.0 - (distances.reshape(-1) ** 2) / 2.0
    if clip_negative:
        similarity = np.maximum(similarity, 0.0)
    adjacency = sparse.coo_matrix((similarity.astype(np.float32), (rows, cols)), shape=(n_samples, n_samples))

    if symmetrize:
        adjacency = adjacency.maximum(adjacency.T)

    return adjacency.tocsr()
