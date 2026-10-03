"""Boundary-aware spatial residual refinement for clustering embeddings."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _embedding_array(embedding: Any) -> Tuple[np.ndarray, np.dtype]:
    source = np.asarray(embedding)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("embedding must be a non-empty N x D matrix")
    if not np.issubdtype(source.dtype, np.number):
        raise TypeError("embedding must be numeric")
    matrix = np.asarray(source, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("embedding contains NaN or Inf")
    output_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float64)
    return matrix, output_dtype


def _coordinate_array(coordinates: Any, n_spots: int) -> np.ndarray:
    matrix = np.asarray(coordinates, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("coordinates must be an N x 2 matrix")
    if matrix.shape[0] != n_spots:
        raise ValueError(
            "embedding and coordinates must contain the same number of spots"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("coordinates contain NaN or Inf")
    return matrix


def _symmetrized_knn_edges(coordinates: np.ndarray, spatial_k: int) -> Tuple[np.ndarray, np.ndarray]:
    n_spots = coordinates.shape[0]
    if isinstance(spatial_k, bool) or int(spatial_k) != spatial_k:
        raise ValueError("spatial_k must be an integer")
    spatial_k = int(spatial_k)
    if spatial_k < 1 or spatial_k >= n_spots:
        raise ValueError("spatial_k must be between 1 and N - 1")

    indices = NearestNeighbors(
        n_neighbors=spatial_k + 1,
        metric="euclidean",
        algorithm="auto",
    ).fit(coordinates).kneighbors(coordinates, return_distance=False)

    edges = set()
    for spot_index, candidates in enumerate(indices):
        neighbor_count = 0
        for candidate in candidates:
            neighbor_index = int(candidate)
            if neighbor_index == spot_index:
                continue
            edges.add(tuple(sorted((spot_index, neighbor_index))))
            neighbor_count += 1
            if neighbor_count == spatial_k:
                break
        if neighbor_count != spatial_k:
            raise RuntimeError("failed to construct the requested spatial kNN graph")

    if not edges:
        raise ValueError("spatial kNN graph contains no edges")
    ordered_edges = np.asarray(sorted(edges), dtype=np.int64)
    return ordered_edges[:, 0], ordered_edges[:, 1]


def boundary_aware_spatial_residual_refinement(
    embedding: Any,
    coordinates: Any,
    spatial_k: int = 3,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refine ``embedding`` with boundary-aware spatial neighbor residuals."""

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    z, output_dtype = _embedding_array(embedding)
    coords = _coordinate_array(coordinates, z.shape[0])
    left, right = _symmetrized_knn_edges(coords, spatial_k)

    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z_normalized = z / np.maximum(norms, eps)
    spatial_distances = np.linalg.norm(coords[left] - coords[right], axis=1)
    latent_distances = np.linalg.norm(
        z_normalized[left] - z_normalized[right], axis=1
    )

    positive_spatial = spatial_distances[spatial_distances > 0]
    positive_latent = latent_distances[latent_distances > 0]
    sigma_spatial = (
        float(np.median(positive_spatial)) if positive_spatial.size else 0.0
    )
    sigma_latent = (
        float(np.median(positive_latent)) if positive_latent.size else 0.0
    )
    if not np.isfinite(sigma_spatial) or sigma_spatial <= 0:
        raise ValueError("sigma_spatial must be positive")
    if not np.isfinite(sigma_latent) or sigma_latent <= 0:
        raise ValueError("sigma_latent must be positive")

    weights = np.exp(-0.5 * np.square(spatial_distances / sigma_spatial))
    weights *= np.exp(-0.5 * np.square(latent_distances / sigma_latent))

    weighted_sum = np.zeros_like(z)
    weight_sum = np.zeros(z.shape[0], dtype=np.float64)
    degree = np.zeros(z.shape[0], dtype=np.int64)
    np.add.at(weighted_sum, left, weights[:, None] * z[right])
    np.add.at(weighted_sum, right, weights[:, None] * z[left])
    np.add.at(weight_sum, left, weights)
    np.add.at(weight_sum, right, weights)
    np.add.at(degree, left, 1)
    np.add.at(degree, right, 1)

    if np.any(degree <= 0):
        raise RuntimeError("symmetrized spatial graph contains an isolated spot")
    if np.any(weight_sum <= 0) or not np.isfinite(weight_sum).all():
        raise RuntimeError("BSRR neighborhood weight sum must be positive and finite")

    neighborhood = weighted_sum / weight_sum[:, None]
    confidence = np.clip(weight_sum / degree, 0.0, 1.0)
    refined = (z + confidence[:, None] * neighborhood) / (
        1.0 + confidence[:, None]
    )
    if not np.isfinite(refined).all():
        raise RuntimeError("BSRR produced NaN or Inf")

    diagnostics: Dict[str, Any] = {
        "enabled": True,
        "method": "bsrr",
        "spatial_k": int(spatial_k),
        "sigma_spatial": sigma_spatial,
        "sigma_latent": sigma_latent,
        "confidence_mean": float(confidence.mean()),
        "confidence_std": float(confidence.std()),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
    }
    return refined.astype(output_dtype, copy=False), diagnostics


def apply_embedding_refinement(
    embedding: Any,
    coordinates: Any,
    *,
    enabled: bool,
    method: str = "bsrr",
    spatial_k: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the configured refinement, or return the raw embedding unchanged."""

    matrix, output_dtype = _embedding_array(embedding)
    method = str(method).lower()
    if method != "bsrr":
        raise ValueError("refinement.method must be 'bsrr'")
    if enabled:
        return boundary_aware_spatial_residual_refinement(
            matrix.astype(output_dtype, copy=False),
            coordinates,
            spatial_k=spatial_k,
        )
    return matrix.astype(output_dtype, copy=True), {
        "enabled": False,
        "method": method,
        "spatial_k": int(spatial_k),
        "sigma_spatial": None,
        "sigma_latent": None,
        "confidence_mean": None,
        "confidence_std": None,
        "confidence_min": None,
        "confidence_max": None,
    }
