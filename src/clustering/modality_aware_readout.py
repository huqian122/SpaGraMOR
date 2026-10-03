"""Final clustering readouts for RNA+ADT and RNA+ATAC experiments."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

from src.clustering.refinement import apply_embedding_refinement


def _validate_embedding(embedding: Any) -> np.ndarray:
    matrix = np.asarray(embedding, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embedding must be a non-empty N x D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding contains NaN or Inf")
    return matrix


def _atac_spectral_readout(
    embedding: np.ndarray,
    n_clusters: int,
    *,
    latent_k: int,
    n_init: int,
    random_state: int,
    eigen_solver: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_spots = embedding.shape[0]
    if latent_k < 1 or latent_k >= n_spots:
        raise ValueError("readout.atac.latent_k must be between 1 and N - 1")
    # Keep BSRR's input/output behavior unchanged. The historical ATAC
    # spectral protocol performs all downstream numeric work in float64.
    embedding64 = np.asarray(embedding, dtype=np.float64)
    normalized = embedding64 / np.maximum(
        np.linalg.norm(embedding64, axis=1, keepdims=True), 1e-12
    )
    nn = NearestNeighbors(
        n_neighbors=latent_k + 1,
        metric="euclidean",
        algorithm="auto",
    )
    nn.fit(normalized)
    distances, indices = nn.kneighbors(normalized)
    directed_distances = []
    rows = []
    cols = []
    for row_index, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        kept = 0
        for distance, column_index in zip(row_distances, row_indices):
            column_index = int(column_index)
            if column_index == row_index:
                continue
            rows.append(row_index)
            cols.append(column_index)
            directed_distances.append(float(distance))
            kept += 1
            if kept == latent_k:
                break
        if kept != latent_k:
            raise RuntimeError("failed to remove self from latent kNN")
    directed_distances_array = np.asarray(directed_distances, dtype=np.float64)
    positive_distances = directed_distances_array[directed_distances_array > 0]
    sigma = float(np.median(positive_distances)) if positive_distances.size else 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("ATAC latent-kNN sigma must be positive")
    directed_weights = np.exp(
        -np.square(directed_distances_array) / (2.0 * sigma * sigma)
    )
    affinity = sparse.coo_matrix(
        (directed_weights, (np.asarray(rows), np.asarray(cols))),
        shape=(n_spots, n_spots),
        dtype=np.float64,
    ).tocsr()
    affinity = affinity.maximum(affinity.T).tocsr()
    if not np.isfinite(affinity.data).all():
        raise ValueError("ATAC affinity contains NaN or Inf")
    components, _ = connected_components(affinity, directed=False)
    warning_count = 0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        predicted = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=n_init,
            random_state=random_state,
            eigen_solver=eigen_solver,
        ).fit_predict(affinity)
        warning_count = len(captured)
    diagnostics = {
        "method": "spectral",
        "latent_k": int(latent_k),
        "sigma": sigma,
        "directed_edge_count": int(len(directed_distances_array)),
        "affinity_nnz": int(affinity.nnz),
        "connected_components": int(components),
        "affinity_symmetry_max_error": float(
            np.max(np.abs((affinity - affinity.T).data))
            if (affinity - affinity.T).nnz
            else 0.0
        ),
        "spectral_warning_count": int(warning_count),
        "affinity_dtype": str(affinity.dtype),
        "assign_labels": "kmeans",
        "n_init": int(n_init),
        "random_state": int(random_state),
        "eigen_solver": eigen_solver,
    }
    return np.asarray(predicted, dtype=np.int64), diagnostics


def modality_aware_readout(
    embedding: Any,
    coordinates: Any,
    *,
    second_modality: str,
    n_clusters: int,
    readout_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply optional BSRR and route by actual modality metadata.

    ``readout_mode=modality_aware`` selects SpectralClustering only for ATAC;
    ``kmeans_all`` uses the configured KMeans readout for both modalities.
    """

    z_concat = _validate_embedding(embedding)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (z_concat.shape[0], 2):
        raise ValueError("coordinates must have shape (N, 2) matching embedding")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain NaN or Inf")
    mode = str(readout_config.get("mode", "modality_aware")).lower()
    if mode not in {"modality_aware", "kmeans_all"}:
        raise ValueError("readout.mode must be modality_aware or kmeans_all")
    bsrr_config = readout_config.get("bsrr", {})
    if not isinstance(bsrr_config, dict):
        raise ValueError("readout.bsrr must be a mapping")
    bsrr_enabled = bool(bsrr_config.get("enabled", True))
    bsrr_k = int(bsrr_config.get("spatial_k", 3))
    z_final, refinement_diagnostics = apply_embedding_refinement(
        z_concat,
        coordinates,
        enabled=bsrr_enabled,
        method="bsrr",
        spatial_k=bsrr_k,
    )
    second_modality = str(second_modality).upper()
    if second_modality not in {"ADT", "ATAC"}:
        raise ValueError(
            "modality-aware readout supports second modality ADT or ATAC; "
            f"received {second_modality!r}"
        )
    if second_modality == "ATAC" and mode == "modality_aware":
        atac_config = readout_config.get("atac", {})
        if str(atac_config.get("method", "spectral")).lower() != "spectral":
            raise ValueError("readout.atac.method must be spectral")
        if str(atac_config.get("assign_labels", "kmeans")).lower() != "kmeans":
            raise ValueError("readout.atac.assign_labels must be kmeans")
        predicted, downstream = _atac_spectral_readout(
            z_final,
            n_clusters,
            latent_k=int(atac_config.get("latent_k", 10)),
            n_init=int(atac_config.get("n_init", 20)),
            random_state=int(atac_config.get("random_state", 0)),
            eigen_solver=str(atac_config.get("eigen_solver", "arpack")),
        )
    else:
        kmeans_config = readout_config.get("adt", {})
        if str(kmeans_config.get("method", "kmeans")).lower() != "kmeans":
            raise ValueError("readout.adt.method must be kmeans")
        n_init = int(kmeans_config.get("n_init", 20))
        random_state = int(kmeans_config.get("random_state", 0))
        predicted = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        ).fit_predict(z_final)
        predicted = np.asarray(predicted, dtype=np.int64)
        downstream = {
            "method": "kmeans",
            "n_init": n_init,
            "random_state": random_state,
        }
    diagnostics = {
        "mode": mode,
        "second_modality": second_modality,
        "refinement": refinement_diagnostics,
        "downstream": downstream,
    }
    return z_final, predicted, diagnostics
