"""Configurable clustering and ARI/NMI evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _kmeans(
    embedding: np.ndarray,
    n_clusters: int,
    *,
    n_init: int,
    random_state: int,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    return model.fit_predict(embedding)


def _mclust(embedding: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Run optional R mclust without making it a hard Python dependency."""

    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri as numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    ro.r["set.seed"](seed)
    mclust = importr("mclust")
    result = mclust.Mclust(np.asarray(embedding, dtype=np.float64), G=n_clusters)
    return np.asarray(result.rx2("classification"), dtype=np.int64).reshape(-1) - 1


def cluster_embedding(
    embedding: Any,
    n_clusters: int,
    *,
    method: str = "kmeans",
    seed: int = 0,
    n_init: int = 20,
    random_state: int = 0,
) -> Tuple[np.ndarray, str]:
    """Cluster an embedding and return labels plus the method actually used.

    ``seed`` controls mclust only. KMeans uses its explicit ``n_init`` and
    ``random_state`` arguments.
    """

    matrix = np.asarray(embedding, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("embedding must be a non-empty 2D matrix")
    if n_clusters < 2 or n_clusters > matrix.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of spots")

    requested = method.lower()
    if requested == "mclust":
        try:
            return _mclust(matrix, n_clusters, seed), "mclust"
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
            return (
                _kmeans(
                    matrix,
                    n_clusters,
                    n_init=n_init,
                    random_state=random_state,
                ),
                "kmeans_fallback_for_mclust",
            )
    if requested != "kmeans":
        raise ValueError("clustering method must be 'kmeans' or 'mclust'")
    return (
        _kmeans(
            matrix,
            n_clusters,
            n_init=n_init,
            random_state=random_state,
        ),
        "kmeans",
    )


def clustering_metrics(
    true_labels: Any,
    predicted_labels: Any,
    *,
    nmi_average_method: str = "max",
) -> Dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted_labels)
    if true_array.shape[0] != predicted_array.shape[0]:
        raise ValueError("true and predicted labels must have equal length")
    return {
        "ARI": float(adjusted_rand_score(true_array, predicted_array)),
        "NMI": float(
            normalized_mutual_info_score(
                true_array,
                predicted_array,
                average_method=nmi_average_method,
            )
        ),
    }
