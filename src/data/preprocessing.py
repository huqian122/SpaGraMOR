"""Small, explicit feature preprocessing helpers for experiment configs."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


def prepare_features(
    features: np.ndarray,
    modality: str,
    config: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Apply only transformations explicitly requested by the config.

    The supplied Phase 1 files already contain finite floating-point ``X``
    values for the P0 Simulation run. The default is therefore an identity
    transform. Optional log, column-standardization, and variance-based
    feature limiting are provided for later datasets without silently
    assuming that an ``.h5ad`` file stores raw counts.
    """

    options = dict(config or {})
    result = np.asarray(features, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{modality} features must be 2D")
    if not np.isfinite(result).all():
        raise ValueError(f"{modality} features contain NaN or Inf")

    if bool(options.get("log1p", False)):
        if np.min(result) < 0:
            raise ValueError("log1p preprocessing requires nonnegative features")
        result = np.log1p(result).astype(np.float32, copy=False)

    if bool(options.get("standardize", False)):
        mean = result.mean(axis=0, keepdims=True)
        scale = result.std(axis=0, keepdims=True)
        result = ((result - mean) / np.maximum(scale, 1e-6)).astype(
            np.float32, copy=False
        )

    max_features = options.get("max_features")
    if max_features is not None:
        max_features = int(max_features)
        if max_features < 1:
            raise ValueError("max_features must be positive")
        if result.shape[1] > max_features:
            variance = result.var(axis=0)
            selected = np.argsort(variance)[-max_features:]
            selected.sort()
            result = result[:, selected]

    if not np.isfinite(result).all():
        raise ValueError(f"{modality} preprocessing produced NaN or Inf")
    return result
