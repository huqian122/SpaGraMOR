"""Config-driven loading of paired spatial multi-omics AnnData files.

The metadata inspection phase uses backed AnnData objects. Experiment runs
intentionally load the configured feature matrices into memory so they can be
preprocessed, graphed, and optimized by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from src.graphs.spatial_graph import extract_spatial_coordinates


@dataclass(frozen=True)
class DatasetSpec:
    relative_dir: str
    modalities: Tuple[Tuple[str, str], Tuple[str, str]]
    default_label_key: str


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "Simulation": DatasetSpec(
        "simulation",
        (("RNA", "adata_RNA.h5ad"), ("ADT", "adata_ADT.h5ad")),
        "ground_truth",
    ),
    "HLN-A1": DatasetSpec(
        "Human_Lymph_Nodes/A1",
        (("RNA", "adata_RNA.h5ad"), ("ADT", "adata_ADT.h5ad")),
        "Spatial_Label",
    ),
    "HLN-D1": DatasetSpec(
        "Human_Lymph_Nodes/D1",
        (("RNA", "adata_RNA.h5ad"), ("ADT", "adata_ADT.h5ad")),
        "Spatial_Label",
    ),
    "E18.5": DatasetSpec(
        "E18.5_mouse_brain",
        (("RNA", "adata_RNA.h5ad"), ("ATAC", "adata_ATAC.h5ad")),
        "Combined_Clusters",
    ),
    "S2-E15": DatasetSpec(
        "Mouse_Embryos_S2/E15",
        (("RNA", "adata_RNA.h5ad"), ("ATAC", "adata_ATAC.h5ad")),
        "Spatial_Label",
    ),
    "S2-E18": DatasetSpec(
        "Mouse_Embryos_S2/E18",
        (("RNA", "adata_RNA.h5ad"), ("ATAC", "adata_ATAC.h5ad")),
        "Spatial_Label",
    ),
}


@dataclass
class SpatialMultiOmicsSample:
    """In-memory experiment inputs and verified alignment metadata."""

    dataset: str
    modality_features: Dict[str, np.ndarray]
    spatial_coordinates: np.ndarray
    labels: Optional[np.ndarray]
    spot_ids: np.ndarray
    modality_paths: Dict[str, Path]
    label_key: Optional[str]
    spatial_source: str
    modality_raw_shapes: Dict[str, Tuple[int, int]]
    modality_spot_ids: Dict[str, np.ndarray]

    @property
    def n_spots(self) -> int:
        return int(self.spatial_coordinates.shape[0])


def resolve_dataset_paths(root: Path, dataset: str) -> Dict[str, Path]:
    """Resolve and validate the two configured AnnData files."""

    if dataset not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {supported}")
    spec = DATASET_REGISTRY[dataset]
    root = Path(root).expanduser()
    paths = {
        modality: root / spec.relative_dir / filename
        for modality, filename in spec.modalities
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Configured data.root is missing dataset file(s): " + ", ".join(missing)
        )
    return paths


def _matrix_to_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    elif hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"feature matrix must be 2D; received shape {array.shape}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("feature matrix contains NaN or Inf")
    return array


def _get_matrix(adata: Any, matrix_source: str) -> Any:
    if matrix_source == "X":
        return adata.X
    if matrix_source.startswith("layers:"):
        layer_key = matrix_source.split(":", 1)[1]
        if layer_key not in adata.layers:
            raise KeyError(f"Requested layer {layer_key!r} is absent")
        return adata.layers[layer_key]
    raise ValueError("matrix_source must be 'X' or 'layers:<key>'")


def _pca_features(
    matrix: np.ndarray,
    modality: str,
    n_pca_components: Optional[int],
) -> np.ndarray:
    """Apply deterministic, modality-specific PCA whitening when configured.

    Whitening keeps the retained components on a comparable unit-variance
    scale. This matters for datasets whose modalities are stored in very
    different units, such as HLN-A1 RNA and ADT. Lower-dimensional inputs
    retain their feature count but are also whitened when this option is
    enabled, so raw-count ADT cannot dominate RNA.
    """

    if n_pca_components is None:
        return matrix
    if n_pca_components < 1:
        raise ValueError("n_pca_components must be positive or None")

    n_components = min(n_pca_components, matrix.shape[1], matrix.shape[0])
    if n_components < 1:
        raise ValueError(f"{modality} has no features or observations")

    # For lower-dimensional modalities, retain all features but still whiten
    # them so one raw-count modality cannot dominate the reconstruction loss.
    reducer = PCA(
        n_components=n_components,
        whiten=True,
        svd_solver="randomized" if n_components < min(matrix.shape) else "auto",
        random_state=0,
    )
    reduced = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
    if not np.isfinite(reduced).all():
        raise ValueError(f"PCA produced NaN or Inf for modality {modality}")
    return reduced


def _select_label_key(adata: Any, requested: Optional[str], default: str) -> str:
    if requested in {None, "", "UNRESOLVED"}:
        requested = default
    if requested not in adata.obs.columns:
        raise KeyError(
            f"Label field {requested!r} is absent; available obs fields are "
            f"{list(adata.obs.columns)!r}"
        )
    return str(requested)


def load_spatial_multiomics(
    root: Path,
    dataset: str,
    *,
    label_key: Optional[str] = None,
    spatial_key: str = "spatial",
    spatial_row_key: str = "array_row",
    spatial_col_key: str = "array_col",
    matrix_source: str = "X",
    n_pca_components: Optional[int] = None,
) -> SpatialMultiOmicsSample:
    """Load one paired dataset, optionally PCA-whiten each modality."""

    try:
        import anndata
    except ImportError as exc:
        raise RuntimeError("anndata is required to run experiments") from exc

    paths = resolve_dataset_paths(root, dataset)
    adatas = {modality: anndata.read_h5ad(path) for modality, path in paths.items()}
    try:
        modality_names = list(paths)
        reference = adatas[modality_names[0]]
        reference_ids = np.asarray([str(value) for value in reference.obs_names])
        if reference_ids.size == 0:
            raise ValueError("dataset contains no spots")

        for modality, adata in adatas.items():
            spot_ids = np.asarray([str(value) for value in adata.obs_names])
            if not np.array_equal(reference_ids, spot_ids):
                raise ValueError(
                    f"spot order mismatch between {modality_names[0]} and {modality}"
                )

        features = {}
        raw_shapes: Dict[str, Tuple[int, int]] = {}
        for modality, adata in adatas.items():
            matrix = _matrix_to_numpy(_get_matrix(adata, matrix_source))
            raw_shapes[modality] = tuple(int(value) for value in matrix.shape)
            reduced = _pca_features(matrix, modality, n_pca_components)
            features[modality] = reduced
        coordinates = extract_spatial_coordinates(
            reference,
            spatial_key=spatial_key,
            row_key=spatial_row_key,
            col_key=spatial_col_key,
        )
        if coordinates.coordinates.shape[0] != reference_ids.size:
            raise ValueError("spatial coordinate count does not match spot count")

        resolved_label_key = _select_label_key(
            reference, label_key, DATASET_REGISTRY[dataset].default_label_key
        )
        labels = np.asarray(reference.obs[resolved_label_key].to_numpy())
        return SpatialMultiOmicsSample(
            dataset=dataset,
            modality_features=features,
            spatial_coordinates=coordinates.coordinates,
            labels=labels,
            spot_ids=reference_ids,
            modality_paths=paths,
            label_key=resolved_label_key,
            spatial_source=coordinates.source,
            modality_raw_shapes=raw_shapes,
            modality_spot_ids={
                modality: np.asarray([str(value) for value in adata.obs_names])
                for modality, adata in adatas.items()
            },
        )
    finally:
        for adata in adatas.values():
            file_manager = getattr(adata, "file", None)
            if file_manager is not None:
                file_manager.close()


def sample_summary(sample: SpatialMultiOmicsSample) -> Mapping[str, Any]:
    """Return small, JSON-friendly metadata for experiment reports."""

    return {
        "dataset": sample.dataset,
        "n_spots": sample.n_spots,
        "modalities": {
            name: {"shape": list(features.shape), "dtype": str(features.dtype)}
            for name, features in sample.modality_features.items()
        },
        "raw_shapes": {
            name: list(shape) for name, shape in sample.modality_raw_shapes.items()
        },
        "spot_id_order_verified": True,
        "spatial_source": sample.spatial_source,
        "label_key": sample.label_key,
        "modality_paths": {name: str(path) for name, path in sample.modality_paths.items()},
    }
