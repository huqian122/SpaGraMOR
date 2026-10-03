"""Audit one SpaGraMOR result directory without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.modality_aware_readout import modality_aware_readout
from src.clustering.predict import cluster_embedding, clustering_metrics
from src.clustering.refinement import boundary_aware_spatial_residual_refinement


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _close(left: float, right: float, name: str) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name} differs: {left} != {right}")


def audit_run(output_dir: Path) -> None:
    metrics = _json(output_dir / "metrics.json")
    manifest = _json(output_dir / "manifest.json")
    config = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config.yaml must contain a mapping")
    required_manifest = {
        "project_root", "python_version", "torch_version", "numpy_version",
        "scipy_version", "sklearn_version", "anndata_version", "device",
        "cuda_version", "source_config_path",
    }
    missing = required_manifest - set(manifest)
    if missing:
        raise AssertionError(f"manifest missing fields: {sorted(missing)}")

    experiment = _section(config, "experiment")
    training = _section(config, "training")
    model = _section(config, "model")
    loss = _section(config, "loss")
    ablation = _section(config, "ablation")
    readout = _section(config, "readout")
    history = metrics.get("loss_history")
    if not isinstance(history, list) or not history:
        raise AssertionError("loss_history is missing")
    warm_up = int(training["warm_up_epochs"])
    configured_cluster = float(loss["lambda_cluster"])
    cluster_ablation = bool(ablation.get("cluster_contrastive", True))
    for record in history:
        epoch = int(record["epoch"])
        expected = configured_cluster if cluster_ablation and epoch > warm_up else 0.0
        _close(float(record["effective_lambda_cluster"]), expected, f"epoch {epoch} cluster lambda")
        expected_rec = float(loss["lambda_rec"]) if bool(ablation.get("reconstruction", True)) else 0.0
        expected_mgcl = float(loss["lambda_mgcl"]) if bool(ablation.get("sample_contrastive", True)) else 0.0
        _close(float(record["lambda_rec"]), expected_rec, f"epoch {epoch} rec lambda")
        _close(float(record["lambda_mgcl"]), expected_mgcl, f"epoch {epoch} MGCL lambda")

    pred = np.load(output_dir / "pred_labels.npy")
    pred_q = np.load(output_dir / "pred_q_argmax.npy")
    pred_raw_saved = np.load(output_dir / "pred_raw_concat_z_kmeans.npy")
    gt = np.load(output_dir / "gt_labels.npy")
    spot_ids = np.load(output_dir / "spot_ids.npy")
    coords = np.load(output_dir / "coords.npy")
    z_concat = np.load(output_dir / "z_concat.npy")
    z_bsrr = np.load(output_dir / "z_bsrr.npy")
    if (
        pred.ndim != 1
        or pred_q.ndim != 1
        or pred_raw_saved.ndim != 1
        or gt.ndim != 1
        or len(pred) != len(gt)
        or len(pred_q) != len(gt)
        or len(pred_raw_saved) != len(gt)
    ):
        raise AssertionError("prediction and GT arrays are inconsistent")
    if spot_ids.ndim != 1 or len(spot_ids) != len(gt):
        raise AssertionError("spot_ids and GT arrays are inconsistent")
    if coords.shape != (len(gt), 2):
        raise AssertionError("coords shape is invalid")
    if z_concat.shape[0] != len(gt) or z_bsrr.shape != z_concat.shape:
        raise AssertionError("embedding shapes are inconsistent")
    if not np.isfinite(z_concat).all() or not np.isfinite(z_bsrr).all():
        raise AssertionError("embedding contains NaN or Inf")
    if bool(_section(config, "spatial").get("enabled", False)):
        raise AssertionError("legacy spatial mechanism is enabled")
    if float(loss["lambda_spatial"]) != 0.0:
        raise AssertionError("lambda_spatial must be zero")

    metric_readout = metrics.get("readout")
    if not isinstance(metric_readout, Mapping):
        raise AssertionError("metrics.json is missing readout")
    resolved = _section(metrics, "resolved_parameters")
    resolved_readout = resolved.get("readout")
    if not isinstance(resolved_readout, Mapping):
        raise AssertionError("resolved_parameters.readout is missing")
    if dict(resolved_readout) != dict(readout):
        raise AssertionError("resolved readout differs from config")

    bsrr = _section(readout, "bsrr")
    bsrr_enabled = bool(bsrr.get("enabled", True))
    metric_refinement = _section(metric_readout, "refinement")
    expected_refinement = {
        "enabled": bsrr_enabled,
        "method": "bsrr",
        "spatial_k": int(bsrr.get("spatial_k", 3)),
    }
    for key, expected in expected_refinement.items():
        if metric_refinement.get(key) != expected:
            raise AssertionError(f"metrics refinement.{key} differs from config")
    if bsrr_enabled:
        recomputed_bsrr, recomputed_diag = boundary_aware_spatial_residual_refinement(
            z_concat, coords, spatial_k=int(bsrr.get("spatial_k", 3))
        )
        if not np.allclose(recomputed_bsrr, z_bsrr):
            raise AssertionError("z_bsrr.npy cannot be reproduced")
        for key in ("sigma_spatial", "sigma_latent", "confidence_mean", "confidence_std", "confidence_min", "confidence_max"):
            _close(float(metric_refinement[key]), float(recomputed_diag[key]), f"refinement.{key}")
    elif not np.array_equal(z_concat, z_bsrr):
        raise AssertionError("disabled BSRR must preserve concat Z")

    rerun_z, rerun_pred, rerun_info = modality_aware_readout(
        z_concat,
        coords,
        second_modality=str(metrics["modalities"][1]),
        n_clusters=int(model["num_clusters"]),
        readout_config=dict(readout),
    )
    if not np.allclose(rerun_z, z_bsrr) or not np.array_equal(rerun_pred, pred):
        raise AssertionError("official readout is not reproducible")
    if not isinstance(metrics.get("modalities"), list) or len(metrics["modalities"]) != 2:
        raise AssertionError("metrics.modalities must contain two modality names")
    second_modality = str(metrics["modalities"][1]).upper()
    metric_downstream = _section(metric_readout, "downstream")
    rerun_downstream = _section(rerun_info, "downstream")
    official_mode = str(readout.get("mode", "")).lower()
    if second_modality == "ATAC" and official_mode == "modality_aware":
        atac_config = _section(readout, "atac")
        for key, expected in (
            ("latent_k", 10),
            ("assign_labels", "kmeans"),
            ("n_init", 20),
            ("random_state", 0),
            ("eigen_solver", "arpack"),
        ):
            if atac_config.get(key) != expected:
                raise AssertionError(f"readout.atac.{key} must be {expected!r}")
            if metric_downstream.get(key) != expected or rerun_downstream.get(key) != expected:
                raise AssertionError(f"ATAC downstream {key} is inconsistent")
        if metric_downstream.get("connected_components") != 1 or rerun_downstream.get("connected_components") != 1:
            raise AssertionError("ATAC affinity must have one connected component")
        for name, downstream in (("metrics", metric_downstream), ("rerun", rerun_downstream)):
            if float(downstream.get("affinity_symmetry_max_error", np.inf)) > 1e-12:
                raise AssertionError(f"{name} ATAC affinity is not symmetric within tolerance")
            if downstream.get("spectral_warning_count") != 0:
                raise AssertionError(f"{name} ATAC spectral readout emitted a warning")
            if downstream.get("affinity_dtype") != "float64":
                raise AssertionError(f"{name} ATAC affinity must use float64")
    else:
        kmeans_config = _section(readout, "adt")
        def _same(actual: Any, expected: Any) -> bool:
            if isinstance(expected, str):
                return str(actual).lower() == expected
            return actual == expected

        for key, expected in (("method", "kmeans"), ("n_init", 20), ("random_state", 0)):
            if not _same(kmeans_config.get(key), expected):
                raise AssertionError(f"readout.adt.{key} must be {expected!r}")
            if not _same(metric_downstream.get(key), expected):
                raise AssertionError(f"metrics KMeans {key} is inconsistent")
            if not _same(rerun_downstream.get(key), expected):
                raise AssertionError(f"rerun KMeans {key} is inconsistent")
    for key in ("ARI", "NMI"):
        actual = clustering_metrics(
            gt, pred, nmi_average_method=str(_section(config, "evaluation").get("nmi_average_method", "max"))
        )[key]
        _close(float(metrics[key]), float(actual), key)
    raw_cfg = _section(readout, "adt")
    raw_pred, _ = cluster_embedding(
        z_concat,
        int(model["num_clusters"]),
        method="kmeans",
        n_init=int(raw_cfg.get("n_init", 20)),
        random_state=int(raw_cfg.get("random_state", 0)),
    )
    saved_raw = np.load(output_dir / "pred_raw_concat_z_kmeans.npy")
    if not np.array_equal(raw_pred, saved_raw):
        raise AssertionError("raw concat-Z diagnostic is not reproducible")

    print(f"Dataset: {metrics['dataset']}")
    print(f"Modalities: {metrics['modalities']}")
    print(f"Seed: {metrics['seed']}")
    print(f"Official readout: {readout.get('mode')}")
    print(f"Epochs: {metrics['epochs']}")
    print(f"ARI: {metrics['ARI']}")
    print(f"NMI: {metrics['NMI']}")
    print(f"z_concat shape: {tuple(z_concat.shape)}")
    print("PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        audit_run(args.output_dir.resolve())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
