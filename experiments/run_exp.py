"""Run one config-driven SpaGraMOR experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.clustering.modality_aware_readout import modality_aware_readout
from src.clustering.predict import cluster_embedding, clustering_metrics
from src.data.dataset import load_spatial_multiomics, sample_summary
from src.data.preprocessing import prepare_features
from src.graphs.feature_graph import build_feature_graph
from src.graphs.spatial_graph import build_spatial_graph
from src.models.spagramor import SpaGraMOR
from scripts.validate_formal_protocol import validate_ablation_config, validate_formal_config


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def _resolve_path(value: Any, *, base: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config must define a non-empty string at {field_name}")
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _resolve_config_path(raw_path: Path) -> Path:
    candidates = [raw_path.expanduser()]
    if not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Config file not found; searched: {searched}")


def _configured_output_dir(config: Mapping[str, Any]) -> Path:
    output = _section(config, "output")
    root = _resolve_path(output.get("root"), base=PROJECT_ROOT, field_name="output.root")
    name = _section(config, "experiment").get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Config must define experiment.name")
    output_dir = (root / name.strip()).resolve()
    if output_dir == root:
        raise ValueError("experiment.name must identify a child output directory")
    return output_dir


def _prepare_output_dir(
    config: Mapping[str, Any], resume_path: Optional[Path] = None
) -> Tuple[Path, bool]:
    output_dir = _configured_output_dir(config)
    if resume_path is not None:
        checkpoint_path = resume_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        if checkpoint_path.parent != output_dir:
            raise ValueError(
                "resume checkpoint must belong to the configured experiment output directory"
            )
        if (output_dir / "metrics.json").is_file():
            raise FileExistsError(
                f"Completed output cannot be resumed or overwritten: {output_dir}"
            )
        saved_config_path = output_dir / "config.yaml"
        if not saved_config_path.is_file():
            raise FileNotFoundError(
                f"resume output is missing its saved config: {saved_config_path}"
            )
        saved_config = _load_config(saved_config_path)
        if saved_config != dict(config):
            raise ValueError(
                "resume config differs from the config saved with the checkpoint; "
                "resume cannot change hyperparameters"
            )
        return output_dir, True

    if (output_dir / "metrics.json").is_file():
        raise FileExistsError(
            f"Output directory already contains metrics.json: {output_dir}; "
            "choose a new experiment.name"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --resume for recovery"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_dir, False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


def _to_sparse_tensor(adjacency: Any, device: torch.device) -> torch.Tensor:
    coo = adjacency.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, device=device
    ).coalesce()


def _build_graphs(
    config: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    coordinates: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    graphs = _section(config, "graphs")
    spatial = graphs.get("spatial", {})
    feature = graphs.get("feature", {})
    spatial_adjacency = build_spatial_graph(
        coordinates,
        k=int(spatial.get("k", 3)),
        include_self=bool(spatial.get("include_self", False)),
        symmetrize=bool(spatial.get("symmetrize", True)),
    )
    feature_adjacencies = {
        modality: build_feature_graph(
            matrix,
            k=int(feature.get("k", 20)),
            symmetrize=bool(feature.get("symmetrize", True)),
            clip_negative=bool(feature.get("clip_negative", True)),
        )
        for modality, matrix in features.items()
    }
    return spatial_adjacency, feature_adjacencies


def _preprocessing_options(config: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
    options = _section(config, "preprocessing")
    modality_options = options.get(modality)
    return modality_options if isinstance(modality_options, Mapping) else options


def _json_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    if not np.isfinite(result):
        raise FloatingPointError("non-finite scalar")
    return result


def _gradient_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum().cpu())
    return float(np.sqrt(squared))


def _save_manifest(output_dir: Path, config_path: Path, device: torch.device) -> None:
    def version(distribution: str, module: Any = None) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return str(getattr(module, "__version__", "unavailable"))

    manifest = {
        "project_root": str(PROJECT_ROOT),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": version("numpy", np),
        "scipy_version": version("scipy"),
        "sklearn_version": version("scikit-learn"),
        "anndata_version": version("anndata"),
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "source_config_path": str(config_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "history": list(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, list[Dict[str, Any]]]:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        state = torch.load(path, map_location=device)
    required = {
        "model", "optimizer", "epoch", "history", "python_rng_state",
        "numpy_rng_state", "torch_rng_state",
    }
    missing = required - set(state) if isinstance(state, Mapping) else required
    if missing:
        raise ValueError(
            "checkpoint is not an exact-resume checkpoint; missing "
            f"fields: {sorted(missing)}"
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    completed_epoch = int(state["epoch"])
    history = state["history"]
    if not isinstance(history, list):
        raise ValueError("checkpoint history must be a list")
    if history and int(history[-1]["epoch"]) != completed_epoch:
        raise ValueError("checkpoint history does not match completed epoch")
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].detach().cpu())
    if torch.cuda.is_available():
        cuda_state = state.get("cuda_rng_state")
        if cuda_state is None:
            raise ValueError("CUDA resume checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in cuda_state])
    return completed_epoch, [dict(record) for record in history]


def _integer_labels(labels: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(labels).astype(np.int64, copy=False)
    except (TypeError, ValueError):
        _, encoded = np.unique(np.asarray(labels).astype(str), return_inverse=True)
        return encoded.astype(np.int64, copy=False)


def _effective_coefficients(
    configured: Tuple[float, float, float],
    *,
    epoch: int,
    warm_up_epochs: int,
    ablation: Mapping[str, Any],
) -> Tuple[float, float, float, Dict[str, bool]]:
    rec = configured[0] if bool(ablation.get("reconstruction", True)) else 0.0
    mgcl = configured[1] if bool(ablation.get("sample_contrastive", True)) else 0.0
    cluster_enabled = bool(ablation.get("cluster_contrastive", True)) and epoch > warm_up_epochs
    cluster = configured[2] if cluster_enabled else 0.0
    return rec, mgcl, cluster, {
        "reconstruction": rec != 0.0,
        "sample_contrastive": mgcl != 0.0,
        "cluster_contrastive": cluster != 0.0,
    }


def _assert_config(config: Mapping[str, Any]) -> Tuple[int, int, float, float, float]:
    training = _section(config, "training")
    model = _section(config, "model")
    loss = _section(config, "loss")
    readout = _section(config, "readout")
    for path, section_name, field_name in (
        ("model.alpha", "model", "alpha"),
        ("training.spatial_start_epoch", "training", "spatial_start_epoch"),
        ("training.cluster_head_init", "training", "cluster_head_init"),
    ):
        if field_name in _section(config, section_name):
            raise ValueError(f"{path} is a legacy no-op and must be removed")
    epochs = int(training.get("epochs", 0))
    warm_up = int(training.get("warm_up_epochs", 0))
    if epochs < 1 or warm_up < 0:
        raise ValueError("training.epochs must be positive and warm_up_epochs nonnegative")
    if int(model.get("num_clusters", 0)) < 2:
        raise ValueError("model.num_clusters must be at least 2")
    required_losses = ("lambda_rec", "lambda_mgcl", "lambda_cluster", "lambda_spatial")
    if any(key not in loss for key in required_losses):
        raise ValueError("loss must define lambda_rec, lambda_mgcl, lambda_cluster, lambda_spatial")
    if float(loss["lambda_spatial"]) != 0.0:
        raise ValueError("SpaGraMOR formal training fixes lambda_spatial=0.0")
    spatial = _section(config, "spatial")
    if any(bool(spatial.get(key, False)) for key in ("enabled", "consistency_weighting", "negative_filter")):
        raise ValueError("legacy spatial mechanisms are disabled in SpaGraMOR")
    if not isinstance(readout, Mapping):
        raise ValueError("readout section is required")
    return (
        epochs,
        warm_up,
        float(loss["lambda_rec"]),
        float(loss["lambda_mgcl"]),
        float(loss["lambda_cluster"]),
    )


def _readout_from_output(
    output: Any,
    *,
    model: SpaGraMOR,
    sample: Any,
    n_clusters: int,
    readout_config: Dict[str, Any],
    labels: np.ndarray,
    nmi_average_method: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    z_concat = torch.cat(
        [output.multigranularity_views[name]["z"] for name in model.view_order],
        dim=1,
    ).detach().cpu().numpy()
    z_final, predicted, readout_diagnostics = modality_aware_readout(
        z_concat,
        sample.spatial_coordinates,
        second_modality=model.modality_names[1],
        n_clusters=n_clusters,
        readout_config=readout_config,
    )
    metrics = clustering_metrics(labels, predicted, nmi_average_method=nmi_average_method)
    return z_concat, z_final, predicted, {**readout_diagnostics, "metrics": metrics}


def _snapshot(
    output_dir: Path,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    predicted_metrics: Mapping[str, float],
    readout_diagnostics: Mapping[str, Any],
) -> None:
    payload = {
        "epoch": int(epoch),
        "ARI": float(predicted_metrics["ARI"]),
        "NMI": float(predicted_metrics["NMI"]),
        "readout": dict(readout_diagnostics),
        "loss_history": list(history),
    }
    (output_dir / f"metrics_epoch{epoch}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_experiment(config_path: Path, resume_path: Optional[Path] = None) -> Dict[str, Any]:
    config = _load_config(config_path)
    if config_path.parent.name == "formal":
        validate_formal_config(config, config_path)
    elif config_path.parent.name == "ablation":
        validate_ablation_config(config, config_path)
    output_dir, resumed = _prepare_output_dir(config, resume_path)
    experiment = _section(config, "experiment")
    data_config = _section(config, "data")
    training = _section(config, "training")
    model_config = dict(_section(config, "model"))
    loss_config = _section(config, "loss")
    ablation = _section(config, "ablation")
    readout_config = dict(_section(config, "readout"))
    epochs, warm_up, lambda_rec_config, lambda_mgcl_config, lambda_cluster_config = _assert_config(config)
    seed = int(experiment.get("seed", 0))
    dataset = str(experiment.get("dataset", ""))
    nmi_average_method = str(_section(config, "evaluation").get("nmi_average_method", "max"))
    _seed_everything(seed)
    device = _device_from_config(config)
    if not resumed or not (output_dir / "manifest.json").is_file():
        _save_manifest(output_dir, config_path, device)

    data_root = _resolve_path(data_config.get("root"), base=PROJECT_ROOT, field_name="data.root")
    n_pca = data_config.get("n_pca_components", 100)
    n_pca = None if n_pca is None else int(n_pca)
    sample = load_spatial_multiomics(
        data_root,
        dataset,
        label_key=data_config.get("label_key"),
        spatial_key=str(data_config.get("spatial_key", "spatial")),
        spatial_row_key=str(data_config.get("spatial_row_key", "array_row")),
        spatial_col_key=str(data_config.get("spatial_col_key", "array_col")),
        matrix_source=str(data_config.get("matrix_source", "X")),
        n_pca_components=n_pca,
    )
    features = {
        modality: prepare_features(
            matrix, modality, _preprocessing_options(config, modality)
        )
        for modality, matrix in sample.modality_features.items()
    }
    if tuple(features) != tuple(sample.modality_features) or len(features) != 2:
        raise ValueError("SpaGraMOR requires exactly two aligned modalities")
    first_modality, second_modality = tuple(features)
    if second_modality not in {"ADT", "ATAC"}:
        raise ValueError("second modality must be ADT or ATAC")
    expected_modalities = data_config.get("expected_modalities")
    if expected_modalities is not None and tuple(expected_modalities) != tuple(features):
        raise ValueError(
            "data.expected_modalities does not match loaded modality metadata: "
            f"{expected_modalities!r} != {list(features)!r}"
        )
    labels = sample.labels
    if labels is None:
        raise ValueError("evaluation requires labels in the configured AnnData")
    n_clusters = int(model_config.get("num_clusters", 0))
    if n_clusters < 2:
        raise ValueError("model.num_clusters must be at least 2")

    preprocessing_metadata = {}
    matrix_source = str(data_config.get("matrix_source", "X"))
    for modality, matrix in features.items():
        options = _preprocessing_options(config, modality)
        preprocessing_metadata[modality] = {
            "modality": modality,
            "raw_shape": list(sample.modality_raw_shapes[modality]),
            "model_input_shape": list(matrix.shape),
            "matrix_source": matrix_source,
            "n_pca_components": n_pca,
            "pca_enabled": n_pca is not None,
            "pca_whitening_enabled": n_pca is not None,
            "log1p": bool(options.get("log1p", False)),
            "standardize": bool(options.get("standardize", False)),
        }
    spatial_adjacency, feature_adjacencies = _build_graphs(
        config, features, sample.spatial_coordinates
    )
    spatial_tensor = _to_sparse_tensor(spatial_adjacency, device)
    feature_tensors = {
        modality: _to_sparse_tensor(adjacency, device)
        for modality, adjacency in feature_adjacencies.items()
    }
    inputs = {
        modality: torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
        for modality, matrix in features.items()
    }

    model_kwargs = dict(model_config)
    model_kwargs["temperature"] = float(model_kwargs.pop("tau_s", 0.3))
    model_kwargs["multigranularity_mode"] = str(ablation.get("multigranularity_mode", "full"))
    model_kwargs["view_weighting"] = str(ablation.get("view_weighting", "adaptive"))
    model_kwargs.pop("num_clusters", None)
    model = SpaGraMOR(
        input_dims={modality: int(matrix.shape[1]) for modality, matrix in features.items()},
        num_clusters=n_clusters,
        **model_kwargs,
    ).to(device)
    lr = float(training.get("lr", 0.001))
    weight_decay = float(training.get("weight_decay", 0.00001))
    cluster_lr_multiplier = float(training.get("cluster_head_lr_multiplier", 8.0))
    if lr <= 0 or weight_decay < 0 or cluster_lr_multiplier <= 0:
        raise ValueError("invalid optimizer parameters")
    cluster_params = list(model.cluster_head.parameters())
    cluster_ids = {id(parameter) for parameter in cluster_params}
    backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in cluster_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr},
            {"params": cluster_params, "lr": lr * cluster_lr_multiplier},
        ],
        weight_decay=weight_decay,
    )
    configured_losses = (lambda_rec_config, lambda_mgcl_config, lambda_cluster_config)
    history: list[Dict[str, Any]] = []
    completed_epoch = 0
    if resumed:
        completed_epoch, history = _load_checkpoint(
            resume_path.expanduser().resolve(), model, optimizer, device
        )
        if completed_epoch > epochs:
            raise ValueError("resume checkpoint completed epoch exceeds configured epochs")
    milestone_epochs = {50, 100, 200, 400}
    for epoch in range(completed_epoch + 1, epochs + 1):
        model.train()
        lambda_rec, lambda_mgcl, lambda_cluster, enabled_terms = _effective_coefficients(
            configured_losses,
            epoch=epoch,
            warm_up_epochs=warm_up,
            ablation=ablation,
        )
        output = model(
            inputs[first_modality], spatial_tensor, feature_tensors[first_modality],
            inputs[second_modality], spatial_tensor, feature_tensors[second_modality],
            lambda_rec=lambda_rec,
            lambda_mgcl=lambda_mgcl,
            lambda_cluster=lambda_cluster,
            modality_a_name=first_modality,
            modality_b_name=second_modality,
        )
        expected_total = (
            lambda_rec * output.reconstruction_loss
            + lambda_mgcl * output.sample_contrastive_loss
            + lambda_cluster * output.cluster_contrastive_loss
        )
        if not torch.isclose(output.total_loss, expected_total, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"total loss mismatch at epoch {epoch}")
        if not torch.isfinite(output.total_loss):
            raise FloatingPointError(f"non-finite total loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        output.total_loss.backward()
        cluster_grad_norm = _gradient_norm(model.cluster_head)
        optimizer.step()
        cluster_diag = model.cluster_contrastive_loss.diagnostics(output.cluster_assignments)
        record = {
            "epoch": epoch,
            "total": _json_float(output.total_loss),
            "reconstruction": _json_float(output.reconstruction_loss),
            "sample_contrastive": _json_float(output.sample_contrastive_loss),
            "cluster_contrastive": _json_float(output.cluster_contrastive_loss),
            "lambda_rec": lambda_rec,
            "lambda_mgcl": lambda_mgcl,
            "lambda_cluster": lambda_cluster,
            "effective_lambda_cluster": lambda_cluster,
            "configured_lambda_cluster": lambda_cluster_config,
            "ablation_terms": enabled_terms,
            "view_weighting": model.view_weighting,
            "mgcl_weight_std": _json_float(output.mgcl_weight_std),
            "neg_count": output.neg_count,
            "snf_enabled": False,
            "cluster_assignment_entropy": _json_float(cluster_diag["assignment_entropy"]),
            "cluster_effective_clusters": _json_float(cluster_diag["effective_clusters"]),
            "cluster_head_gradient_norm": cluster_grad_norm,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs:03d} | total={record['total']:.6f} | "
            f"rec={record['reconstruction']:.6f} | mgcl={record['sample_contrastive']:.6f} | "
            f"cluster={record['cluster_contrastive']:.6f} | "
            f"effective_lambda_cluster={record['effective_lambda_cluster']:.3g}"
        )
        if epoch in milestone_epochs:
            model.eval()
            with torch.no_grad():
                milestone_output = model(
                    inputs[first_modality], spatial_tensor, feature_tensors[first_modality],
                    inputs[second_modality], spatial_tensor, feature_tensors[second_modality],
                    lambda_rec=lambda_rec,
                    lambda_mgcl=lambda_mgcl,
                    lambda_cluster=lambda_cluster,
                    modality_a_name=first_modality,
                    modality_b_name=second_modality,
                )
            _, _, milestone_pred, milestone_info = _readout_from_output(
                milestone_output,
                model=model,
                sample=sample,
                n_clusters=n_clusters,
                readout_config=readout_config,
                labels=labels,
                nmi_average_method=nmi_average_method,
            )
            _snapshot(output_dir, epoch, history, milestone_info["metrics"], milestone_info)
            _save_checkpoint(
                output_dir / f"checkpoint_epoch{epoch}.pt",
                model,
                optimizer,
                epoch,
                history,
            )
            model.train()

    final_lambda_rec, final_lambda_mgcl, final_lambda_cluster, _ = _effective_coefficients(
        configured_losses,
        epoch=epochs,
        warm_up_epochs=warm_up,
        ablation=ablation,
    )
    model.eval()
    with torch.no_grad():
        final_output = model(
            inputs[first_modality], spatial_tensor, feature_tensors[first_modality],
            inputs[second_modality], spatial_tensor, feature_tensors[second_modality],
            lambda_rec=final_lambda_rec,
            lambda_mgcl=final_lambda_mgcl,
            lambda_cluster=final_lambda_cluster,
            modality_a_name=first_modality,
            modality_b_name=second_modality,
        )
    z_concat, z_bsrr, predicted, readout_info = _readout_from_output(
        final_output,
        model=model,
        sample=sample,
        n_clusters=n_clusters,
        readout_config=readout_config,
        labels=labels,
        nmi_average_method=nmi_average_method,
    )
    gt_labels = _integer_labels(labels)
    if len(predicted) != len(gt_labels) or z_concat.shape[0] != len(gt_labels):
        raise RuntimeError("prediction and embedding lengths differ from GT")
    if not np.isfinite(z_concat).all() or not np.isfinite(z_bsrr).all():
        raise RuntimeError("final embeddings contain NaN or Inf")
    official_metrics = clustering_metrics(
        gt_labels, predicted, nmi_average_method=nmi_average_method
    )
    if any(not np.isclose(official_metrics[key], readout_info["metrics"][key], atol=1e-12, rtol=0.0) for key in ("ARI", "NMI")):
        raise RuntimeError("official metrics are not reproducible")
    q_mean = torch.stack(list(final_output.cluster_assignments), dim=0).mean(dim=0)
    pred_q = q_mean.argmax(dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
    raw_kmeans_config = dict(_section(readout_config, "adt"))
    raw_pred, _ = cluster_embedding(
        z_concat,
        n_clusters,
        method="kmeans",
        n_init=int(raw_kmeans_config.get("n_init", 20)),
        random_state=int(raw_kmeans_config.get("random_state", 0)),
    )
    raw_metrics = clustering_metrics(gt_labels, raw_pred, nmi_average_method=nmi_average_method)
    z_mean = torch.stack(
        [final_output.multigranularity_views[name]["z"] for name in model.view_order], dim=0
    ).mean(dim=0).detach().cpu().numpy()
    np.save(output_dir / "pred_labels.npy", predicted.astype(np.int64, copy=False))
    np.save(output_dir / "pred_q_argmax.npy", pred_q)
    np.save(output_dir / "pred_raw_concat_z_kmeans.npy", raw_pred.astype(np.int64, copy=False))
    np.save(output_dir / "gt_labels.npy", gt_labels)
    np.save(output_dir / "spot_ids.npy", np.asarray(sample.spot_ids, dtype=str))
    np.save(output_dir / "coords.npy", np.asarray(sample.spatial_coordinates, dtype=np.float32))
    np.save(output_dir / "embeddings.npy", np.asarray(z_bsrr, dtype=np.float32))
    np.save(output_dir / "z_concat.npy", np.asarray(z_concat, dtype=np.float32))
    np.save(output_dir / "z_bsrr.npy", np.asarray(z_bsrr, dtype=np.float32))
    np.save(output_dir / "z_mean.npy", np.asarray(z_mean, dtype=np.float32))
    np.save(output_dir / "view_weights.npy", final_output.weights.detach().cpu().numpy().astype(np.float32))
    _save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epochs, history)
    report = {
        "method": "SpaGraMOR",
        "dataset": dataset,
        "modalities": list(model.modality_names),
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "warm_up_epochs": warm_up,
        "cluster_loss_start_epoch": warm_up + 1,
        "nmi_average_method": nmi_average_method,
        "configured_loss": {
            "lambda_rec": lambda_rec_config,
            "lambda_mgcl": lambda_mgcl_config,
            "lambda_cluster": lambda_cluster_config,
            "lambda_spatial": 0.0,
        },
        "resolved_parameters": {
            "training": dict(training),
            "model": model_config,
            "loss": dict(loss_config),
            "ablation": dict(ablation),
            "readout": readout_config,
        },
        "readout": readout_info,
        "ARI": official_metrics["ARI"],
        "NMI": official_metrics["NMI"],
        "official_metrics": official_metrics,
        "raw_concat_z_kmeans_metrics": raw_metrics,
        "candidate_metrics": {
            "raw_concat_z_kmeans": raw_metrics,
            "q_argmax": clustering_metrics(gt_labels, pred_q, nmi_average_method=nmi_average_method),
        },
        "preprocessing": preprocessing_metadata,
        "labels_used_for_training": False,
        "data": sample_summary(sample),
        "graph_stats": {
            "spatial": {"shape": list(spatial_adjacency.shape), "nnz": int(spatial_adjacency.nnz)},
            "feature": {modality: {"shape": list(adj.shape), "nnz": int(adj.nnz)} for modality, adj in feature_adjacencies.items()},
        },
        "loss_history": history,
        "artifacts": {
            name: str(output_dir / name)
            for name in (
                "config.yaml", "manifest.json", "checkpoint_last.pt", "pred_labels.npy",
                "pred_q_argmax.npy", "pred_raw_concat_z_kmeans.npy", "gt_labels.npy",
                "spot_ids.npy", "coords.npy", "embeddings.npy", "z_concat.npy", "z_bsrr.npy",
                "z_mean.npy", "view_weights.npy",
            )
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ARI={official_metrics['ARI']:.6f} | NMI={official_metrics['NMI']:.6f}")
    print(f"Output directory: {output_dir}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="resume an interrupted run from a checkpoint")
    args = parser.parse_args(argv)
    resume_path = args.resume.expanduser().resolve() if args.resume is not None else None
    run_experiment(_resolve_config_path(args.config), resume_path=resume_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
