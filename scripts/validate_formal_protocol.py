"""Validate SpaGraMOR's frozen formal and ablation configuration protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProtocolError(ValueError):
    """Raised when a configuration violates the frozen experiment protocol."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _value(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ProtocolError(f"missing required field: {path}")
        current = current[part]
    return current


def _expect(config: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _value(config, path)
    if isinstance(expected, float):
        valid = isinstance(actual, (int, float)) and np.isclose(
            float(actual), expected, rtol=0.0, atol=1e-12
        )
    else:
        valid = actual == expected
    if not valid:
        raise ProtocolError(f"{path} must be {expected!r}; received {actual!r}")


def _reject_legacy_noops(config: Mapping[str, Any]) -> None:
    for path in ("model.alpha", "training.spatial_start_epoch", "training.cluster_head_init"):
        current: Any = config
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.get(part, {}) if isinstance(current, Mapping) else {}
        if isinstance(current, Mapping) and parts[-1] in current:
            raise ProtocolError(f"legacy no-op field must be removed: {path}")


def _legacy_switch_enabled(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("enabled", False))
    return bool(value)


def _validate_common(config: Mapping[str, Any], *, formal: bool) -> None:
    _reject_legacy_noops(config)
    frozen = {
        "training.epochs": 400,
        "training.warm_up_epochs": 10,
        "training.lr": 0.001,
        "training.weight_decay": 0.00001,
        "training.cluster_head_lr_multiplier": 8.0,
        "data.n_pca_components": 100,
        "preprocessing.log1p": False,
        "preprocessing.standardize": False,
        "graphs.spatial.k": 3,
        "graphs.spatial.include_self": False,
        "graphs.spatial.symmetrize": True,
        "graphs.feature.k": 20,
        "graphs.feature.symmetrize": True,
        "graphs.feature.clip_negative": True,
        "model.gcn_hidden_dim": 32,
        "model.fine_dim": 128,
        "model.coarse_dim": 16,
        "model.representation_dim": 16,
        "model.fusion_hidden_dim": 64,
        "model.tau_s": 0.3,
        "model.cluster_temperature": 0.3,
        "model.cluster_regularization_weight": 1.0,
        "loss.lambda_spatial": 0.0,
        "spatial.enabled": False,
        "spatial.consistency_weighting": False,
        "spatial.negative_filter": False,
        "evaluation.nmi_average_method": "max",
        "evaluation.summary_ddof": 0,
        "readout.bsrr.spatial_k": 3,
        "readout.adt.method": "kmeans",
        "readout.adt.n_init": 20,
        "readout.adt.random_state": 0,
        "readout.atac.method": "spectral",
        "readout.atac.latent_k": 10,
        "readout.atac.assign_labels": "kmeans",
        "readout.atac.n_init": 20,
        "readout.atac.random_state": 0,
        "readout.atac.eigen_solver": "arpack",
    }
    if formal:
        frozen.update(
            {
                "loss.lambda_rec": 1.0,
                "loss.lambda_mgcl": 3.0,
                "loss.lambda_cluster": 0.1,
            }
        )
    for path, expected in frozen.items():
        _expect(config, path, expected)

    for legacy_name in ("sc", "snf"):
        if legacy_name in config and _legacy_switch_enabled(config[legacy_name]):
            raise ProtocolError(f"{legacy_name} must be disabled")

    readout = _section(config, "readout")
    mode = str(readout.get("mode", "")).lower()
    if formal and mode != "modality_aware":
        raise ProtocolError("formal readout.mode must be modality_aware")
    if not formal and mode not in {"modality_aware", "kmeans_all"}:
        raise ProtocolError("ablation readout.mode must be modality_aware or kmeans_all")
    if formal:
        _expect(config, "readout.bsrr.enabled", True)
    elif not isinstance(_value(config, "readout.bsrr.enabled"), bool):
        raise ProtocolError("ablation readout.bsrr.enabled must be boolean")


def validate_formal_config(config: Mapping[str, Any], path: Optional[Path] = None) -> None:
    """Validate one formal config against the common frozen protocol."""

    _validate_common(config, formal=True)
    location = f" ({path})" if path is not None else ""
    if not _section(config, "experiment").get("dataset"):
        raise ProtocolError(f"experiment.dataset is required{location}")


def validate_ablation_config(config: Mapping[str, Any], path: Optional[Path] = None) -> None:
    """Validate an ablation template or concrete config."""

    _validate_common(config, formal=False)
    ablation = _section(config, "ablation")
    allowed = {
        "reconstruction",
        "sample_contrastive",
        "cluster_contrastive",
        "multigranularity_mode",
        "view_weighting",
    }
    if set(ablation) != allowed:
        raise ProtocolError(
            f"ablation switches must be exactly {sorted(allowed)}; received {sorted(ablation)}"
        )
    for name in ("reconstruction", "sample_contrastive", "cluster_contrastive"):
        if not isinstance(ablation[name], bool):
            raise ProtocolError(f"ablation.{name} must be boolean")
    if str(ablation["multigranularity_mode"]).lower() not in {
        "full", "fine_only", "coarse_only", "linear_fusion"
    }:
        raise ProtocolError("ablation.multigranularity_mode is invalid")
    if str(ablation["view_weighting"]).lower() not in {"adaptive", "uniform"}:
        raise ProtocolError("ablation.view_weighting is invalid")
    loss = _section(config, "loss")
    for field, full_value, switch in (
        ("lambda_rec", 1.0, "reconstruction"),
        ("lambda_mgcl", 3.0, "sample_contrastive"),
        ("lambda_cluster", 0.1, "cluster_contrastive"),
    ):
        actual = loss.get(field)
        if not isinstance(actual, (int, float)) or not np.isclose(
            float(actual), 0.0, rtol=0.0, atol=1e-12
        ) and not np.isclose(float(actual), full_value, rtol=0.0, atol=1e-12):
            raise ProtocolError(f"loss.{field} may only be 0 or the frozen value {full_value}")
        if bool(ablation[switch]) and not np.isclose(
            float(actual), full_value, rtol=0.0, atol=1e-12
        ):
            raise ProtocolError(
                f"loss.{field} must retain {full_value} when ablation.{switch}=true"
            )


def validate_directory(
    formal_dir: Path,
    ablation_path: Optional[Path] = None,
) -> None:
    formal_paths = sorted(formal_dir.glob("*.yaml"))
    if not formal_paths:
        raise ProtocolError(f"no formal YAML files found in {formal_dir}")
    for path in formal_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ProtocolError(f"{path} must contain a mapping")
        validate_formal_config(config, path)
    if ablation_path is not None:
        config = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ProtocolError(f"{ablation_path} must contain a mapping")
        validate_ablation_config(config, ablation_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=PROJECT_ROOT / "configs" / "formal")
    parser.add_argument(
        "--ablation",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ablation" / "base_ablation.yaml",
    )
    args = parser.parse_args(argv)
    try:
        validate_directory(args.formal_dir, args.ablation)
    except (OSError, ProtocolError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: validated {len(list(args.formal_dir.glob('*.yaml')))} formal configs")
    print(f"PASS: validated ablation config {args.ablation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
