"""Metadata-bounded inspection of the Phase 1 spatial multi-omics datasets.

This module intentionally opens AnnData files in backed read-only mode and
never reads ``adata.X`` values. Exact X min/max values are therefore reported
as UNRESOLVED_NOT_READ under the Phase 1 no-matrix-read boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS: Dict[str, Dict[str, Any]] = {
    "Simulation": {
        "relative_dir": "simulation",
        "modalities": {"RNA": "adata_RNA.h5ad", "ADT": "adata_ADT.h5ad"},
    },
    "HLN-A1": {
        "relative_dir": "Human_Lymph_Nodes/A1",
        "modalities": {"RNA": "adata_RNA.h5ad", "ADT": "adata_ADT.h5ad"},
    },
    "HLN-D1": {
        "relative_dir": "Human_Lymph_Nodes/D1",
        "modalities": {"RNA": "adata_RNA.h5ad", "ADT": "adata_ADT.h5ad"},
    },
    "E18.5": {
        "relative_dir": "E18.5_mouse_brain",
        "modalities": {"RNA": "adata_RNA.h5ad", "ATAC": "adata_ATAC.h5ad"},
    },
    "S2-E15": {
        "relative_dir": "Mouse_Embryos_S2/E15",
        "modalities": {"RNA": "adata_RNA.h5ad", "ATAC": "adata_ATAC.h5ad"},
    },
    "S2-E18": {
        "relative_dir": "Mouse_Embryos_S2/E18",
        "modalities": {"RNA": "adata_RNA.h5ad", "ATAC": "adata_ATAC.h5ad"},
    },
}

LABEL_PATTERN = re.compile(r"(label|domain|cluster|class|annotation|celltype)", re.I)
SPATIAL_PATTERN = re.compile(r"(spatial|coord|position|location|xy)", re.I)
LABEL_ALIAS_KEYS = ("ground_truth", "Spatial_Label")


@dataclass
class InspectionResult:
    dataset: str
    modality: str
    path: Path
    status: str
    details: Dict[str, Any]
    error: Optional[str] = None


def _load_config(path: Path) -> Dict[str, Any]:
    """Load YAML with PyYAML, keeping a precise error when it is unavailable."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read the inspection config. "
            "Install it with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def _config_root(config: Dict[str, Any], config_path: Path) -> Path:
    data = config.get("data", {})
    root_value = data.get("root")
    if not root_value or not isinstance(root_value, str):
        raise ValueError("Config must define a non-empty string at data.root")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root


def _as_list(values: Iterable[Any]) -> List[str]:
    return [str(value) for value in values]


def _candidate_keys(keys: Sequence[str], pattern: re.Pattern[str]) -> List[str]:
    return [key for key in keys if pattern.search(key)]


def _x_metadata(adata: Any) -> Dict[str, Any]:
    """Return X metadata without evaluating or materializing X values."""
    x = adata.X
    x_type = f"{type(x).__module__}.{type(x).__name__}"
    dtype = getattr(x, "dtype", "UNRESOLVED")
    is_sparse = False
    try:
        from scipy import sparse

        is_sparse = bool(sparse.issparse(x))
    except ImportError:
        is_sparse = "sparse" in x_type.lower()
    return {
        "type": x_type,
        "dtype": str(dtype),
        "sparsity": "sparse" if is_sparse else "dense_or_backed",
        "value_range": {
            "min": "UNRESOLVED_NOT_READ",
            "max": "UNRESOLVED_NOT_READ",
        },
    }


def inspect_file(dataset: str, modality: str, path: Path) -> InspectionResult:
    if not path.exists():
        return InspectionResult(
            dataset, modality, path, "MISSING", {}, error="file does not exist"
        )
    try:
        import anndata
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to inspect .h5ad files. "
            "Install the project inspection dependencies first."
        ) from exc

    try:
        # backed='r' exposes AnnData metadata and leaves X on disk.
        adata = anndata.read_h5ad(path, backed="r")
        obs_columns = _as_list(adata.obs.columns)
        var_columns = _as_list(adata.var.columns)
        obsm_keys = _as_list(adata.obsm.keys())
        uns_keys = _as_list(adata.uns.keys())
        layer_keys = _as_list(adata.layers.keys())
        obs_names = [str(value) for value in adata.obs_names]
        details = {
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "x": _x_metadata(adata),
            "obs_columns": obs_columns,
            "var_columns": var_columns,
            "obsm_keys": obsm_keys,
            "uns_keys": uns_keys,
            "layers_keys": layer_keys,
            "raw_exists": adata.raw is not None,
            "candidate_label_fields": sorted(
                set(_candidate_keys(obs_columns, LABEL_PATTERN)).union(
                    {key for key in LABEL_ALIAS_KEYS if key in obs_columns}
                )
            ),
            "candidate_spatial_fields": _candidate_keys(obsm_keys, SPATIAL_PATTERN),
            "spot_id_count": len(obs_names),
            "spot_ids": obs_names,
        }
        adata.file.close()
        return InspectionResult(dataset, modality, path, "OK", details)
    except Exception as exc:
        return InspectionResult(
            dataset,
            modality,
            path,
            "ERROR",
            {},
            error=f"{type(exc).__name__}: {exc}",
        )


def inspect_datasets(root: Path, dataset_names: Sequence[str]) -> List[InspectionResult]:
    results: List[InspectionResult] = []
    for dataset in dataset_names:
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported Phase 1 dataset: {dataset}")
        spec = DATASETS[dataset]
        for modality, filename in spec["modalities"].items():
            path = root / spec["relative_dir"] / filename
            results.append(inspect_file(dataset, modality, path))
    return results


def alignment_summary(results: Sequence[InspectionResult]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[InspectionResult]] = {}
    for result in results:
        grouped.setdefault(result.dataset, []).append(result)

    summary: Dict[str, Dict[str, Any]] = {}
    for dataset, items in grouped.items():
        usable = [item for item in items if item.status == "OK"]
        if len(usable) != len(items):
            summary[dataset] = {
                "status": "UNRESOLVED",
                "reason": "one or more modality files could not be inspected",
            }
            continue
        reference_ids = usable[0].details["spot_ids"]
        mismatches = [
            item.modality
            for item in usable[1:]
            if item.details["spot_ids"] != reference_ids
        ]
        summary[dataset] = {
            "status": "MATCH" if not mismatches else "MISMATCH",
            "reference_modality": usable[0].modality,
            "spot_count": len(reference_ids),
            "mismatched_modalities": mismatches,
        }
    return summary


def _md_list(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "`[]`"


def _write_result(report: List[str], result: InspectionResult) -> None:
    report.extend(
        [
            f"### {result.modality}: `{result.path}`",
            f"- Status: `{result.status}`",
        ]
    )
    if result.error:
        report.append(f"- Error: `{result.error}`")
        report.append("")
        return
    details = result.details
    report.extend(
        [
            f"- `adata.shape`: `{tuple(details['shape'])}`",
            f"- `adata.X.type`: `{details['x']['type']}`",
            f"- `adata.X.dtype`: `{details['x']['dtype']}`",
            f"- `adata.X.sparsity`: `{details['x']['sparsity']}`",
            f"- `adata.X.min`: `{details['x']['value_range']['min']}`",
            f"- `adata.X.max`: `{details['x']['value_range']['max']}`",
            f"- `adata.obs.columns`: {_md_list(details['obs_columns'])}",
            f"- `adata.var.columns`: {_md_list(details['var_columns'])}",
            f"- `adata.obsm.keys()`: {_md_list(details['obsm_keys'])}",
            f"- `adata.uns.keys()`: {_md_list(details['uns_keys'])}",
            f"- `adata.layers.keys()`: {_md_list(details['layers_keys'])}",
            f"- `adata.raw`: `{details['raw_exists']}`",
            f"- Candidate label fields: {_md_list(details['candidate_label_fields'])}",
            f"- Candidate spatial fields: {_md_list(details['candidate_spatial_fields'])}",
            f"- Spot ID count: `{details['spot_id_count']}`",
            "",
        ]
    )


def write_report(
    output_path: Path,
    config_path: Path,
    root: Path,
    results: Sequence[InspectionResult],
) -> None:
    report = [
        "# DATA PROFILE",
        "",
        "Status: Implementation Phase 1 data inspection",
        "Date: 2026-09-05",
        "",
        "## Inspection boundary",
        "",
        "- Files were opened with AnnData `backed='r'`.",
        "- Complete `adata.X` values were not read or materialized.",
        "- Exact X min/max values are therefore `UNRESOLVED_NOT_READ`.",
        "- No preprocessing, graph construction, model code, or training ran.",
        f"- Config: `{config_path}`",
        f"- Resolved `data.root`: `{root}`",
        "",
        "## File-level results",
        "",
    ]
    for result in results:
        _write_result(report, result)

    alignment = alignment_summary(results)
    report.extend(["## Spot ID alignment", ""])
    for dataset, info in alignment.items():
        report.append(f"### {dataset}")
        report.append(f"- Status: `{info['status']}`")
        if info["status"] == "MATCH":
            report.append(f"- Reference modality: `{info['reference_modality']}`")
            report.append(f"- Spot count: `{info['spot_count']}`")
            report.append("- Compared `obs_names` in their stored order.")
        elif info["status"] == "MISMATCH":
            report.append(
                f"- Mismatched modalities: {_md_list(info['mismatched_modalities'])}"
            )
        else:
            report.append(f"- Reason: `{info['reason']}`")
        report.append("")

    report.extend(
        [
            "## Field decisions",
            "",
        "Candidate fields are reported by name pattern only. A final label or "
        "spatial field is not selected in Phase 1 unless its semantics are "
        "verified from the dataset metadata.",
        "",
            "- Label field decision: `UNRESOLVED` pending semantic verification; "
            "known aliases such as `ground_truth` and `Spatial_Label` are listed "
            "only as candidates.",
            "- Spatial field decision: `UNRESOLVED` pending shape/content verification.",
            "- Preprocessing state: `UNRESOLVED` until bounded value diagnostics "
            "are authorized and completed.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/DATA_PROFILE.md"),
        help="Markdown report path, resolved relative to the project root.",
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    root = _config_root(config, config_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Configured data.root does not exist: {root}")

    requested = config.get("inspection", {}).get(
        "datasets", ["Simulation", "HLN-A1", "E18.5"]
    )
    results = inspect_datasets(root, requested)
    project_root = Path(__file__).resolve().parents[2]
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = project_root / output_path
    write_report(output_path, config_path, root, results)

    for result in results:
        print(f"[{result.status}] {result.dataset}/{result.modality}: {result.path}")
    print(f"Wrote report: {output_path}")
    print(json.dumps(alignment_summary(results), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
