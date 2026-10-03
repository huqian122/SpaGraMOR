"""Evaluate readout-only controls from a saved embedding."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.modality_aware_readout import modality_aware_readout
from src.clustering.predict import clustering_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--mode", choices=("modality_aware", "kmeans_all"))
    parser.add_argument("--without-bsrr", action="store_true")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    readout = copy.deepcopy(config["readout"])
    if args.mode is not None:
        readout["mode"] = args.mode
    if args.without_bsrr:
        readout["bsrr"]["enabled"] = False
    z_concat = np.load(result_dir / "z_concat.npy")
    coords = np.load(result_dir / "coords.npy")
    gt = np.load(result_dir / "gt_labels.npy")
    _, predicted, diagnostics = modality_aware_readout(
        z_concat,
        coords,
        second_modality=str(report["modalities"][1]),
        n_clusters=int(config["model"]["num_clusters"]),
        readout_config=readout,
    )
    metrics = clustering_metrics(
        gt,
        predicted,
        nmi_average_method=str(config.get("evaluation", {}).get("nmi_average_method", "max")),
    )
    print(json.dumps({"readout": readout, "metrics": metrics, "diagnostics": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
