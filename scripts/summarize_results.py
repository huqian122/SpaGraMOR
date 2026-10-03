"""Summarize completed SpaGraMOR result directories with population SD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path.parent.name, float(report["ARI"]), float(report["NMI"])))
    if not rows:
        raise SystemExit(f"No metrics.json found under {args.root}")
    aris = np.asarray([row[1] for row in rows])
    nmis = np.asarray([row[2] for row in rows])
    print("name,ARI,NMI")
    for row in rows:
        print(f"{row[0]},{row[1]:.8f},{row[2]:.8f}")
    print(f"mean,{aris.mean():.8f},{nmis.mean():.8f}")
    print(f"population_sd(ddof=0),{aris.std(ddof=0):.8f},{nmis.std(ddof=0):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
