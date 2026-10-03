# SpaGraMOR

SpaGraMOR is a multi-granularity graph representation learning framework
with modality-aware readout for spatial multi-omics clustering.

The training path builds four views: RNA spatial, RNA feature, second-modality
spatial, and second-modality feature. Each view produces `Z_v`, `H_v`, and
`G_v`; adaptive Wasserstein weighting and sample-level contrastive learning
operate on `G_v`, while the shared cluster head operates on `Z_v`.

The final embedding is always `concat(Z_v)`. BSRR is applied to that embedding
when enabled. RNA+ADT uses fixed KMeans. RNA+ATAC uses the configured
modality-aware spectral readout; `readout.mode: kmeans_all` is its control.

Run one experiment from the project root:

```bash
python experiments/run_exp.py --config configs/formal/e185.yaml
python scripts/audit_run.py results/spagramor_e185
```

Readout-only controls can reuse a completed run without retraining:

```bash
python scripts/evaluate_readout.py results/spagramor_e185 --without-bsrr
python scripts/evaluate_readout.py results/spagramor_e185 --mode kmeans_all
```

The runner saves the resolved config, manifest, checkpoints, embeddings,
predictions, metrics, and milestone snapshots. Existing result directories
containing `metrics.json` are rejected.

The formal configurations use 400 epochs and the frozen protocol defined in
`configs/formal/`. `configs/ablation/base_ablation.yaml` exposes the training
and readout switches without generating a dataset-by-variant experiment grid.
