# HiDFilter

The repository currently implements **Phase 3: Self + Physical + Semantic Fine dependency spaces**. Family Evidence, Router, and Family Top-p are intentionally not implemented yet.

## Frozen environment

- Python 3.10.x
- PyTorch 2.1.2 + CUDA 11.8
- BasicTS v1.1.0 at `ab35018dda4dec03f642c6cc50c1b4cdcdd2e5b5`
- One NVIDIA RTX 3090, FP32, AMP off, DDP off

Initialize the environment from the repository root:

```bash
bash scripts/setup_phase0.sh
```

PEMS08 must contain `train_data.npy`, `val_data.npy`, `test_data.npy`, and the BasicTS `adj_mx.pkl` under `/root/autodl-tmp/datasets/PEMS08`. The Physical contract is explicitly frozen as:

```text
artifact: adj_mx.pkl
graph_mode: undirected
weight_semantics: affinity
conversion_scale: null
```

This contract follows BasicTS's PEMS08 preparation code in `third_party/BasicTS/scripts/data_preparation/PEMS08/generate_adj_mx.py`, which constructs a symmetric 0/1 connectivity matrix without self-loops. The runtime validates the artifact against the explicit contract and never infers graph semantics from adjacency values.

Semantic candidates use raw training traffic only, pair-specific common-valid first-difference Pearson correlation, `min_overlap=288`, variance threshold `1e-12`, and exclude self, original one-hop Physical neighbors, and selected Physical sources.

Run the local checks and the formal three-epoch Phase 3 CUDA sanity with:

```bash
pytest -q
python scripts/run_phase3_semantic.py
```

The runner builds or strictly fingerprint-loads the offline `Kp=8` Physical and `Ks=8` Semantic candidate artifacts, benchmarks `num_workers` in `0/2/4/8`, trains only on train/validation data, saves `best.pt` and `last.pt`, strictly reloads `best.pt`, and writes the report to `reports/phase3_semantic_cuda.json`.
