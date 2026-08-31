# HiDFilter

The repository currently implements **Phase 2: Self + Physical Fine dependency spaces**. Semantic candidates, Family Evidence, Router, and Family Top-p are intentionally not implemented yet.

## Frozen environment

- Python 3.10.x
- PyTorch 2.1.2 + CUDA 11.8
- BasicTS v1.1.0 at `ab35018dda4dec03f642c6cc50c1b4cdcdd2e5b5`
- One NVIDIA RTX 3090, FP32, AMP off, DDP off

Initialize the environment from the repository root:

```bash
bash scripts/setup_phase0.sh
```

PEMS08 must contain `train_data.npy`, `val_data.npy`, `test_data.npy`, and the BasicTS `adj_mx.pkl` under `/root/autodl-tmp/datasets/PEMS08`. The Phase 2 Physical contract is explicitly frozen as:

```text
artifact: adj_mx.pkl
graph_mode: undirected
weight_semantics: affinity
conversion_scale: null
```

This contract follows BasicTS's PEMS08 preparation code in `third_party/BasicTS/scripts/data_preparation/PEMS08/generate_adj_mx.py`, which constructs a symmetric 0/1 connectivity matrix without self-loops. The runtime validates the artifact against the explicit contract and never infers graph semantics from adjacency values.

Run the local checks and the formal three-epoch Phase 2 CUDA sanity with:

```bash
pytest -q
python scripts/run_phase2_physical.py
```

The runner builds or strictly fingerprint-loads the offline `Kp=8` Physical candidate artifact, benchmarks `num_workers` in `0/2/4/8`, trains only on train/validation data, saves `best.pt` and `last.pt`, strictly reloads `best.pt`, and writes the report to `reports/phase2_physical_cuda.json`.
