# HiDFilter

The repository is currently limited to **Phase 0: BasicTS + STID CUDA infrastructure sanity**. No HiDFilter model component is implemented yet.

## Frozen environment

- Python 3.10.x
- PyTorch 2.1.2 + CUDA 11.8
- BasicTS v1.1.0 at `ab35018dda4dec03f642c6cc50c1b4cdcdd2e5b5`
- One NVIDIA RTX 3090, FP32, AMP off, DDP off

Initialize the environment from the repository root:

```bash
bash scripts/setup_phase0.sh
```

PEMS08 must already be split into `train_data.npy`, `val_data.npy`, and `test_data.npy` under `/root/autodl-tmp/datasets/PEMS08`. The runner validates the frozen 12-to-12 window counts before training and only constructs train/validation loaders for the development run.

```bash
pytest -q
python scripts/run_phase0_stid.py
```

The CUDA runner benchmarks `num_workers` in `0/2/4/8`, trains the official BasicTS STID through the traffic-only adapter, saves `best.pt` and `last.pt`, strictly reloads `best.pt`, and writes the synchronized runtime baseline to `reports/phase0_stid_cuda.json`.
