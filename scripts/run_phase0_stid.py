#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from hidfilter.protocol.pems08 import (
    NUM_NODES,
    TrafficOnlyForecastingDataset,
    fit_train_scaler,
    move_scaler_to_device,
    validate_pems08_protocol,
)
from hidfilter.runtime.checkpoint import CheckpointManager, load_model_checkpoint
from hidfilter.runtime.determinism import configure_determinism
from hidfilter.runtime.environment import BASICTS_COMMIT, verify_basicts_revision
from hidfilter.runtime.gpu_monitor import NvidiaSmiSampler
from hidfilter.runtime.phase0 import (
    benchmark_worker_counts,
    build_dataloader,
    build_optimizer,
    evaluate,
    select_worker_count,
    train_one_epoch,
)
from hidfilter.runtime.stid import build_traffic_only_stid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Phase 0 BasicTS STID CUDA sanity")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase0_stid_pems08.json",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--allow-non-target-environment",
        action="store_true",
        help="Diagnostic only: bypass Python/PyTorch/RTX 3090 enforcement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_dir = args.dataset_dir or Path(config["dataset_dir"])
    checkpoint_dir = args.checkpoint_dir or REPOSITORY_ROOT / config["checkpoint_dir"]
    report_path = args.report_path or REPOSITORY_ROOT / config["report_path"]
    epochs = args.epochs or int(config["epochs"])

    configure_determinism(int(config["seed"]))
    revision = verify_basicts_revision(REPOSITORY_ROOT / "third_party" / "BasicTS")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.allow_non_target_environment:
        _assert_target_environment(device)
    protocol_counts = validate_pems08_protocol(dataset_dir)

    train_dataset = TrafficOnlyForecastingDataset(
        dataset_dir, "train", memmap=True, expected_num_nodes=NUM_NODES
    )
    val_dataset = TrafficOnlyForecastingDataset(
        dataset_dir, "val", memmap=True, expected_num_nodes=NUM_NODES
    )
    scaler = move_scaler_to_device(fit_train_scaler(train_dataset), device)
    pin_memory = device.type == "cuda"
    worker_results = benchmark_worker_counts(
        train_dataset,
        batch_size=int(config["batch_size"]),
        candidates=tuple(config["worker_candidates"]),
        max_batches=int(config["worker_benchmark_batches"]),
        pin_memory=pin_memory,
    )
    num_workers = args.num_workers if args.num_workers is not None else select_worker_count(worker_results)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
        shuffle=True,
        seed=int(config["seed"]),
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
        shuffle=False,
        seed=int(config["seed"]),
        pin_memory=pin_memory,
    )

    model = build_traffic_only_stid(NUM_NODES).to(device=device, dtype=torch.float32)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=99, eta_min=1e-5)
    checkpoints = CheckpointManager(checkpoint_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_sampler = NvidiaSmiSampler()
    if device.type == "cuda":
        gpu_sampler.start()

    training_epochs = []
    validation_epochs = []
    best_validation = None
    try:
        for epoch_index in range(epochs):
            training = train_one_epoch(
                model,
                train_loader,
                scaler,
                optimizer,
                device,
                grad_clip=float(config["grad_clip"]),
            )
            validation = evaluate(model, val_loader, scaler, device)
            metadata = {
                "epoch": epoch_index + 1,
                "val_mae": validation.metrics.mae,
                "basicts_commit": BASICTS_COMMIT,
            }
            checkpoints.save_last(model, metadata)
            if checkpoints.maybe_save_best(model, validation.metrics.mae, metadata):
                best_validation = validation
            training_epochs.append(training)
            validation_epochs.append(validation)
            scheduler.step()
            print(
                f"epoch={epoch_index + 1} loss={training.loss:.6f} "
                f"val_mae={validation.metrics.mae:.6f} sec={training.seconds:.3f}",
                flush=True,
            )
    finally:
        gpu_utilization = gpu_sampler.stop()

    if best_validation is None:
        raise RuntimeError("best checkpoint was not created")
    best_metadata = load_model_checkpoint(checkpoints.best_path, model, map_location=device, strict=True)
    reload_validation = evaluate(model, val_loader, scaler, device)
    if not math.isclose(
        reload_validation.metrics.mae,
        float(best_metadata["val_mae"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("strict best-checkpoint reload changed validation MAE")

    loss_decreased = len(training_epochs) >= 2 and training_epochs[-1].loss < training_epochs[0].loss
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    report = {
        "phase": 0,
        "baseline": "BasicTS STID",
        "formal_environment": not args.allow_non_target_environment,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "basicts_commit": revision,
        },
        "protocol_window_counts": protocol_counts,
        "config": {
            "batch_size": int(config["batch_size"]),
            "num_workers": num_workers,
            "epochs": epochs,
            "precision": "FP32",
            "amp": False,
            "ddp": False,
            "deterministic": True,
            "traffic_only": True,
        },
        "timer_boundaries": {
            "train_epoch_wall": "CUDA synchronized once at epoch end; includes data and device work",
            "forward_backward_optimizer": "CUDA events, resolved after the epoch-end synchronize",
            "data_wait": "CPU perf_counter around DataLoader next(); no CUDA synchronize",
            "validation_wall": "CUDA synchronized once at validation end",
        },
        "worker_benchmark": [asdict(row) for row in worker_results],
        "training_epochs": [asdict(row) for row in training_epochs],
        "validation_epochs": [asdict(row) for row in validation_epochs],
        "summary": {
            "train_seconds_per_epoch": mean(row.seconds for row in training_epochs),
            "train_milliseconds_per_step": mean(row.milliseconds_per_step for row in training_epochs),
            "train_samples_per_second": mean(row.samples_per_second for row in training_epochs),
            "data_wait_seconds_per_epoch": mean(row.data_wait_seconds for row in training_epochs),
            "validation_seconds": mean(row.seconds for row in validation_epochs),
            "initial_loss": training_epochs[0].loss,
            "final_loss": training_epochs[-1].loss,
            "loss_decreased": loss_decreased,
            "best_val_mae": float(best_metadata["val_mae"]),
            "best_reload_val_mae": reload_validation.metrics.mae,
            "persistence_val_mae": best_validation.persistence.mae,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "gpu_utilization": asdict(gpu_utilization),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}", flush=True)
    if epochs >= 2 and not loss_decreased:
        raise RuntimeError("STID training loss did not decrease")


def _assert_target_environment(device: torch.device) -> None:
    failures = []
    if sys.version_info[:2] != (3, 10):
        failures.append(f"Python is {platform.python_version()}, expected 3.10.x")
    if torch.__version__ != "2.1.2+cu118":
        failures.append(f"PyTorch is {torch.__version__}, expected 2.1.2+cu118")
    cuda_version = torch.version.cuda or ""
    if cuda_version.split(".")[:2] != ["11", "8"]:
        failures.append(f"PyTorch CUDA runtime is {cuda_version or 'unavailable'}, expected 11.8")
    if device.type != "cuda":
        failures.append("CUDA is unavailable")
    else:
        if torch.cuda.device_count() != 1:
            failures.append(f"visible CUDA device count is {torch.cuda.device_count()}, expected 1")
        device_name = torch.cuda.get_device_name(device)
        if "RTX 3090" not in device_name:
            failures.append(f"GPU is {device_name}, expected NVIDIA RTX 3090")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
