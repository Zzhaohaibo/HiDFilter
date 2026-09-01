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

from hidfilter.model import FAMILY_TOP_P_RHO, HiDFilter
from hidfilter.physical import PhysicalGraphContract, load_adjacency_artifact
from hidfilter.protocol.pems08 import (
    NUM_NODES,
    TrafficOnlyForecastingDataset,
    fit_train_scaler,
    move_scaler_to_device,
    validate_pems08_connectivity_adjacency,
    validate_pems08_protocol,
)
from hidfilter.runtime.checkpoint import CheckpointManager, load_model_checkpoint
from hidfilter.runtime.determinism import configure_determinism
from hidfilter.runtime.environment import BASICTS_COMMIT, verify_basicts_revision
from hidfilter.runtime.family_top_p import (
    collect_family_top_p_summary,
    family_top_p_forward,
    phase5_epoch_policy,
)
from hidfilter.runtime.gpu_monitor import NvidiaSmiSampler
from hidfilter.runtime.phase0 import (
    build_dataloader,
    build_optimizer,
    evaluate,
    train_one_epoch,
)
from hidfilter.runtime.physical import prepare_physical_candidates, summarize_physical_candidates
from hidfilter.runtime.semantic import (
    prepare_pems08_semantic_candidates,
    summarize_semantic_candidates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 5 Family Top-p CUDA sanity"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase5_family_top_p_pems08.json",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--adjacency-path", type=Path)
    parser.add_argument("--physical-candidate-cache-path", type=Path)
    parser.add_argument("--semantic-candidate-cache-path", type=Path)
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
    adjacency_path = args.adjacency_path or Path(config["adjacency_path"])
    physical_cache_path = _repository_path(
        args.physical_candidate_cache_path
        or Path(config["physical_candidate_cache_path"])
    )
    semantic_cache_path = _repository_path(
        args.semantic_candidate_cache_path
        or Path(config["semantic_candidate_cache_path"])
    )
    checkpoint_dir = _repository_path(
        args.checkpoint_dir or Path(config["checkpoint_dir"])
    )
    report_path = _repository_path(args.report_path or Path(config["report_path"]))
    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    num_workers = (
        args.num_workers if args.num_workers is not None else int(config["num_workers"])
    )
    if epochs < int(config["family_top_p"]["first_enabled_epoch"]):
        raise ValueError("Phase 5 sanity must run at least 6 epochs")
    if float(config["family_top_p"]["rho"]) != FAMILY_TOP_P_RHO:
        raise ValueError("Phase 5 rho_family must remain 0.8")
    contract = PhysicalGraphContract(**config["graph_contract"])

    configure_determinism(int(config["seed"]))
    revision = verify_basicts_revision(REPOSITORY_ROOT / "third_party" / "BasicTS")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.allow_non_target_environment:
        _assert_target_environment(device)
    protocol_counts = validate_pems08_protocol(dataset_dir)
    adjacency = load_adjacency_artifact(adjacency_path)
    adjacency_evidence = validate_pems08_connectivity_adjacency(adjacency)
    physical_preparation = prepare_physical_candidates(
        adjacency_path,
        physical_cache_path,
        contract,
        kp=int(config["physical_kp"]),
    )
    physical_summary = summarize_physical_candidates(physical_preparation.artifact)

    train_dataset = TrafficOnlyForecastingDataset(
        dataset_dir, "train", memmap=True, expected_num_nodes=NUM_NODES
    )
    semantic_preparation = prepare_pems08_semantic_candidates(
        train_dataset,
        adjacency,
        physical_preparation.artifact.sources,
        semantic_cache_path,
        ks=int(config["semantic_ks"]),
        min_overlap=int(config["semantic_min_overlap"]),
        variance_threshold=float(config["semantic_variance_threshold"]),
    )
    semantic_summary = summarize_semantic_candidates(semantic_preparation.artifact)
    val_dataset = TrafficOnlyForecastingDataset(
        dataset_dir, "val", memmap=True, expected_num_nodes=NUM_NODES
    )
    scaler = move_scaler_to_device(fit_train_scaler(train_dataset), device)
    pin_memory = device.type == "cuda"
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

    model = HiDFilter(
        NUM_NODES,
        physical_candidates=physical_preparation.artifact.candidates,
        semantic_candidates=semantic_preparation.artifact.candidates,
    ).to(device=device, dtype=torch.float32)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["scheduler"]["t_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    checkpoints = CheckpointManager(checkpoint_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_sampler = NvidiaSmiSampler()
    if device.type == "cuda":
        gpu_sampler.start()

    training_results = []
    validation_results = []
    training_epochs = []
    validation_epochs = []
    enabled_per_epoch = []
    best_validation = None
    try:
        for epoch_index in range(epochs):
            epoch_number = epoch_index + 1
            policy = phase5_epoch_policy(epoch_number)
            forward_fn = family_top_p_forward(policy.family_top_p_enabled)
            training = train_one_epoch(
                model,
                train_loader,
                scaler,
                optimizer,
                device,
                grad_clip=float(config["grad_clip"]),
                forward_fn=forward_fn,
            )
            validation = evaluate(
                model,
                val_loader,
                scaler,
                device,
                forward_fn=forward_fn,
            )
            metadata = {
                "epoch": epoch_number,
                "val_mae": validation.metrics.mae,
                "family_top_p_enabled": policy.family_top_p_enabled,
                "rho_family": FAMILY_TOP_P_RHO,
                "basicts_commit": BASICTS_COMMIT,
                "model": "HiDFilter Full Hierarchy",
                "physical_candidate_fingerprint": physical_preparation.artifact.fingerprint,
                "semantic_candidate_fingerprint": semantic_preparation.artifact.fingerprint,
            }
            checkpoints.save_last(model, metadata)
            if policy.best_selection_enabled and checkpoints.maybe_save_best(
                model, validation.metrics.mae, metadata
            ):
                best_validation = validation
            training_results.append(training)
            validation_results.append(validation)
            training_epochs.append(
                {
                    "epoch": epoch_number,
                    "family_top_p_enabled": policy.family_top_p_enabled,
                    **asdict(training),
                }
            )
            validation_epochs.append(
                {
                    "epoch": epoch_number,
                    "family_top_p_enabled": policy.family_top_p_enabled,
                    "best_selection_eligible": policy.best_selection_enabled,
                    "patience_eligible": policy.patience_enabled,
                    **asdict(validation),
                }
            )
            enabled_per_epoch.append(
                {
                    "epoch": epoch_number,
                    "enabled": policy.family_top_p_enabled,
                }
            )
            scheduler.step()
            print(
                f"epoch={epoch_number} family_top_p_enabled={policy.family_top_p_enabled} "
                f"loss={training.loss:.6f} val_mae={validation.metrics.mae:.6f} "
                f"sec={training.seconds:.3f}",
                flush=True,
            )
    finally:
        gpu_utilization = gpu_sampler.stop()

    if best_validation is None:
        raise RuntimeError("best checkpoint was not created after Family Top-p warmup")
    best_metadata = load_model_checkpoint(
        checkpoints.best_path, model, map_location=device, strict=True
    )
    best_epoch = int(best_metadata["epoch"])
    if best_metadata.get("family_top_p_enabled") is not True or best_epoch < 6:
        raise RuntimeError("best checkpoint does not use the final Family Top-p rule")
    if epochs == 6 and best_epoch != 6:
        raise RuntimeError("six-epoch Phase 5 sanity best checkpoint must be epoch 6")
    reload_validation = evaluate(
        model,
        val_loader,
        scaler,
        device,
        forward_fn=family_top_p_forward(True),
    )
    if not math.isclose(
        reload_validation.metrics.mae,
        float(best_metadata["val_mae"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("strict best-checkpoint reload changed validation MAE")
    family_summary = collect_family_top_p_summary(model, val_loader, scaler, device)

    loss_decreased = (
        len(training_results) >= 2
        and training_results[-1].loss < training_results[0].loss
    )
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    family_order = family_summary.family_order
    report = {
        "phase": 5,
        "model": "HiDFilter Full Hierarchy",
        "formal_environment": not args.allow_non_target_environment,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "basicts_commit": revision,
        },
        "protocol_window_counts": protocol_counts,
        "graph": {
            "artifact_path": str(adjacency_path),
            "artifact_identifier": adjacency_path.name,
            **contract.as_dict(),
            **adjacency_evidence,
            "candidate_fingerprint": physical_preparation.artifact.fingerprint,
            "candidate_cache_path": str(physical_cache_path),
            "candidate_cache_hit": physical_preparation.cache_hit,
            "candidate_build_seconds": physical_preparation.seconds,
            **physical_summary,
        },
        "semantic": {
            "candidate_fingerprint": semantic_preparation.artifact.fingerprint,
            "candidate_cache_path": str(semantic_cache_path),
            "candidate_cache_hit": semantic_preparation.cache_hit,
            "candidate_build_seconds": semantic_preparation.seconds,
            "self_exclusion": True,
            "one_hop_physical_exclusion": True,
            "selected_physical_exclusion": True,
            **semantic_summary,
        },
        "family_evidence": {
            "source": "identity-free lag_content",
            "lag_aggregation": "FP32 mean over 12 lags",
            "source_aggregation": "unweighted masked mean over unique source slots",
            "weighting": "none",
            "unavailable_behavior": "exact zero",
        },
        "router": {
            "input_dim": int(config["router"]["input_dim"]),
            "hidden_dim": int(config["router"]["hidden_dim"]),
            "family_order": list(family_order),
            "shared_scorer": True,
            "route_family_embedding_separate_from_fine": True,
            "availability_masked_softmax": True,
            "dense_probability_summary": {
                "overall_mean": dict(
                    zip(family_order, family_summary.dense_overall_mean)
                ),
                "minimum": dict(zip(family_order, family_summary.dense_minimum)),
                "maximum": dict(zip(family_order, family_summary.dense_maximum)),
                "positions": family_summary.positions,
            },
        },
        "family_top_p": {
            "rho": FAMILY_TOP_P_RHO,
            "input": "availability-masked dense Router probability [B,N,H,3]",
            "warmup_epochs": 5,
            "first_enabled_epoch": 6,
            "canonical_order": list(family_order),
            "detached_support": True,
            "differentiable_retained_weights": True,
            "crossing_item_retained": True,
            "exact_renormalization": True,
            "unavailable_keep_false_and_weight_zero": True,
            "implementation_reuse": "shared deterministic edge_top_p core",
            "top_p_enabled_per_epoch": enabled_per_epoch,
            "retained_weight_overall_mean": dict(
                zip(family_order, family_summary.retained_weight_overall_mean)
            ),
            "retained_family_count": {
                "mean": family_summary.retained_family_count_mean,
                "minimum": family_summary.retained_family_count_min,
                "maximum": family_summary.retained_family_count_max,
            },
            "summary_seconds": family_summary.seconds,
        },
        "warmup_checkpoint_policy": {
            "best_selection_started_epoch": 6,
            "patience_accounting_started_epoch": 6,
            "warmup_validation_logged": True,
            "warmup_participates_in_best_selection": False,
            "warmup_participates_in_patience": False,
            "best_epoch": best_epoch,
        },
        "config": {
            "batch_size": int(config["batch_size"]),
            "num_workers": num_workers,
            "epochs": epochs,
            "precision": "FP32",
            "amp": False,
            "ddp": False,
            "deterministic": True,
            "traffic_only": True,
            "edge_top_p_rho": 0.8,
            "family_top_p_rho": FAMILY_TOP_P_RHO,
            "dependency_spaces": ["Self", "Physical", "Semantic"],
            "family_fusion": "Family Top-p retained Router weight",
            "formal_training_compatibility": config[
                "formal_training_compatibility"
            ],
        },
        "timer_boundaries": {
            "train_epoch_wall": "CUDA synchronized once at epoch end; includes data and device work",
            "forward_backward_optimizer": "CUDA events, resolved after the epoch-end synchronize",
            "data_wait": "CPU perf_counter around DataLoader next(); no CUDA synchronize",
            "validation_wall": "CUDA synchronized once at validation end",
            "family_top_p_summary": "separate post-reload inference reduction; one final device transfer",
        },
        "training_epochs": training_epochs,
        "validation_epochs": validation_epochs,
        "summary": {
            "train_seconds_per_epoch": mean(row.seconds for row in training_results),
            "train_milliseconds_per_step": mean(
                row.milliseconds_per_step for row in training_results
            ),
            "train_samples_per_second": mean(
                row.samples_per_second for row in training_results
            ),
            "forward_seconds_per_epoch": mean(
                row.forward_seconds for row in training_results
            ),
            "backward_seconds_per_epoch": mean(
                row.backward_seconds for row in training_results
            ),
            "optimizer_seconds_per_epoch": mean(
                row.optimizer_seconds for row in training_results
            ),
            "data_wait_seconds_per_epoch": mean(
                row.data_wait_seconds for row in training_results
            ),
            "validation_seconds": mean(row.seconds for row in validation_results),
            "initial_loss": training_results[0].loss,
            "final_loss": training_results[-1].loss,
            "loss_decreased": loss_decreased,
            "best_selection_started_epoch": 6,
            "best_epoch": best_epoch,
            "best_val_mae": float(best_metadata["val_mae"]),
            "best_reload_val_mae": reload_validation.metrics.mae,
            "persistence_val_mae": best_validation.persistence.mae,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "gpu_utilization": asdict(gpu_utilization),
            "physical_candidate_build_seconds": physical_preparation.seconds,
            "semantic_candidate_build_seconds": semantic_preparation.seconds,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}", flush=True)
    if epochs >= 2 and not loss_decreased:
        raise RuntimeError("Phase 5 training loss did not decrease")


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _assert_target_environment(device: torch.device) -> None:
    failures = []
    if sys.version_info[:2] != (3, 10):
        failures.append(f"Python is {platform.python_version()}, expected 3.10.x")
    if torch.__version__ != "2.1.2+cu118":
        failures.append(f"PyTorch is {torch.__version__}, expected 2.1.2+cu118")
    cuda_version = torch.version.cuda or ""
    if cuda_version.split(".")[:2] != ["11", "8"]:
        failures.append(
            f"PyTorch CUDA runtime is {cuda_version or 'unavailable'}, expected 11.8"
        )
    if device.type != "cuda":
        failures.append("CUDA is unavailable")
    else:
        if torch.cuda.device_count() != 1:
            failures.append(
                f"visible CUDA device count is {torch.cuda.device_count()}, expected 1"
            )
        device_name = torch.cuda.get_device_name(device)
        if "RTX 3090" not in device_name:
            failures.append(f"GPU is {device_name}, expected NVIDIA RTX 3090")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
