#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from hidfilter.model import EDGE_TOP_P_RHO, FAMILY_TOP_P_RHO, HiDFilter
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
from hidfilter.runtime.diagnostics import evaluate_with_diagnostics
from hidfilter.runtime.environment import BASICTS_COMMIT, verify_basicts_revision
from hidfilter.runtime.family_top_p import family_top_p_forward, phase5_epoch_policy
from hidfilter.runtime.formal import (
    FORMAL_CONFIG_VERSION,
    FormalTestOnceGuard,
    WarmupEarlyStopping,
    formal_config_fingerprint,
    set_formal_learning_rate,
)
from hidfilter.runtime.gpu_monitor import NvidiaSmiSampler
from hidfilter.runtime.phase0 import build_dataloader, build_optimizer, evaluate, train_one_epoch
from hidfilter.runtime.physical import prepare_physical_candidates, summarize_physical_candidates
from hidfilter.runtime.semantic import (
    prepare_pems08_semantic_candidates,
    summarize_semantic_candidates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen formal PEMS08 HiDFilter lifecycle"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "formal_pems08.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("development", "final"), default="development")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--adjacency-path", type=Path)
    parser.add_argument("--physical-candidate-cache-path", type=Path)
    parser.add_argument("--semantic-candidate-cache-path", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--allow-non-target-environment",
        action="store_true",
        help="Local protocol diagnostic only; bypass the frozen AutoDL checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    max_epochs = (
        args.max_epochs if args.max_epochs is not None else int(config["max_epochs"])
    )
    if not 6 <= max_epochs <= 100:
        raise ValueError("formal max_epochs must be in 6..100")
    num_workers = (
        args.num_workers if args.num_workers is not None else int(config["num_workers"])
    )
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
    run_name = f"{args.mode}_seed_{seed}"
    checkpoint_dir = _repository_path(
        args.checkpoint_dir or Path(config["checkpoint_root"]) / run_name
    )
    report_path = _repository_path(
        args.report_path or Path(config["report_root"]) / f"{run_name}.json"
    )
    contract = PhysicalGraphContract(**config["graph_contract"])

    configure_determinism(seed)
    basicts_revision = verify_basicts_revision(REPOSITORY_ROOT / "third_party" / "BasicTS")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.allow_non_target_environment:
        _assert_target_environment(device)
    protocol_counts = validate_pems08_protocol(dataset_dir)
    adjacency = load_adjacency_artifact(adjacency_path)
    adjacency_evidence = validate_pems08_connectivity_adjacency(adjacency)
    physical = prepare_physical_candidates(
        adjacency_path,
        physical_cache_path,
        contract,
        kp=int(config["physical_kp"]),
    )
    train_dataset = TrafficOnlyForecastingDataset(
        dataset_dir, "train", memmap=True, expected_num_nodes=NUM_NODES
    )
    semantic = prepare_pems08_semantic_candidates(
        train_dataset,
        adjacency,
        physical.artifact.sources,
        semantic_cache_path,
        ks=int(config["semantic_ks"]),
        min_overlap=int(config["semantic_min_overlap"]),
        variance_threshold=float(config["semantic_variance_threshold"]),
    )
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
        seed=seed,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )

    model = HiDFilter(
        NUM_NODES,
        physical_candidates=physical.artifact.candidates,
        semantic_candidates=semantic.artifact.candidates,
    ).to(device=device, dtype=torch.float32)
    optimizer = build_optimizer(model)
    formal_contract = {
        "formal_config_version": FORMAL_CONFIG_VERSION,
        "phase": 6,
        "model": config["model"],
        "seed": seed,
        "dataset_protocol": {
            "dataset": "PEMS08",
            "split": "official pre-split",
            "traffic_only": True,
            "history_length": 12,
            "forecast_horizon": 12,
            "window_counts": protocol_counts,
            "raw_validity": "finite & ~isclose(raw, 0, atol=5e-5)",
            "scaler": "train-only global scalar Z-score ddof=0",
        },
        "graph_contract": contract.as_dict(),
        "physical_candidate_fingerprint": physical.artifact.fingerprint,
        "semantic_candidate_fingerprint": semantic.artifact.fingerprint,
        "rho_edge": EDGE_TOP_P_RHO,
        "rho_family": FAMILY_TOP_P_RHO,
        "warmup_epochs": int(config["family_top_p_warmup_epochs"]),
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "early_stopping": config["early_stopping"],
        "max_epochs": max_epochs,
        "batch_size": int(config["batch_size"]),
        "num_workers": num_workers,
        "precision": "FP32",
        "amp": False,
        "ddp": False,
        "training_resume": False,
        "basicts_commit": BASICTS_COMMIT,
    }
    config_fingerprint = formal_config_fingerprint(formal_contract)
    checkpoints = CheckpointManager(checkpoint_dir)
    early_stopping = WarmupEarlyStopping(
        patience=int(config["early_stopping"]["patience"]),
        min_delta=float(config["early_stopping"]["min_delta"]),
        first_eligible_epoch=int(config["early_stopping"]["first_eligible_epoch"]),
    )
    guard = FormalTestOnceGuard(mode=args.mode)
    training_history: list[dict[str, object]] = []
    validation_history: list[dict[str, object]] = []
    training_results = []
    gpu_sampler = NvidiaSmiSampler()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_sampler.start()
    try:
        for epoch_number in range(1, max_epochs + 1):
            learning_rate = set_formal_learning_rate(optimizer, epoch_number)
            policy = phase5_epoch_policy(epoch_number)
            forward_fn = family_top_p_forward(policy.family_top_p_enabled)
            training = train_one_epoch(
                model,
                train_loader,
                scaler,
                optimizer,
                device,
                grad_clip=float(config["optimizer"]["grad_clip"]),
                forward_fn=forward_fn,
            )
            validation = evaluate(
                model,
                val_loader,
                scaler,
                device,
                forward_fn=forward_fn,
            )
            decision = early_stopping.observe(epoch_number, validation.metrics.mae)
            metadata = {
                "seed": seed,
                "epoch": epoch_number,
                "val_mae": validation.metrics.mae,
                "rho_family": FAMILY_TOP_P_RHO,
                "rho_edge": EDGE_TOP_P_RHO,
                "family_top_p_enabled": policy.family_top_p_enabled,
                "physical_candidate_fingerprint": physical.artifact.fingerprint,
                "semantic_candidate_fingerprint": semantic.artifact.fingerprint,
                "basicts_commit": BASICTS_COMMIT,
                "formal_config_version": FORMAL_CONFIG_VERSION,
                "formal_config_fingerprint": config_fingerprint,
            }
            checkpoints.save_last(model, metadata)
            if decision.improved:
                if not checkpoints.maybe_save_best(
                    model, validation.metrics.mae, metadata
                ):
                    raise RuntimeError("early-stopping and best-checkpoint states disagree")
            training_results.append(training)
            training_history.append(
                {
                    "epoch": epoch_number,
                    "learning_rate": learning_rate,
                    "family_top_p_enabled": policy.family_top_p_enabled,
                    **asdict(training),
                }
            )
            validation_history.append(
                {
                    "epoch": epoch_number,
                    "family_top_p_enabled": policy.family_top_p_enabled,
                    "best_selection_eligible": decision.eligible,
                    "improved": decision.improved,
                    "non_improving_epochs": decision.non_improving_epochs,
                    **asdict(validation),
                }
            )
            print(
                f"epoch={epoch_number} lr={learning_rate:.10f} "
                f"family_top_p_enabled={policy.family_top_p_enabled} "
                f"loss={training.loss:.6f} val_mae={validation.metrics.mae:.6f} "
                f"patience={decision.non_improving_epochs}/{early_stopping.patience}",
                flush=True,
            )
            if decision.should_stop:
                break
    finally:
        gpu_utilization = gpu_sampler.stop()

    guard.mark_training_complete()
    if early_stopping.best_epoch is None or not checkpoints.best_path.is_file():
        raise RuntimeError("eligible best checkpoint was not created")
    best_metadata = load_model_checkpoint(
        checkpoints.best_path, model, map_location=device, strict=True
    )
    if best_metadata.get("family_top_p_enabled") is not True:
        raise RuntimeError("formal best checkpoint must have Family Top-p enabled")
    if best_metadata.get("formal_config_fingerprint") != config_fingerprint:
        raise RuntimeError("formal best checkpoint config fingerprint mismatch")
    guard.mark_best_reloaded()
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
        raise RuntimeError("strict best reload changed validation MAE")

    test_evaluation = None
    if guard.should_run_test:
        guard.begin_test()
        test_dataset = TrafficOnlyForecastingDataset(
            dataset_dir, "test", memmap=True, expected_num_nodes=NUM_NODES
        )
        test_loader = build_dataloader(
            test_dataset,
            batch_size=int(config["batch_size"]),
            num_workers=num_workers,
            shuffle=False,
            seed=seed,
            pin_memory=pin_memory,
        )
        diagnostic_evaluation = evaluate_with_diagnostics(
            model, test_loader, scaler, device
        )
        test_evaluation = asdict(diagnostic_evaluation.metrics)
        diagnostic_split = "test"
    else:
        diagnostic_evaluation = evaluate_with_diagnostics(
            model, val_loader, scaler, device
        )
        diagnostic_split = "validation"

    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    report = {
        "phase": 6,
        "mode": args.mode,
        "formal_environment": not args.allow_non_target_environment,
        "run_manifest": {
            "repository_commit": _git_revision(),
            "seed": seed,
            "formal_config_version": FORMAL_CONFIG_VERSION,
            "formal_config_fingerprint": config_fingerprint,
            "basicts_commit": basicts_revision,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        "formal_config": formal_contract,
        "protocol_window_counts": protocol_counts,
        "graph": {
            "artifact_path": str(adjacency_path),
            **contract.as_dict(),
            **adjacency_evidence,
            "candidate_fingerprint": physical.artifact.fingerprint,
            "candidate_cache_hit": physical.cache_hit,
            **summarize_physical_candidates(physical.artifact),
        },
        "semantic": {
            "candidate_fingerprint": semantic.artifact.fingerprint,
            "candidate_cache_hit": semantic.cache_hit,
            **summarize_semantic_candidates(semantic.artifact),
        },
        "training_history": training_history,
        "validation_history": validation_history,
        "selection": {
            "best_epoch": int(best_metadata["epoch"]),
            "best_val_mae": float(best_metadata["val_mae"]),
            "best_reload_val_mae": reload_validation.metrics.mae,
            "stopped_epoch": training_history[-1]["epoch"],
            "early_stopped": training_history[-1]["epoch"] < max_epochs,
        },
        "test_metrics": test_evaluation,
        "test_evaluation_count": int(guard.test_executed),
        "diagnostics_split": diagnostic_split,
        "diagnostics": diagnostic_evaluation.diagnostics,
        "runtime": {
            "diagnostic_seconds": diagnostic_evaluation.seconds,
            "diagnostic_samples_per_second": diagnostic_evaluation.samples_per_second,
            "mean_train_seconds_per_epoch": mean(row.seconds for row in training_results),
            "mean_train_milliseconds_per_step": mean(
                row.milliseconds_per_step for row in training_results
            ),
            "mean_train_samples_per_second": mean(
                row.samples_per_second for row in training_results
            ),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "gpu_utilization": asdict(gpu_utilization),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}", flush=True)


def _validate_config(config: dict[str, object]) -> None:
    if int(config["formal_config_version"]) != FORMAL_CONFIG_VERSION:
        raise ValueError("unsupported formal config version")
    if int(config["max_epochs"]) != 100:
        raise ValueError("formal config max_epochs must remain 100")
    if float(config["rho_edge"]) != EDGE_TOP_P_RHO:
        raise ValueError("rho_edge must remain 0.8")
    if float(config["rho_family"]) != FAMILY_TOP_P_RHO:
        raise ValueError("rho_family must remain 0.8")
    if config["precision"] != "fp32" or config["amp"] or config["ddp"]:
        raise ValueError("formal precision must remain single-process FP32")
    if config["training_resume"] is not False:
        raise ValueError("training resume is not implemented")


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
