#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Sequence

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
from hidfilter.runtime.formal import (
    FORMAL_CONFIG_VERSION,
    WarmupEarlyStopping,
    formal_config_fingerprint,
    set_formal_learning_rate,
)
from hidfilter.runtime.gpu_monitor import NvidiaSmiSampler
from hidfilter.runtime.performance_diagnosis import (
    BASE_COMMIT,
    DIAGNOSIS_VARIANTS,
    assert_target_environment,
    diagnosis_epoch_policy,
    diagnosis_forward,
    validate_diagnosis_config,
    validate_hidfilter_diagnosis_epochs,
)
from hidfilter.runtime.phase0 import (
    build_dataloader,
    build_optimizer,
    evaluate,
    train_one_epoch,
)
from hidfilter.runtime.physical import (
    prepare_physical_candidates,
    summarize_physical_candidates,
)
from hidfilter.runtime.semantic import (
    prepare_pems08_semantic_candidates,
    summarize_semantic_candidates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one development-only HiDFilter filtration diagnosis"
    )
    parser.add_argument("--variant", choices=DIAGNOSIS_VARIANTS, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "formal_pems08.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--adjacency-path", type=Path)
    parser.add_argument("--physical-candidate-cache-path", type=Path)
    parser.add_argument("--semantic-candidate-cache-path", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument(
        "--allow-non-target-environment",
        action="store_true",
        help="Local protocol smoke only; bypass the frozen AutoDL checks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_diagnosis_config(config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    max_epochs = (
        args.max_epochs if args.max_epochs is not None else int(config["max_epochs"])
    )
    validate_hidfilter_diagnosis_epochs(max_epochs)
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
    run_name = f"{args.variant}_seed_{seed}"
    checkpoint_dir = _repository_path(
        args.checkpoint_dir
        or Path("checkpoints") / "performance_diagnosis" / run_name
    )
    report_path = _repository_path(
        args.report_path
        or Path("reports") / "performance_diagnosis" / f"{run_name}.json"
    )
    contract = PhysicalGraphContract(**config["graph_contract"])

    configure_determinism(seed)
    basicts_revision = verify_basicts_revision(
        REPOSITORY_ROOT / "third_party" / "BasicTS"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.allow_non_target_environment:
        assert_target_environment(device)
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
    num_workers = int(config["num_workers"])
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
    variant_policy = _variant_policy_description(args.variant)
    diagnosis_contract = {
        "task": "performance_diagnosis_gate1",
        "mode": "development",
        "diagnostic_only": True,
        "variant": args.variant,
        "base_commit": BASE_COMMIT,
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
        "variant_policy": variant_policy,
        "rho_edge": EDGE_TOP_P_RHO,
        "rho_family": FAMILY_TOP_P_RHO,
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
    config_fingerprint = formal_config_fingerprint(diagnosis_contract)
    checkpoints = CheckpointManager(checkpoint_dir)
    early_stopping = WarmupEarlyStopping(
        patience=int(config["early_stopping"]["patience"]),
        min_delta=float(config["early_stopping"]["min_delta"]),
        first_eligible_epoch=6,
    )
    training_history: list[dict[str, object]] = []
    validation_history: list[dict[str, object]] = []
    training_results = []
    validation_results = []
    gpu_sampler = NvidiaSmiSampler()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_sampler.start()
    try:
        for epoch_number in range(1, max_epochs + 1):
            learning_rate = set_formal_learning_rate(optimizer, epoch_number)
            policy = diagnosis_epoch_policy(args.variant, epoch_number)
            forward_fn = diagnosis_forward(policy)
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
            if decision.eligible is not policy.best_selection_enabled:
                raise RuntimeError("variant policy and best-selection lifecycle disagree")
            metadata = {
                "variant": args.variant,
                "seed": seed,
                "epoch": epoch_number,
                "val_metrics": asdict(validation.metrics),
                "val_mae": validation.metrics.mae,
                "family_top_p_enabled": policy.family_top_p_enabled,
                "edge_top_p_enabled": policy.edge_top_p_enabled,
                "rho_family": FAMILY_TOP_P_RHO,
                "rho_edge": EDGE_TOP_P_RHO,
                "physical_candidate_fingerprint": physical.artifact.fingerprint,
                "semantic_candidate_fingerprint": semantic.artifact.fingerprint,
                "basicts_commit": BASICTS_COMMIT,
                "formal_config_version": FORMAL_CONFIG_VERSION,
                "diagnosis_config_fingerprint": config_fingerprint,
            }
            checkpoints.save_last(model, metadata)
            if decision.improved:
                if not checkpoints.maybe_save_best(
                    model, validation.metrics.mae, metadata
                ):
                    raise RuntimeError(
                        "early-stopping and best-checkpoint states disagree"
                    )
            training_results.append(training)
            validation_results.append(validation)
            training_history.append(
                {
                    "epoch": epoch_number,
                    "learning_rate": learning_rate,
                    "family_top_p_enabled": policy.family_top_p_enabled,
                    "edge_top_p_enabled": policy.edge_top_p_enabled,
                    **asdict(training),
                }
            )
            validation_history.append(
                {
                    "epoch": epoch_number,
                    "family_top_p_enabled": policy.family_top_p_enabled,
                    "edge_top_p_enabled": policy.edge_top_p_enabled,
                    "best_selection_eligible": decision.eligible,
                    "improved": decision.improved,
                    "non_improving_epochs": decision.non_improving_epochs,
                    **asdict(validation),
                }
            )
            print(
                f"variant={args.variant} epoch={epoch_number} "
                f"lr={learning_rate:.10f} "
                f"family_top_p_enabled={policy.family_top_p_enabled} "
                f"edge_top_p_enabled={policy.edge_top_p_enabled} "
                f"loss={training.loss:.6f} val_mae={validation.metrics.mae:.6f} "
                f"patience={decision.non_improving_epochs}/{early_stopping.patience}",
                flush=True,
            )
            if decision.should_stop:
                break
    finally:
        gpu_utilization = gpu_sampler.stop()

    if early_stopping.best_epoch is None or not checkpoints.best_path.is_file():
        raise RuntimeError("eligible best checkpoint was not created")
    final_alpha = float(model.alpha.detach().cpu())
    best_metadata = load_model_checkpoint(
        checkpoints.best_path, model, map_location=device, strict=True
    )
    if best_metadata.get("diagnosis_config_fingerprint") != config_fingerprint:
        raise RuntimeError("diagnosis best checkpoint config fingerprint mismatch")
    best_epoch = int(best_metadata["epoch"])
    best_policy = diagnosis_epoch_policy(args.variant, best_epoch)
    if best_metadata.get("family_top_p_enabled") is not best_policy.family_top_p_enabled:
        raise RuntimeError("best checkpoint Family Top-p policy mismatch")
    if best_metadata.get("edge_top_p_enabled") is not best_policy.edge_top_p_enabled:
        raise RuntimeError("best checkpoint Edge Top-p policy mismatch")
    reload_validation = evaluate(
        model,
        val_loader,
        scaler,
        device,
        forward_fn=diagnosis_forward(best_policy),
    )
    _assert_reload_metrics(best_metadata["val_metrics"], reload_validation.metrics)
    diagnostic_evaluation = evaluate_with_diagnostics(
        model,
        val_loader,
        scaler,
        device,
        family_top_p_enabled=best_policy.family_top_p_enabled,
        edge_top_p_enabled=best_policy.edge_top_p_enabled,
    )
    _validate_variant_diagnostics(args.variant, diagnostic_evaluation.diagnostics)

    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    report = {
        "task": "performance_diagnosis_gate1",
        "mode": "development",
        "diagnostic_only": True,
        "variant": args.variant,
        "base_commit": BASE_COMMIT,
        "run_manifest": {
            "repository_commit": _git_revision(),
            "seed": seed,
            "diagnosis_config_fingerprint": config_fingerprint,
            "basicts_commit": basicts_revision,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "target_environment": not args.allow_non_target_environment,
        },
        "training_protocol": diagnosis_contract,
        "variant_policy": variant_policy,
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
            "best_epoch": best_epoch,
            "best_validation_metrics": asdict(reload_validation.metrics),
            "saved_best_validation_metrics": best_metadata["val_metrics"],
            "strict_reload_equal": True,
            "stopped_epoch": training_history[-1]["epoch"],
            "early_stopped": training_history[-1]["epoch"] < max_epochs,
        },
        "alpha": {
            "final_before_reload": final_alpha,
            "best": float(model.alpha.detach().cpu()),
        },
        "test_metrics": None,
        "test_evaluated": False,
        "diagnostics_split": "validation",
        "diagnostics": diagnostic_evaluation.diagnostics,
        "runtime": {
            "diagnostic_seconds": diagnostic_evaluation.seconds,
            "diagnostic_samples_per_second": diagnostic_evaluation.samples_per_second,
            "mean_train_seconds_per_epoch": mean(
                row.seconds for row in training_results
            ),
            "mean_train_milliseconds_per_step": mean(
                row.milliseconds_per_step for row in training_results
            ),
            "mean_train_samples_per_second": mean(
                row.samples_per_second for row in training_results
            ),
            "mean_forward_seconds_per_epoch": mean(
                row.forward_seconds for row in training_results
            ),
            "mean_backward_seconds_per_epoch": mean(
                row.backward_seconds for row in training_results
            ),
            "mean_optimizer_seconds_per_epoch": mean(
                row.optimizer_seconds for row in training_results
            ),
            "mean_data_wait_seconds_per_epoch": mean(
                row.data_wait_seconds for row in training_results
            ),
            "mean_validation_seconds": mean(
                row.seconds for row in validation_results
            ),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "gpu_utilization": asdict(gpu_utilization),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}", flush=True)


def _variant_policy_description(variant: str) -> dict[str, object]:
    if variant == "no_family_top_p":
        return {
            "edge_top_p_enabled": "always",
            "family_top_p_enabled": "never",
            "best_selection_first_epoch": 6,
        }
    if variant == "no_edge_top_p":
        return {
            "edge_top_p_enabled": "never",
            "family_top_p_enabled": "epochs_6_plus",
            "best_selection_first_epoch": 6,
        }
    if variant == "no_top_p":
        return {
            "edge_top_p_enabled": "never",
            "family_top_p_enabled": "never",
            "best_selection_first_epoch": 6,
        }
    raise ValueError(f"unknown performance diagnosis variant: {variant}")


def _assert_reload_metrics(saved: dict[str, object], reloaded: object) -> None:
    for name in ("mae", "rmse", "mape"):
        if not math.isclose(
            float(saved[name]),
            float(getattr(reloaded, name)),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(f"strict best reload changed validation {name.upper()}")


def _validate_variant_diagnostics(
    variant: str, diagnostics: dict[str, object]
) -> None:
    family_count = diagnostics["family"]["active_family_count"]
    fine = diagnostics["fine"]
    if variant in {"no_family_top_p", "no_top_p"}:
        if family_count["minimum"] != 3 or family_count["maximum"] != 3:
            raise RuntimeError("Family Top-p OFF must retain all three PEMS08 families")
    if variant in {"no_edge_top_p", "no_top_p"}:
        for family_name, expected in (("Self", 12), ("Physical", 96), ("Semantic", 96)):
            count = fine[family_name]["edge_retained_count"]
            if count["minimum"] != expected or count["maximum"] != expected:
                raise RuntimeError(
                    f"Edge Top-p OFF must retain {expected} valid {family_name} candidates"
                )
    if variant == "no_top_p":
        overall = fine["overall"]["effective_edge_count"]
        if overall["minimum"] != 204 or overall["maximum"] != 204:
            raise RuntimeError("no_top_p must retain exactly 204 effective candidates")


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


if __name__ == "__main__":
    main()
