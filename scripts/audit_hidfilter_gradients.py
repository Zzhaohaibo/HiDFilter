#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Sequence

import torch
from basicts.metrics import masked_mae

from hidfilter.model import HiDFilter
from hidfilter.physical import PhysicalGraphContract, load_adjacency_artifact
from hidfilter.protocol.pems08 import (
    NUM_NODES,
    TrafficOnlyForecastingDataset,
    fit_train_scaler,
    move_scaler_to_device,
    prepare_batch,
    validate_pems08_connectivity_adjacency,
    validate_pems08_protocol,
)
from hidfilter.runtime.determinism import configure_determinism
from hidfilter.runtime.environment import verify_basicts_revision
from hidfilter.runtime.performance_diagnosis import (
    BASE_COMMIT,
    FINE_FAMILY_GRADIENT_TOLERANCE,
    assert_target_environment,
    gradient_group_l2_norms,
    validate_diagnosis_config,
)
from hidfilter.runtime.phase0 import build_dataloader
from hidfilter.runtime.physical import prepare_physical_candidates
from hidfilter.runtime.semantic import prepare_pems08_semantic_candidates


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one HiDFilter training-batch gradient path"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "formal_pems08.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--adjacency-path", type=Path)
    parser.add_argument("--physical-candidate-cache-path", type=Path)
    parser.add_argument("--semantic-candidate-cache-path", type=Path)
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
    report_path = _repository_path(
        args.report_path
        or Path("reports") / "performance_diagnosis" / f"gradient_audit_seed_{seed}.json"
    )

    configure_determinism(seed)
    basicts_revision = verify_basicts_revision(
        REPOSITORY_ROOT / "third_party" / "BasicTS"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.allow_non_target_environment:
        assert_target_environment(device)
    protocol_counts = validate_pems08_protocol(dataset_dir)
    adjacency = load_adjacency_artifact(adjacency_path)
    validate_pems08_connectivity_adjacency(adjacency)
    contract = PhysicalGraphContract(**config["graph_contract"])
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
    scaler = move_scaler_to_device(fit_train_scaler(train_dataset), device)
    loader = build_dataloader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    cpu_batch = next(iter(loader))
    raw_batch = {
        "inputs": cpu_batch["inputs"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "targets": cpu_batch["targets"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
    }
    prepared = prepare_batch(raw_batch, scaler)
    model = HiDFilter(
        NUM_NODES,
        physical_candidates=physical.artifact.candidates,
        semantic_candidates=semantic.artifact.candidates,
    ).to(device=device, dtype=torch.float32)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(
        prepared.inputs,
        family_top_p_enabled=True,
        edge_top_p_enabled=True,
    )
    loss = masked_mae(prediction, prepared.targets, prepared.targets_valid)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gradient_norms = gradient_group_l2_norms(model)
    if not math.isfinite(float(loss.detach().cpu())):
        raise FloatingPointError("gradient audit produced a non-finite loss")
    if any(not math.isfinite(value) for value in gradient_norms.values()):
        raise FloatingPointError("gradient audit produced a non-finite gradient norm")
    fine_gradient = gradient_norms["fine_family_embedding"]
    conclusion = (
        "CONFIRMED"
        if fine_gradient <= FINE_FAMILY_GRADIENT_TOLERANCE
        else "NOT CONFIRMED"
    )
    report = {
        "task": "performance_diagnosis_gate1_gradient_audit",
        "mode": "development",
        "diagnostic_only": True,
        "base_commit": BASE_COMMIT,
        "repository_commit": _git_revision(),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "basicts_commit": basicts_revision,
            "target_environment": not args.allow_non_target_environment,
        },
        "protocol_window_counts": protocol_counts,
        "batch_size": int(prepared.inputs.shape[0]),
        "family_top_p_enabled": True,
        "edge_top_p_enabled": True,
        "optimizer_step": False,
        "loss": float(loss.detach().cpu()),
        "gradient_l2_norm": gradient_norms,
        "fine_family_embedding_tolerance": FINE_FAMILY_GRADIENT_TOLERANCE,
        "fine_family_embedding_conclusion": conclusion,
        "test_metrics": None,
        "test_evaluated": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"report={report_path}", flush=True)


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
