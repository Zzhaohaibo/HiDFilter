from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import sys
from typing import Callable

import torch
from torch import nn


BASE_COMMIT = "cb3db5e1998d12316502d11823d91fed7e7f6dcc"
DIAGNOSIS_VARIANTS = (
    "no_family_top_p",
    "no_edge_top_p",
    "no_top_p",
)
FINE_FAMILY_GRADIENT_TOLERANCE = 1.0e-8
_FROZEN_DIAGNOSIS_CONFIG = {
    "phase": 6,
    "formal_config_version": 1,
    "model": "HiDFilter Full Hierarchy",
    "max_epochs": 100,
    "batch_size": 64,
    "num_workers": 0,
    "precision": "fp32",
    "amp": False,
    "ddp": False,
    "training_resume": False,
    "graph_contract": {
        "graph_mode": "undirected",
        "weight_semantics": "affinity",
        "conversion_scale": None,
    },
    "physical_kp": 8,
    "semantic_ks": 8,
    "semantic_min_overlap": 288,
    "semantic_variance_threshold": 1.0e-12,
    "rho_edge": 0.8,
    "rho_family": 0.8,
    "family_top_p_warmup_epochs": 5,
    "optimizer": {
        "name": "AdamW",
        "lr": 1.0e-3,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "matrix_weight_decay": 1.0e-4,
        "other_weight_decay": 0.0,
        "grad_clip": 5.0,
    },
    "scheduler": {
        "name": "frozen_cosine",
        "epoch_index_start": 0,
        "epoch_index_end": 99,
        "eta_max": 1.0e-3,
        "eta_min": 1.0e-5,
    },
    "early_stopping": {
        "first_eligible_epoch": 6,
        "patience": 15,
        "min_delta": 0.0,
    },
}


@dataclass(frozen=True)
class DiagnosisEpochPolicy:
    variant: str
    epoch_number: int
    family_top_p_enabled: bool
    edge_top_p_enabled: bool
    best_selection_enabled: bool
    patience_enabled: bool


def validate_diagnosis_config(config: dict[str, object]) -> None:
    """Reject drift from the frozen same-stack training semantics."""

    for name, expected in _FROZEN_DIAGNOSIS_CONFIG.items():
        actual = config.get(name, "<missing>")
        if _canonical_json(actual) != _canonical_json(expected):
            raise ValueError(
                f"frozen diagnosis config mismatch for {name}: "
                f"{actual!r} != {expected!r}"
            )


def validate_hidfilter_diagnosis_epochs(max_epochs: int) -> None:
    if not 6 <= max_epochs <= 100:
        raise ValueError("HiDFilter diagnosis max_epochs must be in 6..100")


def validate_stid_diagnosis_epochs(max_epochs: int) -> None:
    if not 1 <= max_epochs <= 100:
        raise ValueError("STID diagnosis max_epochs must be in 1..100")


def assert_target_environment(device: torch.device) -> None:
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


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def diagnosis_epoch_policy(variant: str, epoch_number: int) -> DiagnosisEpochPolicy:
    """Return the two support switches without changing the shared lifecycle."""

    if variant not in DIAGNOSIS_VARIANTS:
        raise ValueError(f"unknown performance diagnosis variant: {variant}")
    if epoch_number <= 0:
        raise ValueError("epoch_number must be a positive 1-based epoch")
    eligible = epoch_number >= 6
    family_top_p_enabled = variant == "no_edge_top_p" and eligible
    edge_top_p_enabled = variant == "no_family_top_p"
    return DiagnosisEpochPolicy(
        variant=variant,
        epoch_number=epoch_number,
        family_top_p_enabled=family_top_p_enabled,
        edge_top_p_enabled=edge_top_p_enabled,
        best_selection_enabled=eligible,
        patience_enabled=eligible,
    )


def diagnosis_forward(
    policy: DiagnosisEpochPolicy,
) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    """Bind one diagnosis epoch to the existing model forward path."""

    def forward(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
        return model(
            history,
            family_top_p_enabled=policy.family_top_p_enabled,
            edge_top_p_enabled=policy.edge_top_p_enabled,
        )

    return forward


def gradient_group_l2_norms(model: nn.Module) -> dict[str, float]:
    """Collect one post-backward audit with a single grouped host transfer."""

    groups: tuple[tuple[str, nn.Module | nn.Parameter], ...] = (
        ("context_encoder", model.context_encoder),
        ("lag_content_encoder", model.lag_content_encoder),
        ("node_embedding", model.node_embedding),
        ("horizon_embedding", model.horizon_embedding),
        ("fine_family_embedding", model.fine_family_embedding),
        ("route_family_embedding", model.route_family_embedding),
        ("wq", model.wq),
        ("wk", model.wk),
        ("wv", model.wv),
        ("router_scorer", model.router_scorer),
        ("decoder", model.decoder),
        ("alpha_raw", model.alpha_raw),
    )
    device = next(model.parameters()).device
    norms = []
    for _, group in groups:
        parameters = (group,) if isinstance(group, nn.Parameter) else group.parameters()
        squared = torch.zeros((), dtype=torch.float64, device=device)
        for parameter in parameters:
            if parameter.grad is not None:
                squared += parameter.grad.detach().to(torch.float64).square().sum()
        norms.append(squared.sqrt())
    reduced = torch.stack(norms).cpu().tolist()
    return {name: float(value) for (name, _), value in zip(groups, reduced)}
