from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from hidfilter.runtime.formal import WarmupEarlyStopping


ROOT = Path(__file__).resolve().parents[2]
HIDFILTER_RUNNER = ROOT / "scripts" / "run_performance_diagnosis.py"
STID_RUNNER = ROOT / "scripts" / "run_stid_performance_diagnosis.py"
GRADIENT_AUDIT = ROOT / "scripts" / "audit_hidfilter_gradients.py"
FORMAL_CONFIG = ROOT / "configs" / "formal_pems08.json"


def _diagnosis_module():
    if importlib.util.find_spec("hidfilter.runtime.performance_diagnosis") is None:
        pytest.fail("performance diagnosis runtime helpers are not implemented")
    return importlib.import_module("hidfilter.runtime.performance_diagnosis")


@pytest.mark.parametrize(
    ("variant", "epoch", "family_top_p", "edge_top_p", "best_eligible"),
    [
        ("no_family_top_p", 1, False, True, False),
        ("no_family_top_p", 6, False, True, True),
        ("no_edge_top_p", 1, False, False, False),
        ("no_edge_top_p", 6, True, False, True),
        ("no_top_p", 1, False, False, False),
        ("no_top_p", 6, False, False, True),
    ],
)
def test_diagnosis_epoch_policy_isolates_requested_operator(
    variant: str,
    epoch: int,
    family_top_p: bool,
    edge_top_p: bool,
    best_eligible: bool,
) -> None:
    policy = _diagnosis_module().diagnosis_epoch_policy(variant, epoch)

    assert policy.family_top_p_enabled is family_top_p
    assert policy.edge_top_p_enabled is edge_top_p
    assert policy.best_selection_enabled is best_eligible
    assert policy.patience_enabled is best_eligible


def test_all_hidfilter_variants_keep_best_selection_at_epoch_six() -> None:
    for variant in ("no_family_top_p", "no_edge_top_p", "no_top_p"):
        stopping = WarmupEarlyStopping(
            patience=15,
            min_delta=0.0,
            first_eligible_epoch=6,
        )
        for epoch in range(1, 6):
            assert not stopping.observe(epoch, 20.0 - epoch).eligible
        assert stopping.observe(6, 14.0).improved
        assert stopping.best_epoch == 6


def test_diagnosis_forward_binds_both_support_switches() -> None:
    module = _diagnosis_module()
    policy = module.diagnosis_epoch_policy("no_edge_top_p", 6)

    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flags: tuple[bool, bool] | None = None

        def forward(
            self,
            history: torch.Tensor,
            *,
            family_top_p_enabled: bool,
            edge_top_p_enabled: bool,
        ) -> torch.Tensor:
            self.flags = (family_top_p_enabled, edge_top_p_enabled)
            return history

    model = RecordingModel()
    history = torch.ones(1)

    assert module.diagnosis_forward(policy)(model, history) is history
    assert model.flags == (True, False)


def test_diagnosis_config_reuses_and_locks_frozen_formal_semantics() -> None:
    module = _diagnosis_module()
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))

    module.validate_diagnosis_config(config)
    for path, value in (
        (("batch_size",), 32),
        (("optimizer", "betas"), [0.8, 0.999]),
        (("scheduler", "eta_min"), 2.0e-5),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="frozen diagnosis config"):
            module.validate_diagnosis_config(mutated)


def test_diagnosis_epoch_limits_keep_long_run_protocol_and_short_smoke() -> None:
    module = _diagnosis_module()

    module.validate_hidfilter_diagnosis_epochs(6)
    module.validate_hidfilter_diagnosis_epochs(100)
    module.validate_stid_diagnosis_epochs(1)
    module.validate_stid_diagnosis_epochs(100)
    with pytest.raises(ValueError, match="6..100"):
        module.validate_hidfilter_diagnosis_epochs(5)
    with pytest.raises(ValueError, match="1..100"):
        module.validate_stid_diagnosis_epochs(0)


def test_target_environment_guard_fails_clearly_off_target() -> None:
    module = _diagnosis_module()

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        module.assert_target_environment(torch.device("cpu"))


def test_stid_development_lifecycle_selects_from_epoch_one() -> None:
    stopping = WarmupEarlyStopping(
        patience=15,
        min_delta=0.0,
        first_eligible_epoch=1,
    )

    first = stopping.observe(1, 20.0)

    assert first.eligible
    assert first.improved
    assert stopping.best_epoch == 1


def test_gradient_group_norms_cover_frozen_parameter_groups() -> None:
    module = _diagnosis_module()

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = nn.Linear(2, 2)
            self.lag_content_encoder = nn.Linear(2, 2)
            self.node_embedding = nn.Embedding(2, 2)
            self.horizon_embedding = nn.Embedding(2, 2)
            self.fine_family_embedding = nn.Embedding(3, 2)
            self.route_family_embedding = nn.Embedding(3, 2)
            self.wq = nn.Linear(2, 2)
            self.wk = nn.Linear(2, 2)
            self.wv = nn.Linear(2, 2)
            self.router_scorer = nn.Linear(2, 1)
            self.decoder = nn.Linear(2, 1)
            self.alpha_raw = nn.Parameter(torch.tensor(1.0))

    model = TinyModel()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    norms = module.gradient_group_l2_norms(model)

    assert tuple(norms) == (
        "context_encoder",
        "lag_content_encoder",
        "node_embedding",
        "horizon_embedding",
        "fine_family_embedding",
        "route_family_embedding",
        "wq",
        "wk",
        "wv",
        "router_scorer",
        "decoder",
        "alpha_raw",
    )
    assert all(value > 0.0 for value in norms.values())


def test_development_only_entrypoints_exist_and_contain_no_test_evaluation() -> None:
    for path in (HIDFILTER_RUNNER, STID_RUNNER, GRADIENT_AUDIT):
        assert path.is_file(), f"missing entrypoint: {path.name}"

    for path in (HIDFILTER_RUNNER, STID_RUNNER):
        source = path.read_text(encoding="utf-8")
        assert '"mode": "development"' in source
        assert '"test_metrics": None' in source
        assert '"test_evaluated": False' in source
        assert '"test"' not in source


@pytest.mark.parametrize(
    ("runner", "required_args"),
    [
        (HIDFILTER_RUNNER, ("--variant", "no_top_p")),
        (STID_RUNNER, ()),
    ],
)
def test_development_runners_hard_reject_final_mode(
    runner: Path,
    required_args: tuple[str, ...],
) -> None:
    environment = os.environ.copy()
    python_paths = (ROOT / "src", ROOT / "third_party" / "BasicTS" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in python_paths
    )
    result = subprocess.run(
        [sys.executable, str(runner), *required_args, "--mode", "final"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --mode final" in result.stderr
