from __future__ import annotations

import subprocess

import torch

from hidfilter.runtime.checkpoint import CheckpointManager, load_model_checkpoint
from hidfilter.runtime.determinism import configure_determinism
from hidfilter.runtime.environment import BASICTS_COMMIT, verify_basicts_revision


def test_frozen_basicts_revision():
    assert verify_basicts_revision("third_party/BasicTS") == BASICTS_COMMIT


def test_deterministic_policy_is_fully_enabled():
    configure_determinism(2026)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_best_last_checkpoint_strict_reload_without_resume_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    manager = CheckpointManager(tmp_path)
    manager.save_last(model, {"epoch": 1})
    assert manager.maybe_save_best(model, metric=2.0, metadata={"epoch": 1})
    assert not manager.maybe_save_best(model, metric=3.0, metadata={"epoch": 2})

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    metadata = load_model_checkpoint(manager.best_path, model, map_location="cpu", strict=True)

    assert metadata == {"epoch": 1}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    payload = torch.load(manager.last_path, map_location="cpu")
    assert set(payload) == {"model_state", "metadata"}
    assert "optimizer_state" not in payload
    assert "rng_state" not in payload
