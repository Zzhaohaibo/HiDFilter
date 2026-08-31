from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_pems08_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("pems08")
    split_steps = {"train": 10_713, "val": 3_571, "test": 3_572}
    offsets = {"train": 1.0, "val": 10_000_000.0, "test": 20_000_000.0}
    for split, steps in split_steps.items():
        values = np.arange(steps * 170, dtype=np.float32).reshape(steps, 170)
        np.save(tmp_path / f"{split}_data.npy", values + offsets[split])
    return tmp_path
