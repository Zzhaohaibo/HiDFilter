#!/usr/bin/env bash
set -euo pipefail

readonly BASICTS_COMMIT='ab35018dda4dec03f642c6cc50c1b4cdcdd2e5b5'

verify_frozen_environment() {
    python - <<'PY'
import platform
import sys

import torch

failures = []
if sys.version_info[:2] != (3, 10):
    failures.append(f"Python is {platform.python_version()}, expected 3.10.x")
if torch.__version__ != "2.1.2+cu118":
    failures.append(f"PyTorch is {torch.__version__}, expected 2.1.2+cu118")
cuda_version = torch.version.cuda or ""
if cuda_version.split(".")[:2] != ["11", "8"]:
    failures.append(f"PyTorch CUDA runtime is {cuda_version or 'unavailable'}, expected 11.8")
if not torch.cuda.is_available():
    failures.append("CUDA is unavailable")
else:
    device_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in device_name:
        failures.append(f"GPU is {device_name}, expected NVIDIA RTX 3090")
if failures:
    raise SystemExit("; ".join(failures))
PY
}

git submodule update --init --recursive
test "$(git -C third_party/BasicTS rev-parse HEAD)" = "${BASICTS_COMMIT}"
verify_frozen_environment
python -m pip install --constraint scripts/phase0_constraints.txt -r third_party/BasicTS/requirements.txt 'blosc2==2.7.1'
python -m pip install --constraint scripts/phase0_constraints.txt 'pytest>=8,<9'
python -m pip install --no-deps -e third_party/BasicTS
python -m pip install --no-deps -e .
verify_frozen_environment
python -c 'import basicts; assert basicts.__version__ == "1.1.0"'
python -m pip check
