from __future__ import annotations

import subprocess
from pathlib import Path

import basicts


BASICTS_VERSION = "1.1.0"
BASICTS_COMMIT = "ab35018dda4dec03f642c6cc50c1b4cdcdd2e5b5"


def verify_basicts_revision(path: str | Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != BASICTS_COMMIT:
        raise RuntimeError(f"BasicTS revision is {revision}, expected {BASICTS_COMMIT}")
    if basicts.__version__ != BASICTS_VERSION:
        raise RuntimeError(f"BasicTS version is {basicts.__version__}, expected {BASICTS_VERSION}")
    return revision
