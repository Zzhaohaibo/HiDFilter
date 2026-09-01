from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


FORMAL_CONFIG_VERSION = 1


def formal_learning_rate(epoch_number: int) -> float:
    """Frozen cosine value for displayed epochs 1..100."""

    if not 1 <= epoch_number <= 100:
        raise ValueError("epoch_number must be in 1..100")
    epoch_index = epoch_number - 1
    return 1.0e-5 + 0.5 * (1.0e-3 - 1.0e-5) * (
        1.0 + math.cos(math.pi * epoch_index / 99)
    )


def set_formal_learning_rate(optimizer: Any, epoch_number: int) -> float:
    learning_rate = formal_learning_rate(epoch_number)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


@dataclass(frozen=True)
class EarlyStoppingDecision:
    epoch: int
    eligible: bool
    improved: bool
    should_stop: bool
    non_improving_epochs: int


class WarmupEarlyStopping:
    def __init__(
        self,
        *,
        patience: int,
        min_delta: float,
        first_eligible_epoch: int,
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.min_delta = min_delta
        self.first_eligible_epoch = first_eligible_epoch
        self.best_metric = float("inf")
        self.best_epoch: int | None = None
        self.non_improving_epochs = 0
        self._last_epoch = 0

    def observe(self, epoch: int, metric: float) -> EarlyStoppingDecision:
        if epoch <= self._last_epoch:
            raise ValueError("epochs must be observed in strictly increasing order")
        if not math.isfinite(metric):
            raise FloatingPointError("early-stopping metric must be finite")
        self._last_epoch = epoch
        if epoch < self.first_eligible_epoch:
            return EarlyStoppingDecision(
                epoch=epoch,
                eligible=False,
                improved=False,
                should_stop=False,
                non_improving_epochs=self.non_improving_epochs,
            )
        improved = metric < self.best_metric - self.min_delta
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.non_improving_epochs = 0
        else:
            self.non_improving_epochs += 1
        return EarlyStoppingDecision(
            epoch=epoch,
            eligible=True,
            improved=improved,
            should_stop=self.non_improving_epochs >= self.patience,
            non_improving_epochs=self.non_improving_epochs,
        )


class FormalTestOnceGuard:
    """Process-local final-evaluation lifecycle guard."""

    def __init__(self, *, mode: str) -> None:
        if mode not in {"development", "final"}:
            raise ValueError("mode must be development or final")
        self.mode = mode
        self.training_complete = False
        self.best_reloaded = False
        self.test_executed = False

    @property
    def should_run_test(self) -> bool:
        return (
            self.mode == "final"
            and self.training_complete
            and self.best_reloaded
            and not self.test_executed
        )

    def mark_training_complete(self) -> None:
        self.training_complete = True

    def mark_best_reloaded(self) -> None:
        if not self.training_complete:
            raise RuntimeError("best reload requires completed training")
        self.best_reloaded = True

    def begin_test(self) -> None:
        if self.mode == "development":
            raise RuntimeError("development mode must not evaluate test data")
        if self.test_executed:
            raise RuntimeError("test evaluation already executed")
        if not self.training_complete or not self.best_reloaded:
            raise RuntimeError("test evaluation requires training completion and best reload")
        self.test_executed = True


def formal_config_fingerprint(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
