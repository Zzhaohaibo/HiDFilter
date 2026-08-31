from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from basicts.metrics import masked_mae
from basicts.scaler import ZScoreScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from hidfilter.protocol.metrics import RawMetricAccumulator, RawMetricReport
from hidfilter.protocol.pems08 import prepare_batch
from hidfilter.runtime.determinism import seed_worker
from hidfilter.runtime.stid import stid_forward


@dataclass(frozen=True)
class WorkerBenchmark:
    num_workers: int
    batches: int
    samples: int
    seconds: float
    samples_per_second: float


@dataclass(frozen=True)
class TrainingEpoch:
    loss: float
    steps: int
    samples: int
    seconds: float
    milliseconds_per_step: float
    samples_per_second: float
    data_wait_seconds: float
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float


@dataclass(frozen=True)
class ValidationEpoch:
    metrics: RawMetricReport
    persistence: RawMetricReport
    samples: int
    seconds: float
    samples_per_second: float
    data_wait_seconds: float


class _OperationTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._cuda_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_seconds = 0.0

    def start(self) -> torch.cuda.Event | float:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, started: torch.cuda.Event | float) -> None:
        if self.device.type == "cuda":
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self._cuda_pairs.append((started, ended))
        else:
            self._cpu_seconds += time.perf_counter() - started

    def seconds(self) -> float:
        if self.device.type == "cuda":
            return sum(start.elapsed_time(end) for start, end in self._cuda_pairs) / 1_000.0
        return self._cpu_seconds


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def benchmark_worker_counts(
    dataset: Dataset,
    *,
    batch_size: int,
    candidates: Sequence[int] = (0, 2, 4, 8),
    max_batches: int = 64,
    pin_memory: bool = False,
) -> tuple[WorkerBenchmark, ...]:
    results = []
    for num_workers in candidates:
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=0,
            pin_memory=pin_memory,
        )
        batches = 0
        samples = 0
        started = time.perf_counter()
        for batch in loader:
            batches += 1
            samples += int(batch["inputs"].shape[0])
            if batches >= max_batches:
                break
        seconds = time.perf_counter() - started
        results.append(
            WorkerBenchmark(
                num_workers=num_workers,
                batches=batches,
                samples=samples,
                seconds=seconds,
                samples_per_second=samples / seconds,
            )
        )
    return tuple(results)


def select_worker_count(results: Iterable[WorkerBenchmark]) -> int:
    rows = tuple(results)
    if not rows:
        raise ValueError("at least one worker benchmark is required")
    return max(rows, key=lambda row: row.samples_per_second).num_workers


def build_optimizer(model: nn.Module) -> AdamW:
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    scaler: ZScoreScaler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float,
) -> TrainingEpoch:
    model.train()
    forward_timer = _OperationTimer(device)
    backward_timer = _OperationTimer(device)
    optimizer_timer = _OperationTimer(device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_sum = torch.zeros((), dtype=torch.float64, device=device)
    data_wait_seconds = 0.0
    samples = 0
    steps = 0
    epoch_started = time.perf_counter()
    iterator = iter(loader)

    while True:
        wait_started = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            break
        data_wait_seconds += time.perf_counter() - wait_started
        raw_batch = _move_batch(cpu_batch, device)
        prepared = prepare_batch(raw_batch, scaler)
        optimizer.zero_grad(set_to_none=True)

        started = forward_timer.start()
        prediction = stid_forward(model, prepared.inputs)
        loss = masked_mae(prediction, prepared.targets, prepared.targets_valid)
        forward_timer.stop(started)

        started = backward_timer.start()
        loss.backward()
        backward_timer.stop(started)

        started = optimizer_timer.start()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer_timer.stop(started)

        valid = prepared.targets_valid.sum()
        loss_sum += loss.detach().to(torch.float64) * valid
        valid_sum += valid
        samples += int(prepared.inputs.shape[0])
        steps += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - epoch_started
    valid_count = valid_sum.item()
    if valid_count == 0:
        raise FloatingPointError("training epoch contains no valid targets")
    average_loss = loss_sum.item() / valid_count
    _assert_finite_boundary(model, average_loss)
    return TrainingEpoch(
        loss=average_loss,
        steps=steps,
        samples=samples,
        seconds=seconds,
        milliseconds_per_step=seconds * 1_000.0 / steps,
        samples_per_second=samples / seconds,
        data_wait_seconds=data_wait_seconds,
        forward_seconds=forward_timer.seconds(),
        backward_seconds=backward_timer.seconds(),
        optimizer_seconds=optimizer_timer.seconds(),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler: ZScoreScaler,
    device: torch.device,
) -> ValidationEpoch:
    model.eval()
    metrics = RawMetricAccumulator(horizons=12)
    persistence = RawMetricAccumulator(horizons=12)
    data_wait_seconds = 0.0
    samples = 0
    started = time.perf_counter()
    iterator = iter(loader)

    while True:
        wait_started = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            break
        data_wait_seconds += time.perf_counter() - wait_started
        raw_batch = _move_batch(cpu_batch, device)
        prepared = prepare_batch(raw_batch, scaler)
        prediction = scaler.inverse_transform(stid_forward(model, prepared.inputs))
        metrics.update(prediction, raw_batch["targets"], prepared.targets_valid)
        latest = raw_batch["inputs"][:, -1:, :, :]
        persistence_prediction = latest.expand(-1, 12, -1, -1)
        persistence.update(persistence_prediction, raw_batch["targets"], prepared.targets_valid)
        samples += int(prepared.inputs.shape[0])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return ValidationEpoch(
        metrics=metrics.compute(),
        persistence=persistence.compute(),
        samples=samples,
        seconds=seconds,
        samples_per_second=samples / seconds,
        data_wait_seconds=data_wait_seconds,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "inputs": batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True),
        "targets": batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True),
    }


def _assert_finite_boundary(model: nn.Module, loss: float) -> None:
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    gradients = [parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    if not torch.isfinite(parameters).all().item():
        raise FloatingPointError("non-finite model parameter at epoch boundary")
    if gradients and not torch.isfinite(torch.cat(gradients)).all().item():
        raise FloatingPointError("non-finite gradient at epoch boundary")
