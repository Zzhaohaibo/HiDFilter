#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import time
from statistics import mean

import torch

from hidfilter.model import SelfOnlyHiDFilter
from hidfilter.runtime.phase0 import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Phase 1 performance diagnostic")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SelfOnlyHiDFilter(num_nodes=170).to(device=device, dtype=torch.float32).train()
    optimizer = build_optimizer(model)
    history = torch.randn(64, 12, 170, 1, device=device)
    target = torch.randn(64, 12, 170, 1, device=device)

    for _ in range(args.warmup):
        _full_step(model, optimizer, history, target)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_times = []
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        prediction = model(history)
        _synchronize(device)
        forward_times.append(time.perf_counter() - started)
        del prediction

    forward_backward_times = []
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        prediction = model(history)
        torch.nn.functional.l1_loss(prediction, target).backward()
        _synchronize(device)
        forward_backward_times.append(time.perf_counter() - started)

    optimizer_times = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        _synchronize(device)
        optimizer_times.append(time.perf_counter() - started)

    print("LOCAL PERFORMANCE DIAGNOSTIC")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print("shape=B64,N170,L12,H12,D64,C12")
    print(f"forward_ms={mean(forward_times) * 1_000.0:.3f}")
    print(f"forward_backward_ms={mean(forward_backward_times) * 1_000.0:.3f}")
    print(f"optimizer_ms={mean(optimizer_times) * 1_000.0:.3f}")
    if device.type == "cuda":
        print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
        print(f"peak_reserved_bytes={torch.cuda.max_memory_reserved(device)}")
    else:
        print("peak_allocated_bytes=unavailable")
        print("peak_reserved_bytes=unavailable")

    if args.profile:
        _profile_one_step(model, optimizer, history, target, device)


def _full_step(
    model: SelfOnlyHiDFilter,
    optimizer: torch.optim.Optimizer,
    history: torch.Tensor,
    target: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_one_step(
    model: SelfOnlyHiDFilter,
    optimizer: torch.optim.Optimizer,
    history: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
        _full_step(model, optimizer, history, target)
        _synchronize(device)
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profile.key_averages().table(sort_by=sort_by, row_limit=15))


if __name__ == "__main__":
    main()
