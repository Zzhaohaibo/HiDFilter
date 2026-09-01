#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import time
from statistics import mean
from typing import Callable, TypeVar

import numpy as np
import torch

from hidfilter.model import FORECAST_HORIZON, HIDDEN_DIM, HiDFilter, dense_router_fusion
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.phase0 import build_optimizer
from hidfilter.semantic import SemanticCandidateMetadata


T = TypeVar("T")
NUM_NODES = 170
BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Phase 4 dense Router performance diagnostic")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    physical, semantic = _synthetic_candidate_metadata()
    model = HiDFilter(
        NUM_NODES,
        physical_candidates=physical,
        semantic_candidates=semantic,
    ).to(device=device, dtype=torch.float32).train()
    optimizer = build_optimizer(model)
    history = torch.randn(BATCH_SIZE, 12, NUM_NODES, 1, device=device)
    target = torch.randn(BATCH_SIZE, 12, NUM_NODES, 1, device=device)

    for _ in range(args.warmup):
        _full_step(model, optimizer, history, target)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_step_ms, _ = _measure(
        lambda: _full_step(model, optimizer, history, target),
        device=device,
        iterations=args.iterations,
    )
    with torch.no_grad():
        total_forward_ms, _ = _measure(
            lambda: model(history), device=device, iterations=args.iterations
        )
    forward_backward_ms, _ = _measure(
        lambda: _forward_backward(model, optimizer, history, target),
        device=device,
        iterations=args.iterations,
    )
    optimizer_ms, _ = _measure(
        lambda: _optimizer_step(model, optimizer),
        device=device,
        iterations=args.iterations,
    )

    with torch.no_grad():
        context = model.context_encoder(history)
        lag_content = model.lag_content_encoder(history)
        evidence_ms, family_evidence = _measure(
            lambda: model.encode_family_evidence(lag_content),
            device=device,
            iterations=args.iterations,
        )
        router_ms, probability = _measure(
            lambda: model.compute_router_probability(context, family_evidence),
            device=device,
            iterations=args.iterations,
        )
        messages = tuple(
            torch.randn(
                BATCH_SIZE,
                NUM_NODES,
                FORECAST_HORIZON,
                HIDDEN_DIM,
                device=device,
            )
            for _ in range(3)
        )
        fusion_ms, _ = _measure(
            lambda: dense_router_fusion(*messages, probability),
            device=device,
            iterations=args.iterations,
        )

    print("LOCAL DIAGNOSTIC ONLY")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print("shape=B64,N170,L12,H12,D64,R3,Cself12,Cphysical96,Csemantic96")
    print(f"total_step_ms={total_step_ms:.3f}")
    print(f"total_forward_ms={total_forward_ms:.3f}")
    print(f"total_forward_backward_ms={forward_backward_ms:.3f}")
    print(f"optimizer_ms={optimizer_ms:.3f}")
    print(f"family_evidence_ms={evidence_ms:.3f}")
    print(f"router_ms={router_ms:.3f}")
    print(f"dense_fusion_ms={fusion_ms:.3f}")
    if device.type == "cuda":
        print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
        print(f"peak_reserved_bytes={torch.cuda.max_memory_reserved(device)}")
    else:
        print("peak_allocated_bytes=unavailable")
        print("peak_reserved_bytes=unavailable")

    if args.profile:
        _profile_one_step(model, optimizer, history, target, device)


def _synthetic_candidate_metadata():
    adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float64)
    for node in range(NUM_NODES):
        adjacency[node, (node + 1) % NUM_NODES] = 1.0
        adjacency[(node + 1) % NUM_NODES, node] = 1.0
    physical = build_physical_candidates(
        adjacency,
        PhysicalGraphContract("undirected", "affinity", None),
        kp=8,
    ).candidates

    target = torch.arange(NUM_NODES, dtype=torch.int64).view(-1, 1)
    offsets = torch.arange(10, 18, dtype=torch.int64).view(1, -1)
    source_slots = (target + offsets) % NUM_NODES
    source_index = source_slots.repeat_interleave(12, dim=1)
    lag_index = (
        torch.arange(12, dtype=torch.int64)
        .repeat(8)
        .view(1, -1)
        .expand(NUM_NODES, -1)
        .clone()
    )
    valid = torch.ones((NUM_NODES, 96), dtype=torch.bool)
    semantic = SemanticCandidateMetadata(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=source_index * 12 + lag_index,
        valid=valid,
        prior=torch.ones((NUM_NODES, 96), dtype=torch.float32),
    )
    return physical, semantic


def _measure(
    operation: Callable[[], T],
    *,
    device: torch.device,
    iterations: int,
) -> tuple[float, T]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        result = None
        for start, end in zip(starts, ends):
            start.record()
            result = operation()
            end.record()
        torch.cuda.synchronize(device)
        timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    else:
        timings = []
        result = None
        for _ in range(iterations):
            started = time.perf_counter()
            result = operation()
            timings.append((time.perf_counter() - started) * 1_000.0)
    assert result is not None
    return mean(timings), result


def _forward_backward(
    model: HiDFilter,
    optimizer: torch.optim.Optimizer,
    history: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    return loss


def _optimizer_step(model: HiDFilter, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return norm


def _full_step(
    model: HiDFilter,
    optimizer: torch.optim.Optimizer,
    history: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    loss = _forward_backward(model, optimizer, history, target)
    _optimizer_step(model, optimizer)
    return loss


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_one_step(
    model: HiDFilter,
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
    print(profile.key_averages().table(sort_by=sort_by, row_limit=20))


if __name__ == "__main__":
    main()
