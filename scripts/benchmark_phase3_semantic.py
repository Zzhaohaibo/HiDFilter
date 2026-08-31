#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import platform
import time
from statistics import mean
from typing import Callable, TypeVar

import numpy as np
import torch

from hidfilter.filtration import edge_top_p, safe_masked_softmax
from hidfilter.model import (
    EDGE_TOP_P_RHO,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    SEMANTIC_FAMILY_ID,
    HiDFilter,
    gather_candidates,
)
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.phase0 import build_optimizer
from hidfilter.semantic import build_semantic_candidates


T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Phase 3 Semantic performance diagnostic")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjacency = np.zeros((170, 170), dtype=np.float64)
    for node in range(170):
        adjacency[node, (node + 1) % 170] = 1.0
        adjacency[(node + 1) % 170, node] = 1.0
    physical = build_physical_candidates(
        adjacency,
        PhysicalGraphContract("undirected", "affinity", None),
        kp=8,
    )
    time_axis = np.arange(400, dtype=np.float64)[:, None]
    scales = np.linspace(0.5, 2.0, 170, dtype=np.float64)[None, :]
    raw_train = 10.0 + scales * np.square(time_axis) + np.sin(
        time_axis / np.arange(2, 172, dtype=np.float64)[None, :]
    )
    one_hop = adjacency > 0
    build_started = time.perf_counter()
    semantic = build_semantic_candidates(
        raw_train,
        np.ones_like(raw_train, dtype=np.bool_),
        one_hop,
        physical.sources.source_index.numpy(),
        physical.sources.valid.numpy(),
        ks=8,
        min_overlap=288,
        variance_threshold=1.0e-12,
    )
    candidate_build_seconds = time.perf_counter() - build_started
    model = HiDFilter(
        170,
        physical_candidates=physical.candidates,
        semantic_candidates=semantic.candidates,
    ).to(device=device, dtype=torch.float32).train()
    optimizer = build_optimizer(model)
    history = torch.randn(64, 12, 170, 1, device=device)
    target = torch.randn(64, 12, 170, 1, device=device)

    for _ in range(args.warmup):
        _full_step(model, optimizer, history, target)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        forward_ms, _ = _measure(
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
        query, fine_tokens, value_global, family_identity = _shared_fine_inputs(model, history)
        key_projection_ms, semantic_key_global = _measure(
            lambda: model.wk(fine_tokens + family_identity.view(1, 1, 1, HIDDEN_DIM)),
            device=device,
            iterations=args.iterations,
        )
        gather_ms, gathered = _measure(
            lambda: (
                gather_candidates(semantic_key_global, model.semantic_flat_index),
                gather_candidates(value_global, model.semantic_flat_index),
            ),
            device=device,
            iterations=args.iterations,
        )
        candidate_key, candidate_value = gathered
        fine_ms, dense_probability = _measure(
            lambda: _semantic_dense_probability(model, query, candidate_key),
            device=device,
            iterations=args.iterations,
        )
        valid = model.semantic_valid.view(1, 170, 1, 96)
        top_p_ms, top_p_output = _measure(
            lambda: edge_top_p(dense_probability, valid, rho=EDGE_TOP_P_RHO),
            device=device,
            iterations=args.iterations,
        )
        _, edge_weight = top_p_output
        message_ms, _ = _measure(
            lambda: torch.einsum("bnhc,bncd->bnhd", edge_weight, candidate_value),
            device=device,
            iterations=args.iterations,
        )

    print("LOCAL DIAGNOSTIC ONLY")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print("shape=B64,N170,L12,H12,D64,Cself12,Cphysical96,Csemantic96")
    print(f"semantic_candidate_build_seconds={candidate_build_seconds:.6f}")
    print(f"total_forward_ms={forward_ms:.3f}")
    print(f"total_forward_backward_ms={forward_backward_ms:.3f}")
    print(f"optimizer_ms={optimizer_ms:.3f}")
    print(f"semantic_key_projection_ms={key_projection_ms:.3f}")
    print(f"semantic_gather_ms={gather_ms:.3f}")
    print(f"semantic_fine_score_softmax_ms={fine_ms:.3f}")
    print(f"semantic_top_p_ms={top_p_ms:.3f}")
    print(f"semantic_message_ms={message_ms:.3f}")
    if device.type == "cuda":
        print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
        print(f"peak_reserved_bytes={torch.cuda.max_memory_reserved(device)}")
    else:
        print("peak_allocated_bytes=unavailable")
        print("peak_reserved_bytes=unavailable")

    if args.profile:
        _profile_one_step(model, optimizer, history, target, device)


def _shared_fine_inputs(
    model: HiDFilter,
    history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context = model.context_encoder(history)
    lag_content = model.lag_content_encoder(history)
    fine_tokens = model.encode_fine_tokens(lag_content)
    node_identity = model.node_projection(model.node_embedding.weight)
    horizon_identity = model.horizon_projection(model.horizon_embedding.weight)
    query_input = (
        context.unsqueeze(2)
        + node_identity.view(1, 170, 1, HIDDEN_DIM)
        + horizon_identity.view(1, 1, FORECAST_HORIZON, HIDDEN_DIM)
    )
    query = model.wq(model.query_norm(query_input))
    value_global = model.wv(fine_tokens)
    family_identity = model.fine_family_projection(
        model.fine_family_embedding.weight[SEMANTIC_FAMILY_ID]
    )
    return query, fine_tokens, value_global, family_identity


def _semantic_dense_probability(
    model: HiDFilter,
    query: torch.Tensor,
    candidate_key: torch.Tensor,
) -> torch.Tensor:
    score = torch.einsum("bnhd,bncd->bnhc", query, candidate_key) / math.sqrt(HIDDEN_DIM)
    score = score + (
        model.alpha * torch.log(model.semantic_prior.clamp_min(1.0e-6))
    ).view(1, 170, 1, 96)
    return safe_masked_softmax(score, model.semantic_valid.view(1, 170, 1, 96))


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
) -> None:
    _forward_backward(model, optimizer, history, target)
    _optimizer_step(model, optimizer)


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
