#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import time
from statistics import mean
from typing import Callable, TypeVar

import numpy as np
import torch

from hidfilter.model import HiDFilter
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.diagnostics import DiagnosticAccumulator
from hidfilter.semantic import SemanticCandidateMetadata


T = TypeVar("T")
NUM_NODES = 170


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare plain and post-hoc diagnostic HiDFilter inference"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
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
    ).to(device=device, dtype=torch.float32).eval()
    history = torch.randn(args.batch_size, 12, NUM_NODES, 1, device=device)
    valid_query = torch.ones(
        (args.batch_size, NUM_NODES, 12), dtype=torch.bool, device=device
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(history, family_top_p_enabled=True)
            model.forward_with_diagnostics(history, family_top_p_enabled=True)
        _synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        plain_ms, plain = _measure(
            lambda: model(history, family_top_p_enabled=True),
            device=device,
            iterations=args.iterations,
        )
        plain_peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        diagnostic_forward_ms, diagnostic = _measure(
            lambda: model.forward_with_diagnostics(
                history, family_top_p_enabled=True
            ),
            device=device,
            iterations=args.iterations,
        )
        diagnostic_forward_peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        diagnostic_pass_ms, report = _measure(
            lambda: _diagnostic_pass(model, history, valid_query, device),
            device=device,
            iterations=args.iterations,
        )
        diagnostic_pass_peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )

    if not torch.equal(plain, diagnostic.prediction):
        raise RuntimeError("diagnostic forward changed prediction")
    print("LOCAL DIAGNOSTIC ONLY")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(
        f"device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}"
    )
    print(f"shape=B{args.batch_size},N170,L12,H12,R3,C12+96+96")
    print(f"plain_forward_ms={plain_ms:.3f}")
    print(f"diagnostic_forward_ms={diagnostic_forward_ms:.3f}")
    print(f"diagnostic_full_reduce_ms={diagnostic_pass_ms:.3f}")
    print(f"plain_peak_allocated_bytes={plain_peak}")
    print(f"diagnostic_forward_peak_allocated_bytes={diagnostic_forward_peak}")
    print(f"diagnostic_full_reduce_peak_allocated_bytes={diagnostic_pass_peak}")
    print(f"valid_query_count={report['valid_query_count']}")
    print("training_hot_path_instrumentation=OFF")


def _diagnostic_pass(
    model: HiDFilter,
    history: torch.Tensor,
    valid_query: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    output = model.forward_with_diagnostics(history, family_top_p_enabled=True)
    accumulator = DiagnosticAccumulator(device=device)
    accumulator.update(output.state, valid_query)
    return accumulator.finalize()


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
    targets = torch.arange(NUM_NODES, dtype=torch.int64).view(-1, 1)
    source_slots = (targets + torch.arange(10, 18, dtype=torch.int64)) % NUM_NODES
    source_index = source_slots.repeat_interleave(12, dim=1)
    lag_index = torch.arange(12, dtype=torch.int64).repeat(8).view(1, -1)
    lag_index = lag_index.expand(NUM_NODES, -1).clone()
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
    timings = []
    result = None
    for _ in range(iterations):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            result = operation()
            timings.append((time.perf_counter() - started) * 1_000.0)
    if result is None:
        raise RuntimeError("benchmark operation did not run")
    return mean(timings), result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
