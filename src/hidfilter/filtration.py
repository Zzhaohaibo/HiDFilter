from __future__ import annotations

import torch


def safe_masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute candidate softmax in FP32 with exact-zero invalid rows."""

    logits_fp32 = logits.to(torch.float32)
    valid = torch.broadcast_to(valid.to(torch.bool), logits.shape)
    finite_or_invalid = torch.isfinite(
        torch.where(valid, logits_fp32, torch.zeros((), device=logits.device))
    )
    torch._assert_async(finite_or_invalid.all(), "non-finite valid logits")

    any_valid = valid.any(dim=-1, keepdim=True)
    masked_logits = logits_fp32.masked_fill(~valid, -torch.inf)
    safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
    probability = torch.softmax(safe_logits, dim=-1)
    return torch.where(valid & any_valid, probability, torch.zeros_like(probability))


def edge_top_p(
    probability: torch.Tensor,
    valid: torch.Tensor,
    *,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply deterministic Top-p along canonical candidate order."""

    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0,1], got {rho}")
    valid = torch.broadcast_to(valid.to(torch.bool), probability.shape)
    support_probability = probability.detach().to(torch.float32)

    if rho == 1.0:
        keep = valid
    else:
        masked = torch.where(
            valid,
            support_probability,
            torch.full((), -torch.inf, dtype=torch.float32, device=probability.device),
        )
        sorted_probability = torch.sort(masked, dim=-1, descending=True, stable=True).values
        cumulative = _deterministic_inclusive_scan(sorted_probability)
        tolerance = 4.0 * (probability.shape[-1] + 1) * torch.finfo(torch.float32).eps
        crossing = (cumulative + tolerance >= rho) & torch.isfinite(sorted_probability)
        valid_count = valid.sum(dim=-1)
        first_crossing = crossing.to(torch.int64).argmax(dim=-1) + 1
        support_size = torch.where(crossing.any(dim=-1), first_crossing, valid_count)

        cutoff_position = (support_size - 1).clamp_min(0).unsqueeze(-1)
        cutoff = sorted_probability.gather(dim=-1, index=cutoff_position).squeeze(-1)
        above_cutoff = valid & (support_probability > cutoff.unsqueeze(-1))
        tied_at_cutoff = valid & (support_probability == cutoff.unsqueeze(-1))
        remaining = (support_size - above_cutoff.sum(dim=-1)).clamp_min(0)
        tie_rank = _deterministic_inclusive_scan(tied_at_cutoff.to(torch.int64))
        keep = above_cutoff | (tied_at_cutoff & (tie_rank <= remaining.unsqueeze(-1)))

    selected = torch.where(
        keep & (probability > 0), probability, torch.zeros((), dtype=probability.dtype, device=probability.device)
    )
    denominator = selected.sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    weight = selected / safe_denominator
    weight = torch.where(denominator > 0, weight, torch.zeros_like(weight))
    return keep, weight


def _deterministic_inclusive_scan(values: torch.Tensor) -> torch.Tensor:
    """Inclusive last-axis scan without PyTorch 2.1's nondeterministic CUDA cumsum."""

    result = values
    offset = 1
    while offset < values.shape[-1]:
        shifted = torch.cat(
            (torch.zeros_like(result[..., :offset]), result[..., :-offset]),
            dim=-1,
        )
        result = result + shifted
        offset *= 2
    return result
