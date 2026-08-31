from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hidfilter.filtration import edge_top_p, safe_masked_softmax


HISTORY_LENGTH = 12
FORECAST_HORIZON = 12
HIDDEN_DIM = 64
IDENTITY_DIM = 16
FAMILY_COUNT = 3
SELF_FAMILY_ID = 0
EDGE_TOP_P_RHO = 0.8


def history_to_lag_values(history: torch.Tensor) -> torch.Tensor:
    """Map oldest-to-newest history to latest-to-oldest lag order."""

    if history.ndim != 4 or history.shape[1] != HISTORY_LENGTH or history.shape[-1] != 1:
        raise ValueError(f"history must have shape [B,12,N,1], got {history.shape}")
    return history.flip(dims=(1,)).permute(0, 2, 1, 3)


class TargetContextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HISTORY_LENGTH, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        target_history = history[..., 0].permute(0, 2, 1)
        return self.network(target_history)


class LagContentEncoder(nn.Module):
    """Identity-free pointwise encoder for exact sensor-lag content."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.lag_embedding = nn.Embedding(HISTORY_LENGTH, IDENTITY_DIM)
        self.lag_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        lag_values = history_to_lag_values(history)
        lag_identity = self.lag_projection(self.lag_embedding.weight)
        return self.mlp(lag_values) + lag_identity.view(1, 1, HISTORY_LENGTH, HIDDEN_DIM)


@dataclass(frozen=True)
class SelfCandidateMetadata:
    source_index: torch.Tensor
    lag_index: torch.Tensor
    flat_index: torch.Tensor
    valid: torch.Tensor
    prior: torch.Tensor


def build_self_candidate_metadata(num_nodes: int) -> SelfCandidateMetadata:
    source_index = torch.arange(num_nodes, dtype=torch.int64).unsqueeze(1).expand(-1, HISTORY_LENGTH)
    lag_index = torch.arange(HISTORY_LENGTH, dtype=torch.int64).unsqueeze(0).expand(num_nodes, -1)
    flat_index = source_index * HISTORY_LENGTH + lag_index
    return SelfCandidateMetadata(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=flat_index,
        valid=torch.ones((num_nodes, HISTORY_LENGTH), dtype=torch.bool),
        prior=torch.ones((num_nodes, HISTORY_LENGTH), dtype=torch.float32),
    )


def gather_candidates(global_values: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    """Gather target-specific candidates from sensor-major, lag-minor global values."""

    batch_size, _, _, width = global_values.shape
    flat_values = global_values.reshape(batch_size, -1, width)
    gathered = flat_values[:, flat_index.reshape(-1), :]
    return gathered.reshape(batch_size, *flat_index.shape, width)


class SelfOnlyHiDFilter(nn.Module):
    """Phase 1 HiDFilter with only the Self dependency space."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.context_encoder = TargetContextEncoder()
        self.lag_content_encoder = LagContentEncoder()

        self.node_embedding = nn.Embedding(num_nodes, IDENTITY_DIM)
        self.horizon_embedding = nn.Embedding(FORECAST_HORIZON, IDENTITY_DIM)
        self.fine_family_embedding = nn.Embedding(FAMILY_COUNT, IDENTITY_DIM)
        self.node_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)
        self.horizon_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)
        self.fine_family_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)

        self.fine_token_norm = nn.LayerNorm(HIDDEN_DIM)
        self.query_norm = nn.LayerNorm(HIDDEN_DIM)
        self.wq = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.wk = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.wv = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32))

        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2 + IDENTITY_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, 1),
        )

        metadata = build_self_candidate_metadata(num_nodes)
        self.register_buffer("self_source_index", metadata.source_index, persistent=False)
        self.register_buffer("self_lag_index", metadata.lag_index, persistent=False)
        self.register_buffer("self_flat_index", metadata.flat_index, persistent=False)
        self.register_buffer("self_valid", metadata.valid, persistent=False)
        self.register_buffer("self_prior", metadata.prior, persistent=False)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def encode_fine_tokens(self, lag_content: torch.Tensor) -> torch.Tensor:
        source_identity = self.node_projection(self.node_embedding.weight)
        return self.fine_token_norm(lag_content + source_identity.view(1, self.num_nodes, 1, HIDDEN_DIM))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[1:] != (HISTORY_LENGTH, self.num_nodes, 1):
            raise ValueError(
                f"history must have shape [B,{HISTORY_LENGTH},{self.num_nodes},1], got {history.shape}"
            )

        context = self.context_encoder(history)
        lag_content = self.lag_content_encoder(history)
        fine_tokens = self.encode_fine_tokens(lag_content)

        node_identity = self.node_projection(self.node_embedding.weight)
        horizon_identity = self.horizon_projection(self.horizon_embedding.weight)
        query_input = (
            context.unsqueeze(2)
            + node_identity.view(1, self.num_nodes, 1, HIDDEN_DIM)
            + horizon_identity.view(1, 1, FORECAST_HORIZON, HIDDEN_DIM)
        )
        query = self.wq(self.query_norm(query_input))

        fine_family_identity = self.fine_family_projection(
            self.fine_family_embedding.weight[SELF_FAMILY_ID]
        )
        key_global = self.wk(fine_tokens + fine_family_identity.view(1, 1, 1, HIDDEN_DIM))
        value_global = self.wv(fine_tokens)
        candidate_key = gather_candidates(key_global, self.self_flat_index)
        candidate_value = gather_candidates(value_global, self.self_flat_index)

        score = torch.einsum("bnhd,bncd->bnhc", query, candidate_key) / math.sqrt(HIDDEN_DIM)
        prior_bias = self.alpha * torch.log(self.self_prior.clamp_min(1.0e-6))
        score = score + prior_bias.view(1, self.num_nodes, 1, HISTORY_LENGTH)
        valid = self.self_valid.view(1, self.num_nodes, 1, HISTORY_LENGTH)
        dense_probability = safe_masked_softmax(score, valid)
        _, edge_weight = edge_top_p(dense_probability, valid, rho=EDGE_TOP_P_RHO)
        message = torch.einsum("bnhc,bncd->bnhd", edge_weight, candidate_value)

        horizon_embedding = self.horizon_embedding.weight.view(
            1, 1, FORECAST_HORIZON, IDENTITY_DIM
        ).expand(history.shape[0], self.num_nodes, -1, -1)
        decoder_input = torch.cat(
            (context.unsqueeze(2).expand(-1, -1, FORECAST_HORIZON, -1), message, horizon_embedding),
            dim=-1,
        )
        delta = self.decoder(decoder_input)
        latest = history[:, -1, :, :].unsqueeze(2)
        return (latest + delta).permute(0, 2, 1, 3)
