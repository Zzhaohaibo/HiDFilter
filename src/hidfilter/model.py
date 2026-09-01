from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hidfilter.filtration import edge_top_p, family_top_p, safe_masked_softmax
from hidfilter.physical import PhysicalCandidateMetadata
from hidfilter.semantic import SemanticCandidateMetadata


HISTORY_LENGTH = 12
FORECAST_HORIZON = 12
HIDDEN_DIM = 64
IDENTITY_DIM = 16
FAMILY_COUNT = 3
SELF_FAMILY_ID = 0
PHYSICAL_FAMILY_ID = 1
SEMANTIC_FAMILY_ID = 2
EDGE_TOP_P_RHO = 0.8
FAMILY_TOP_P_RHO = 0.8
ROUTER_INPUT_DIM = HIDDEN_DIM + IDENTITY_DIM + IDENTITY_DIM + HIDDEN_DIM + IDENTITY_DIM
ROUTER_HIDDEN_DIM = 64


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


def compute_family_evidence(
    lag_mean: torch.Tensor,
    source_index: torch.Tensor,
    source_valid: torch.Tensor,
) -> torch.Tensor:
    """Aggregate identity-free source content with an unweighted masked mean."""

    batch_size, _, width = lag_mean.shape
    gathered = lag_mean[:, source_index.reshape(-1), :]
    gathered = gathered.reshape(batch_size, *source_index.shape, width)
    valid = source_valid.view(1, *source_valid.shape, 1)
    source_sum = torch.where(valid, gathered, torch.zeros_like(gathered)).sum(dim=2)
    source_count = valid.sum(dim=2)
    denominator = torch.where(source_count > 0, source_count, torch.ones_like(source_count))
    evidence = source_sum / denominator
    return torch.where(source_count > 0, evidence, torch.zeros_like(evidence))


def masked_router_softmax(
    router_logits: torch.Tensor,
    family_available: torch.Tensor,
) -> torch.Tensor:
    """Normalize dense family logits over available families only."""

    available = family_available.view(1, family_available.shape[0], 1, FAMILY_COUNT)
    return safe_masked_softmax(router_logits, available)


def compute_router_probability(
    context: torch.Tensor,
    family_evidence: torch.Tensor,
    node_identity: torch.Tensor,
    horizon_identity: torch.Tensor,
    route_family_identity: torch.Tensor,
    family_available: torch.Tensor,
    router_scorer: nn.Sequential,
) -> torch.Tensor:
    """Score the three families without materializing the 176-wide Router input."""

    input_layer = router_scorer[0]
    weight = input_layer.weight
    context_end = HIDDEN_DIM
    node_end = context_end + IDENTITY_DIM
    horizon_end = node_end + IDENTITY_DIM
    evidence_end = horizon_end + HIDDEN_DIM

    context_hidden = F.linear(
        context,
        weight[:, :context_end],
        input_layer.bias,
    ).unsqueeze(2).unsqueeze(3)
    node_hidden = F.linear(
        node_identity,
        weight[:, context_end:node_end],
    ).view(1, node_identity.shape[0], 1, 1, ROUTER_HIDDEN_DIM)
    horizon_hidden = F.linear(
        horizon_identity,
        weight[:, node_end:horizon_end],
    ).view(1, 1, horizon_identity.shape[0], 1, ROUTER_HIDDEN_DIM)
    evidence_hidden = F.linear(
        family_evidence,
        weight[:, horizon_end:evidence_end],
    ).unsqueeze(2)
    route_hidden = F.linear(
        route_family_identity,
        weight[:, evidence_end:ROUTER_INPUT_DIM],
    ).view(1, 1, 1, FAMILY_COUNT, ROUTER_HIDDEN_DIM)
    hidden = (
        context_hidden
        + node_hidden
        + horizon_hidden
        + evidence_hidden
        + route_hidden
    )
    router_logits = router_scorer[2](router_scorer[1](hidden)).squeeze(-1)
    return masked_router_softmax(router_logits, family_available)


def dense_router_fusion(
    self_message: torch.Tensor,
    physical_message: torch.Tensor,
    semantic_message: torch.Tensor,
    router_probability: torch.Tensor,
) -> torch.Tensor:
    """Fuse family messages with dense Router probabilities."""

    return (
        router_probability[..., SELF_FAMILY_ID, None] * self_message
        + router_probability[..., PHYSICAL_FAMILY_ID, None] * physical_message
        + router_probability[..., SEMANTIC_FAMILY_ID, None] * semantic_message
    )


@dataclass(frozen=True)
class FineFamilyOutput:
    dense_probability: torch.Tensor
    edge_keep: torch.Tensor
    edge_weight: torch.Tensor
    message: torch.Tensor


def compute_fine_family(
    query: torch.Tensor,
    key_global: torch.Tensor,
    value_global: torch.Tensor,
    flat_index: torch.Tensor,
    valid: torch.Tensor,
    prior: torch.Tensor,
    alpha: torch.Tensor,
) -> FineFamilyOutput:
    """Compute one independent family distribution without materializing horizon-value tensors."""

    candidate_key = gather_candidates(key_global, flat_index)
    candidate_value = gather_candidates(value_global, flat_index)
    candidate_count = flat_index.shape[-1]
    score = torch.einsum("bnhd,bncd->bnhc", query, candidate_key) / math.sqrt(HIDDEN_DIM)
    prior_bias = alpha * torch.log(prior.clamp_min(1.0e-6))
    score = score + prior_bias.view(1, flat_index.shape[0], 1, candidate_count)
    broadcast_valid = valid.view(1, flat_index.shape[0], 1, candidate_count)
    dense_probability = safe_masked_softmax(score, broadcast_valid)
    edge_keep, edge_weight = edge_top_p(
        dense_probability, broadcast_valid, rho=EDGE_TOP_P_RHO
    )
    message = torch.einsum("bnhc,bncd->bnhd", edge_weight, candidate_value)
    return FineFamilyOutput(
        dense_probability=dense_probability,
        edge_keep=edge_keep,
        edge_weight=edge_weight,
        message=message,
    )


def equal_average_family_messages(
    self_message: torch.Tensor,
    physical_message: torch.Tensor,
    physical_available: torch.Tensor,
) -> torch.Tensor:
    """Phase 2 development fusion: equal average over available families only."""

    available = physical_available.to(dtype=self_message.dtype, device=self_message.device)
    available = available.view(1, -1, 1, 1)
    return (self_message + available * physical_message) / (1.0 + available)


def equal_average_three_family_messages(
    self_message: torch.Tensor,
    physical_message: torch.Tensor,
    physical_available: torch.Tensor,
    semantic_message: torch.Tensor,
    semantic_available: torch.Tensor,
) -> torch.Tensor:
    """Phase 3 development fusion: equal average over available families only."""

    physical = physical_available.to(dtype=self_message.dtype, device=self_message.device)
    semantic = semantic_available.to(dtype=self_message.dtype, device=self_message.device)
    physical = physical.view(1, -1, 1, 1)
    semantic = semantic.view(1, -1, 1, 1)
    return (
        self_message + physical * physical_message + semantic * semantic_message
    ) / (1.0 + physical + semantic)


class HiDFilter(nn.Module):
    """Single evolving core for optional Physical and Semantic Fine families."""

    def __init__(
        self,
        num_nodes: int,
        physical_candidates: PhysicalCandidateMetadata | None = None,
        semantic_candidates: SemanticCandidateMetadata | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.context_encoder = TargetContextEncoder()
        self.lag_content_encoder = LagContentEncoder()

        self.node_embedding = nn.Embedding(num_nodes, IDENTITY_DIM)
        self.horizon_embedding = nn.Embedding(FORECAST_HORIZON, IDENTITY_DIM)
        self.fine_family_embedding = nn.Embedding(FAMILY_COUNT, IDENTITY_DIM)
        self.route_family_embedding = nn.Embedding(FAMILY_COUNT, IDENTITY_DIM)
        self.node_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)
        self.horizon_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)
        self.fine_family_projection = nn.Linear(IDENTITY_DIM, HIDDEN_DIM)

        self.fine_token_norm = nn.LayerNorm(HIDDEN_DIM)
        self.query_norm = nn.LayerNorm(HIDDEN_DIM)
        self.wq = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.wk = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.wv = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32))
        self.router_scorer = nn.Sequential(
            nn.Linear(ROUTER_INPUT_DIM, ROUTER_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(ROUTER_HIDDEN_DIM, 1),
        )

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

        self.has_physical = physical_candidates is not None
        if physical_candidates is None:
            physical_candidates = PhysicalCandidateMetadata(
                source_index=torch.empty((num_nodes, 0), dtype=torch.int64),
                lag_index=torch.empty((num_nodes, 0), dtype=torch.int64),
                flat_index=torch.empty((num_nodes, 0), dtype=torch.int64),
                valid=torch.empty((num_nodes, 0), dtype=torch.bool),
                prior=torch.empty((num_nodes, 0), dtype=torch.float32),
            )
        self._validate_physical_candidates(physical_candidates)
        self.register_buffer(
            "physical_source_index", physical_candidates.source_index, persistent=False
        )
        self.register_buffer("physical_lag_index", physical_candidates.lag_index, persistent=False)
        self.register_buffer("physical_flat_index", physical_candidates.flat_index, persistent=False)
        self.register_buffer("physical_valid", physical_candidates.valid, persistent=False)
        self.register_buffer("physical_prior", physical_candidates.prior, persistent=False)
        self.register_buffer(
            "physical_available", physical_candidates.valid.any(dim=-1), persistent=False
        )
        self.register_buffer(
            "physical_evidence_source_index",
            physical_candidates.source_index[:, ::HISTORY_LENGTH].contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "physical_evidence_source_valid",
            physical_candidates.valid[:, ::HISTORY_LENGTH].contiguous(),
            persistent=False,
        )

        self.has_semantic = semantic_candidates is not None
        if semantic_candidates is None:
            semantic_candidates = SemanticCandidateMetadata(
                source_index=torch.empty((num_nodes, 0), dtype=torch.int64),
                lag_index=torch.empty((num_nodes, 0), dtype=torch.int64),
                flat_index=torch.empty((num_nodes, 0), dtype=torch.int64),
                valid=torch.empty((num_nodes, 0), dtype=torch.bool),
                prior=torch.empty((num_nodes, 0), dtype=torch.float32),
            )
        self._validate_semantic_candidates(semantic_candidates)
        self.register_buffer(
            "semantic_source_index", semantic_candidates.source_index, persistent=False
        )
        self.register_buffer("semantic_lag_index", semantic_candidates.lag_index, persistent=False)
        self.register_buffer(
            "semantic_flat_index", semantic_candidates.flat_index, persistent=False
        )
        self.register_buffer("semantic_valid", semantic_candidates.valid, persistent=False)
        self.register_buffer("semantic_prior", semantic_candidates.prior, persistent=False)
        self.register_buffer(
            "semantic_available", semantic_candidates.valid.any(dim=-1), persistent=False
        )
        self.register_buffer(
            "semantic_evidence_source_index",
            semantic_candidates.source_index[:, ::HISTORY_LENGTH].contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "semantic_evidence_source_valid",
            semantic_candidates.valid[:, ::HISTORY_LENGTH].contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "family_available",
            torch.stack(
                (
                    torch.ones(num_nodes, dtype=torch.bool),
                    self.physical_available,
                    self.semantic_available,
                ),
                dim=-1,
            ),
            persistent=False,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def encode_fine_tokens(self, lag_content: torch.Tensor) -> torch.Tensor:
        source_identity = self.node_projection(self.node_embedding.weight)
        return self.fine_token_norm(lag_content + source_identity.view(1, self.num_nodes, 1, HIDDEN_DIM))

    def encode_family_evidence(self, lag_content: torch.Tensor) -> torch.Tensor:
        lag_mean = lag_content.float().mean(dim=2)
        physical_evidence = compute_family_evidence(
            lag_mean,
            self.physical_evidence_source_index,
            self.physical_evidence_source_valid,
        )
        semantic_evidence = compute_family_evidence(
            lag_mean,
            self.semantic_evidence_source_index,
            self.semantic_evidence_source_valid,
        )
        return torch.stack((lag_mean, physical_evidence, semantic_evidence), dim=2)

    def compute_router_probability(
        self,
        context: torch.Tensor,
        family_evidence: torch.Tensor,
    ) -> torch.Tensor:
        return compute_router_probability(
            context,
            family_evidence,
            self.node_embedding.weight,
            self.horizon_embedding.weight,
            self.route_family_embedding.weight,
            self.family_available,
            self.router_scorer,
        )

    def router_probability(self, history: torch.Tensor) -> torch.Tensor:
        self._validate_history(history)
        context = self.context_encoder(history)
        lag_content = self.lag_content_encoder(history)
        evidence = self.encode_family_evidence(lag_content)
        return self.compute_router_probability(context, evidence)

    def _validate_history(self, history: torch.Tensor) -> None:
        if history.ndim != 4 or history.shape[1:] != (HISTORY_LENGTH, self.num_nodes, 1):
            raise ValueError(
                f"history must have shape [B,{HISTORY_LENGTH},{self.num_nodes},1], got {history.shape}"
            )

    def forward(
        self,
        history: torch.Tensor,
        *,
        family_top_p_enabled: bool = False,
    ) -> torch.Tensor:
        self._validate_history(history)

        context = self.context_encoder(history)
        lag_content = self.lag_content_encoder(history)
        family_evidence = self.encode_family_evidence(lag_content)
        router_probability = self.compute_router_probability(context, family_evidence)
        fine_tokens = self.encode_fine_tokens(lag_content)

        node_identity = self.node_projection(self.node_embedding.weight)
        horizon_identity = self.horizon_projection(self.horizon_embedding.weight)
        query_input = (
            context.unsqueeze(2)
            + node_identity.view(1, self.num_nodes, 1, HIDDEN_DIM)
            + horizon_identity.view(1, 1, FORECAST_HORIZON, HIDDEN_DIM)
        )
        query = self.wq(self.query_norm(query_input))

        self_family_identity = self.fine_family_projection(
            self.fine_family_embedding.weight[SELF_FAMILY_ID]
        )
        value_global = self.wv(fine_tokens)
        self_key_global = self.wk(
            fine_tokens + self_family_identity.view(1, 1, 1, HIDDEN_DIM)
        )
        self_output = compute_fine_family(
            query,
            self_key_global,
            value_global,
            self.self_flat_index,
            self.self_valid,
            self.self_prior,
            self.alpha,
        )
        physical_message = torch.zeros_like(self_output.message)
        if self.has_physical:
            physical_family_identity = self.fine_family_projection(
                self.fine_family_embedding.weight[PHYSICAL_FAMILY_ID]
            )
            physical_key_global = self.wk(
                fine_tokens + physical_family_identity.view(1, 1, 1, HIDDEN_DIM)
            )
            physical_output = compute_fine_family(
                query,
                physical_key_global,
                value_global,
                self.physical_flat_index,
                self.physical_valid,
                self.physical_prior,
                self.alpha,
            )
            physical_message = physical_output.message

        semantic_message = torch.zeros_like(self_output.message)
        if self.has_semantic:
            semantic_family_identity = self.fine_family_projection(
                self.fine_family_embedding.weight[SEMANTIC_FAMILY_ID]
            )
            semantic_key_global = self.wk(
                fine_tokens + semantic_family_identity.view(1, 1, 1, HIDDEN_DIM)
            )
            semantic_output = compute_fine_family(
                query,
                semantic_key_global,
                value_global,
                self.semantic_flat_index,
                self.semantic_valid,
                self.semantic_prior,
                self.alpha,
            )
            semantic_message = semantic_output.message

        family_weight = router_probability
        if family_top_p_enabled:
            _, family_weight = family_top_p(
                router_probability,
                self.family_available,
                rho=FAMILY_TOP_P_RHO,
            )
        message = dense_router_fusion(
            self_output.message,
            physical_message,
            semantic_message,
            family_weight,
        )

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

    def _validate_physical_candidates(self, metadata: PhysicalCandidateMetadata) -> None:
        candidate_count = metadata.flat_index.shape[1]
        expected_shape = (self.num_nodes, candidate_count)
        if any(
            tensor.shape != expected_shape
            for tensor in (
                metadata.source_index,
                metadata.lag_index,
                metadata.flat_index,
                metadata.valid,
                metadata.prior,
            )
        ):
            raise ValueError("Physical candidate metadata shapes are inconsistent")
        if candidate_count not in {0, 96}:
            raise ValueError(f"Physical candidates must contain 96 positions, got {candidate_count}")
        if metadata.source_index.dtype != torch.int64 or metadata.flat_index.dtype != torch.int64:
            raise TypeError("Physical source/flat index must be int64")
        if metadata.lag_index.dtype != torch.int64 or metadata.valid.dtype != torch.bool:
            raise TypeError("Physical lag index must be int64 and valid must be bool")
        if metadata.prior.dtype != torch.float32:
            raise TypeError("Physical prior must be float32")

    def _validate_semantic_candidates(self, metadata: SemanticCandidateMetadata) -> None:
        candidate_count = metadata.flat_index.shape[1]
        expected_shape = (self.num_nodes, candidate_count)
        if any(
            tensor.shape != expected_shape
            for tensor in (
                metadata.source_index,
                metadata.lag_index,
                metadata.flat_index,
                metadata.valid,
                metadata.prior,
            )
        ):
            raise ValueError("Semantic candidate metadata shapes are inconsistent")
        if candidate_count not in {0, 96}:
            raise ValueError(f"Semantic candidates must contain 96 positions, got {candidate_count}")
        if metadata.source_index.dtype != torch.int64 or metadata.flat_index.dtype != torch.int64:
            raise TypeError("Semantic source/flat index must be int64")
        if metadata.lag_index.dtype != torch.int64 or metadata.valid.dtype != torch.bool:
            raise TypeError("Semantic lag index must be int64 and valid must be bool")
        if metadata.prior.dtype != torch.float32:
            raise TypeError("Semantic prior must be float32")


# Phase 1 import compatibility without maintaining a second architecture.
SelfOnlyHiDFilter = HiDFilter
