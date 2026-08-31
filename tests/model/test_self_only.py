from __future__ import annotations

import importlib
import importlib.util

import torch

from hidfilter.runtime.phase0 import build_optimizer


def _model_module():
    assert importlib.util.find_spec("hidfilter.model") is not None
    return importlib.import_module("hidfilter.model")


def test_lag_mapping_latest_to_oldest_golden():
    history_to_lag_values = _model_module().history_to_lag_values
    history = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2, 1)

    lag_values = history_to_lag_values(history)

    assert lag_values.shape == (1, 2, 12, 1)
    assert lag_values[:, :, 0].equal(history[:, 11])
    assert lag_values[:, :, 11].equal(history[:, 0])


def test_context_shape_and_lag_content_exact_locality():
    module = _model_module()
    context_encoder = module.TargetContextEncoder()
    lag_encoder = module.LagContentEncoder()
    history = torch.randn(2, 12, 4, 1)
    changed = history.clone()
    changed[0, 3, 2, 0] += 7.0

    context = context_encoder(history)
    before = lag_encoder(history)
    after = lag_encoder(changed)
    changed_mask = (before != after).any(dim=-1)

    assert context.shape == (2, 4, 64)
    assert before.shape == (2, 4, 12, 64)
    assert changed_mask.sum().item() == 1
    assert changed_mask[0, 2, 8]


def test_node_identity_enters_only_fine_token():
    model = _model_module().SelfOnlyHiDFilter(num_nodes=3).eval()
    history = torch.randn(2, 12, 3, 1)
    lag_content_before = model.lag_content_encoder(history)
    fine_before = model.encode_fine_tokens(lag_content_before)

    with torch.no_grad():
        model.node_embedding.weight[1].add_(3.0)

    lag_content_after = model.lag_content_encoder(history)
    fine_after = model.encode_fine_tokens(lag_content_after)

    torch.testing.assert_close(lag_content_after, lag_content_before)
    torch.testing.assert_close(fine_after[:, 0], fine_before[:, 0])
    assert not torch.equal(fine_after[:, 1], fine_before[:, 1])
    torch.testing.assert_close(fine_after[:, 2], fine_before[:, 2])


def test_self_candidate_metadata_and_flat_gather_mapping():
    module = _model_module()
    metadata = module.build_self_candidate_metadata(num_nodes=4)
    global_values = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12, 1)

    gathered = module.gather_candidates(global_values, metadata.flat_index)

    assert metadata.source_index.dtype == torch.int64
    assert metadata.lag_index.dtype == torch.int64
    assert metadata.flat_index.dtype == torch.int64
    assert metadata.valid.dtype == torch.bool
    assert metadata.prior.dtype == torch.float32
    assert metadata.source_index.tolist() == [[0] * 12, [1] * 12, [2] * 12, [3] * 12]
    assert metadata.lag_index.tolist() == [list(range(12))] * 4
    assert metadata.flat_index[3].tolist() == list(range(36, 48))
    assert metadata.valid.all()
    assert metadata.prior.equal(torch.ones(4, 12))
    torch.testing.assert_close(gathered[:, 3], global_values[:, 3])


def test_shared_projection_and_global_alpha_state_schema():
    model = _model_module().SelfOnlyHiDFilter(num_nodes=5)
    linear_names = {name for name, child in model.named_modules() if isinstance(child, torch.nn.Linear)}
    alpha_parameters = [(name, value) for name, value in model.named_parameters() if "alpha" in name]

    assert {"wq", "wk", "wv"}.issubset(linear_names)
    assert not any(name.startswith(("self_q", "self_k", "self_v", "physical", "semantic")) for name in linear_names)
    assert len(alpha_parameters) == 1
    assert alpha_parameters[0][0] == "alpha_raw"
    torch.testing.assert_close(model.alpha, torch.tensor(1.0), rtol=0.0, atol=1e-6)
    assert model.fine_family_embedding.weight.shape == (3, 16)


def test_prediction_shape_static_metadata_and_zero_residual_path():
    model = _model_module().SelfOnlyHiDFilter(num_nodes=5).eval()
    history = torch.randn(3, 12, 5, 1)
    flat_index_pointer = model.self_flat_index.data_ptr()
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()

    prediction = model(history)
    expected = history[:, -1:, :, :].expand(-1, 12, -1, -1)

    assert prediction.shape == (3, 12, 5, 1)
    torch.testing.assert_close(prediction, expected)
    assert model.self_flat_index.data_ptr() == flat_index_pointer
    assert "self_flat_index" in dict(model.named_buffers())


def test_forward_backward_reaches_all_core_parameters_except_alpha():
    model = _model_module().SelfOnlyHiDFilter(num_nodes=4).train()
    history = torch.randn(3, 12, 4, 1)
    target = torch.randn(3, 12, 4, 1)

    prediction = model(history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    required_prefixes = (
        "context_encoder",
        "lag_content_encoder",
        "node_embedding",
        "horizon_embedding",
        "fine_family_embedding",
        "node_projection",
        "horizon_projection",
        "fine_family_projection",
        "wq",
        "wk",
        "wv",
        "decoder",
    )
    for prefix in required_prefixes:
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name == prefix or name.startswith(prefix + ".")
        ]
        assert gradients, prefix
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), prefix
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0.0, prefix
    assert model.alpha_raw.grad is not None
    assert model.alpha_raw.grad.item() == 0.0


def test_fixed_batch_learning_reduces_mae():
    torch.manual_seed(7)
    model = _model_module().SelfOnlyHiDFilter(num_nodes=3).eval()
    history = torch.randn(4, 12, 3, 1)
    target = history[:, -1:, :, :].expand(-1, 12, -1, -1) + 0.75
    optimizer = build_optimizer(model)

    with torch.no_grad():
        initial = torch.nn.functional.l1_loss(model(history), target).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(history), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.l1_loss(model(history), target).item()

    assert final < initial * 0.5
