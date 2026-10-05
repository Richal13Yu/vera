"""Numerical contracts for teacher motion and masked Jacobian supervision."""

import importlib.util
from pathlib import Path

import pytest
import torch


# Avoid importing vera.idm, which registers the optional heavy training stack.
_path = Path(__file__).resolve().parents[1] / "vera/idm/jacobian/warp_supervision.py"
_spec = importlib.util.spec_from_file_location("warp_supervision_under_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
normalize_teacher_flow = _module.normalize_teacher_flow
masked_flow_loss = _module.masked_flow_loss
masked_flow_gradient_loss = _module.masked_flow_gradient_loss
recover_action = _module.recover_action


@pytest.mark.parametrize("shape", [(3, 2, 2, 4), (2, 3, 2, 2, 2, 4)])
@pytest.mark.parametrize(
    "metadata,channel_multipliers",
    [
        ({"flow_normalization_mode": "scale", "oflow_scale": 0.25}, [0.25, 0.25]),
        ({"flow_normalization_mode": "symmetric_percentile", "oflow_abs_scale": [2., 4.]},
         [0.5, 0.25]),
    ],
)
def test_teacher_normalization_preserves_zero_and_scales_channels(shape, metadata, channel_multipliers):
    zeros = torch.zeros(shape)
    torch.testing.assert_close(normalize_teacher_flow(zeros, metadata), zeros, rtol=0, atol=0)
    flow = torch.ones(shape)
    flow[..., 0, :, :] = 2.
    flow[..., 1, :, :] = -8.
    expected = flow.clone()
    expected[..., 0, :, :] *= channel_multipliers[0]
    expected[..., 1, :, :] *= channel_multipliers[1]
    torch.testing.assert_close(normalize_teacher_flow(flow, metadata), expected)


@pytest.mark.parametrize(
    "metadata",
    [
        {"flow_normalization_mode": "percentile_minmax"},
        {"flow_normalization_mode": "scale", "oflow_scale": 0.},
        {"flow_normalization_mode": "scale", "oflow_scale": float("nan")},
        {"flow_normalization_mode": "symmetric_percentile", "oflow_abs_scale": [1.]},
        {"flow_normalization_mode": "symmetric_percentile", "oflow_abs_scale": [1., -1.]},
    ],
)
def test_teacher_normalization_rejects_affine_or_invalid_scales(metadata):
    with pytest.raises(ValueError):
        normalize_teacher_flow(torch.ones(1, 2, 2, 2), metadata)


@pytest.mark.parametrize("kind", ["mse", "charbonnier"])
def test_nan_target_at_invalid_pixels_has_zero_gradient(kind):
    pred = torch.zeros(1, 2, 1, 2, requires_grad=True)
    target = torch.tensor([[[[1., float("nan")]], [[3., float("nan")]]]])
    weights = torch.tensor([[[[0.5, 0.]]]])
    loss = masked_flow_loss(pred, target, weights, kind=kind)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad[..., 1], torch.zeros_like(pred.grad[..., 1]))
    assert (pred.grad[..., 0] < 0).all()
    if kind == "mse":
        torch.testing.assert_close(loss.detach(), torch.tensor(5.))
        torch.testing.assert_close(pred.grad[..., 0], torch.tensor([[[-1.], [-3.]]]))


def translation_jacobian(batch=1, height=2, width=3):
    jacobian = torch.zeros(batch, 2, 2, height, width)
    jacobian[:, 0, 0] = 1.
    jacobian[:, 1, 1] = 1.
    return jacobian


def test_empty_masks_produce_finite_zero_losses_action_and_gradients():
    pred = torch.randn(2, 2, 2, 3, requires_grad=True)
    target = torch.full_like(pred, float("nan"))
    weights = torch.zeros(2, 1, 2, 3)
    jacobian = translation_jacobian(batch=2).requires_grad_()
    flow_loss = masked_flow_loss(pred, target, weights)
    gradient_loss = masked_flow_gradient_loss(pred, target, weights)
    action = recover_action(jacobian, target, weights, damping=1e-3)
    torch.testing.assert_close(flow_loss, torch.tensor(0.))
    torch.testing.assert_close(gradient_loss, torch.tensor(0.))
    torch.testing.assert_close(action, torch.zeros(2, 2))
    (flow_loss + gradient_loss + action.square().sum()).backward()
    for grad in (pred.grad, jacobian.grad):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_known_translation_jacobian_recovers_actions():
    jacobian = translation_jacobian(batch=2)
    action = torch.tensor([[2.5, -1.75], [-3., 0.5]])
    flow = torch.einsum("nashw,na->nshw", jacobian, action)
    weights = torch.ones(2, 1, 2, 3)
    recovered = recover_action(jacobian, flow, weights, damping=1e-6)
    torch.testing.assert_close(recovered, action, rtol=1e-5, atol=1e-5)


def test_known_dense_jacobian_recovers_coupled_actions():
    jacobian = torch.tensor([
        [[[1., 0.], [2., -1.]], [[0., 1.], [1., 2.]]],
        [[[0., 2.], [-1., 1.]], [[1., 0.], [3., -1.]]],
        [[[2., -1.], [1., 0.]], [[-1., 2.], [0., 1.]]],
    ]).unsqueeze(0)
    action = torch.tensor([[0.7, -1.4, 2.2]])
    flow = torch.einsum("nashw,na->nshw", jacobian, action)
    weights = torch.tensor([[[[1., 0.5], [0.25, 0.75]]]])
    recovered = recover_action(jacobian, flow, weights, damping=1e-6)
    torch.testing.assert_close(recovered, action, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("invalid_value", [1e20, float("nan")])
def test_invalid_jacobian_and_flow_cannot_pollute_inverse_action(invalid_value):
    jacobian = translation_jacobian()
    action = torch.tensor([[2., -1.]])
    flow = torch.einsum("nashw,na->nshw", jacobian, action)
    weights = torch.ones(1, 1, 2, 3)
    weights[..., -1] = 0.
    reference = recover_action(jacobian, flow, weights, damping=1e-3)
    jacobian[..., -1] = invalid_value
    flow[..., -1] = -invalid_value
    jacobian.requires_grad_()
    recovered = recover_action(jacobian, flow, weights, damping=1e-3)
    torch.testing.assert_close(recovered, reference)
    recovered.sum().backward()
    assert torch.isfinite(jacobian.grad).all()
    torch.testing.assert_close(jacobian.grad[..., -1], torch.zeros_like(jacobian.grad[..., -1]))


def test_gradient_loss_excludes_edges_crossing_invalid_mask():
    pred = torch.tensor([[[[0., 1., 1e20]], [[0., 1., -1e20]]]], requires_grad=True)
    target = torch.tensor([[[[0., 0., float("nan")]], [[0., 0., float("nan")]]]])
    weights = torch.tensor([[[[1., 1., 0.]]]])
    loss = masked_flow_gradient_loss(pred, target, weights, eps=1e-3)
    # Only the first horizontal edge is valid. Height 1 has no vertical edges.
    torch.testing.assert_close(loss, torch.sqrt(torch.tensor(1. + 1e-6)))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[..., :2].abs() > 0).all()
    torch.testing.assert_close(pred.grad[..., 2], torch.zeros_like(pred.grad[..., 2]))


def test_isolated_valid_pixel_has_no_valid_gradient_edges():
    pred = torch.randn(1, 2, 3, 3, requires_grad=True)
    target = torch.full_like(pred, float("nan"))
    target[..., 1, 1] = 8.
    weights = torch.zeros(1, 1, 3, 3)
    weights[..., 1, 1] = 1.
    loss = masked_flow_gradient_loss(pred, target, weights)
    torch.testing.assert_close(loss, torch.tensor(0.))
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred.grad))
