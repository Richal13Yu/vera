"""Masked motion supervision; all displacements use the input image's pixels."""

from collections.abc import Mapping

import torch
from torch import Tensor


def normalize_teacher_flow(flow_px: Tensor, metadata: Mapping) -> Tensor:
    """Apply VERA's zero-preserving flow scale, after any image resizing.

    A Jacobian predicts displacement with zero intercept, so asymmetric affine
    normalization is deliberately rejected. No full-resolution scale is inferred.
    """
    mode = metadata.get("flow_normalization_mode", "scale")
    if mode == "symmetric_percentile":
        scale = torch.as_tensor(metadata["oflow_abs_scale"], device=flow_px.device,
                                dtype=flow_px.dtype)
        if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("oflow_abs_scale must contain two finite positive values")
        shape = [1] * flow_px.ndim
        shape[-3] = 2
        return flow_px / (scale.reshape(shape) + 1e-8)
    if mode == "scale":
        scale = float(metadata.get("oflow_scale", 1.0))
        if not 0 < scale < float("inf"):
            raise ValueError("oflow_scale must be finite and positive")
        return flow_px * scale
    raise ValueError("Warp supervision requires scale or symmetric_percentile flow normalization")


def weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    """Mean over valid entries only, including a differentiable empty-mask zero."""
    weights = weights.expand_as(values)
    values = torch.where(weights > 0, values, torch.zeros_like(values))
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def masked_flow_loss(pred: Tensor, target: Tensor, weights: Tensor,
                     *, kind: str = "charbonnier", eps: float = 1e-3) -> Tensor:
    # Remove invalid operands before nonlinear operations to avoid NaN gradients.
    delta = torch.where(weights > 0, pred - target, torch.zeros_like(pred)).float()
    if kind == "mse":
        residual = delta.square()
    elif kind == "charbonnier":
        residual = torch.sqrt(delta.square() + eps * eps)
    else:
        raise ValueError(f"Unsupported flow loss: {kind}")
    return weighted_mean(residual, weights)


def masked_flow_gradient_loss(pred: Tensor, target: Tensor, weights: Tensor,
                              *, eps: float = 1e-3) -> Tensor:
    total = pred[..., :0].sum()
    for axis in (-1, -2):
        if pred.shape[axis] < 2:
            continue
        left = [slice(None)] * pred.ndim
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        edge_weights = torch.minimum(weights[left], weights[right])
        total = total + masked_flow_loss(
            pred[right] - pred[left], target[right] - target[left], edge_weights, eps=eps
        )
    return total


def recover_action(jacobian: Tensor, flow: Tensor, weights: Tensor,
                   *, damping: float) -> Tensor:
    """Weighted damped least squares in float32; J is [N,A,2,H,W]."""
    if damping <= 0:
        raise ValueError("inverse_action_damping must be positive")
    with torch.autocast(device_type=jacobian.device.type, enabled=False):
        matrix = jacobian.float().flatten(2).transpose(1, 2)
        motion = flow.float().flatten(1)
        rows = weights.expand_as(flow).float().flatten(1)
        matrix = torch.where(rows[..., None] > 0, matrix, 0)
        motion = torch.where(rows > 0, motion, 0)
        jtj = matrix.transpose(1, 2) @ (matrix * rows[..., None])
        jty = matrix.transpose(1, 2) @ (motion * rows).unsqueeze(-1)
        ridge = damping * jtj.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1)
        eye = torch.eye(jtj.shape[-1], device=jtj.device, dtype=jtj.dtype)
        return torch.linalg.solve(jtj + ridge[:, None, None] * eye, jty).squeeze(-1)
