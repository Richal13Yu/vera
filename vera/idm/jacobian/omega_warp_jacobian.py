"""Train VERA's action Jacobian against a frozen Omega-Warp correspondence teacher."""

from dataclasses import dataclass, field
import time
from typing import Literal

import torch
import torch.nn.functional as F

from vera.idm.registry import register_algorithm
from .image_jacobian import ImageJacobian, ImageJacobianCfg
from .models.base import InputCommand, InputObservation
from .warp_teacher import OmegaWarpTeacher, WarpTeacherCfg
from .warp_supervision import (
    masked_flow_gradient_loss, masked_flow_loss, normalize_teacher_flow,
    recover_action, weighted_mean,
)


@dataclass
class OmegaWarpJacobianCfg(ImageJacobianCfg):
    name: Literal["omega_warp_jacobian"]
    warp_teacher: WarpTeacherCfg = field(default_factory=WarpTeacherCfg)


@register_algorithm("omega_warp_jacobian", cfg_cls=OmegaWarpJacobianCfg)
class OmegaWarpJacobian(ImageJacobian):
    """Same serving model as ImageJacobian; the teacher is only used for losses.

    The teacher holder is a plain Python object: no teacher weights in VERA's
    optimizer or state_dict. Each training rank lazily creates its own frozen copy.
    """

    def __init__(self, cfg: OmegaWarpJacobianCfg):
        if cfg.supervision != "flow":
            raise ValueError("OmegaWarpJacobian currently supports supervision=flow")
        if cfg.view_flow_balance_mode != "none":
            raise ValueError("Use view_flow_balance_mode=none; warp uses valid confidence weights")
        super().__init__(cfg)
        self.warp_teacher = OmegaWarpTeacher(cfg.warp_teacher)

    def _compute_loss(self, batch, namespace="training", *, collect_diagnostics=True):
        timing = {}
        start = time.perf_counter()
        rgb = batch["rgb"]
        if "rgb_next" not in batch:
            raise ValueError("Omega warp supervision requires dataset.load_rgb_next=true")
        flow_px, weights = self.warp_teacher(rgb, batch["rgb_next"])
        flow = normalize_teacher_flow(flow_px, self.dataset_metadata)
        # Also used by the inherited validation visualizer. These are pseudo labels.
        batch["flow"] = flow
        batch["flow_valid"] = weights > 0
        timing["warp_teacher_s"] = time.perf_counter() - start

        joint = rgb.ndim == 6 and self._model_supports_joint_multiview()
        model_rgb, model_du, target, _, loss_du, view_ids = self._prepare_model_inputs(
            rgb=rgb, du=batch["du"], flow_gt=flow, joint_multiview=joint,
        )
        start = time.perf_counter()
        output = self.model(InputObservation(rgb=model_rgb, view_ids=view_ids),
                            InputCommand(du=model_du))
        if joint:
            output = self._flatten_joint_model_output(output)
        timing["model_forward_s"] = time.perf_counter() - start
        pred = output.optical_flow.float()
        target = target.float()
        if pred.shape != target.shape or pred.shape[-3] != 2:
            raise ValueError(f"Expected 2D warp motion {target.shape}, got {pred.shape}")
        weights = weights.reshape(-1, 1, *pred.shape[-2:]).detach().float()
        loss_weights = weights
        if self.cfg.motion_aware_flow_weighting:
            moving = target.norm(dim=-3, keepdim=True) > self.cfg.motion_aware_flow_threshold
            loss_weights = weights * torch.where(moving, self.cfg.motion_aware_moving_weight, 1.)

        loss_metrics, diagnostics = {}, {}
        num_views = rgb.shape[2] if rgb.ndim == 6 else 1
        per_view, view_present = [], []
        for v in range(num_views):
            # _prepare_model_inputs flattens B,T,V with V varying fastest.
            value = masked_flow_loss(pred[v::num_views], target[v::num_views],
                                     loss_weights[v::num_views], kind=self.cfg.flow_loss,
                                     eps=self.cfg.flow_charbonnier_eps)
            per_view.append(value)
            view_present.append((loss_weights[v::num_views].sum() > 0).float())
            if self.cfg.log_flow_per_view:
                view_name = self._get_view_names(num_views)[v].replace("/", "_")
                loss_metrics[f"flow/view_{view_name}"] = value
        if self.cfg.view_balanced_flow_loss:
            flow_loss = weighted_mean(torch.stack(per_view), torch.stack(view_present))
        else:
            flow_loss = masked_flow_loss(pred, target, loss_weights, kind=self.cfg.flow_loss,
                                         eps=self.cfg.flow_charbonnier_eps)
        losses = {"flow": flow_loss}
        if self.cfg.flow_gradient_weight > 0:
            losses["flow_gradient"] = self.cfg.flow_gradient_weight * masked_flow_gradient_loss(
                pred, target, weights, eps=self.cfg.flow_charbonnier_eps)
        if self.cfg.jacobian_tv_weight > 0:
            losses["jacobian_tv"] = self.cfg.jacobian_tv_weight * self._jacobian_tv_loss(
                output.jacobian, model_rgb)
        if self.cfg.predict_uncertainty and output.flow_confidence is not None:
            confidence = output.flow_confidence.float().clamp_min(1e-8)
            if self.cfg.pmap_confidence_clamp > 0:
                confidence = confidence.clamp(max=self.cfg.pmap_confidence_clamp)
            delta = torch.where(weights > 0, pred - target, torch.zeros_like(pred))
            residual = (delta.square() + self.cfg.flow_charbonnier_eps**2).sqrt().mean(-3, keepdim=True)
            losses["flow_aleatoric"] = weighted_mean(
                confidence * residual - self.cfg.pmap_uncertainty_weight * confidence.log(), weights)
        # Evaluate action recovery even when its training loss is disabled.
        if self.cfg.inverse_action_weight > 0 or (collect_diagnostics and namespace != "training"):
            du_hat = recover_action(output.jacobian, target, weights,
                                    damping=self.cfg.inverse_action_damping)
            valid_sample = weights.flatten(1).sum(-1) > 0
            inverse_error = F.smooth_l1_loss(du_hat, loss_du.float(), reduction="none").mean(-1)
            if self.cfg.inverse_action_weight > 0:
                losses["inverse_action"] = self.cfg.inverse_action_weight * weighted_mean(
                    inverse_error, valid_sample.float())
            diagnostics["du_hat_flow_mse"] = weighted_mean(
                (du_hat.detach() - loss_du).square().mean(-1), valid_sample.float())
        if collect_diagnostics:
            diagnostics["warp_valid_ratio"] = (weights > 0).float().mean()
            diagnostics["warp_certainty_mean"] = weighted_mean(weights, (weights > 0).float())
            raw_pred = pred / normalize_teacher_flow(torch.ones_like(pred), self.dataset_metadata)
            raw_target = flow_px.reshape_as(pred)
            diagnostics["warp_epe_px"] = weighted_mean(
                (raw_pred.detach() - raw_target).norm(dim=-3, keepdim=True), weights)
        total = torch.stack(list(losses.values())).sum()
        return losses, loss_metrics, total, output, diagnostics, timing
