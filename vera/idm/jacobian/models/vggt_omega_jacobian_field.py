"""Trainable Omega image encoder with VERA's action-conditioned Jacobian head.

Only the current observation belongs in ``InputObservation.rgb``. Its optional
view axis contains simultaneous cameras, never the future target images used
by the separate warp teacher. Omega's attention mixes that entire axis.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from .registry import register_model
from .vggt_jacobian_field import VggtJacobianField, VggtJacobianFieldCfg


@dataclass
class VggtOmegaJacobianFieldCfg(VggtJacobianFieldCfg):
    name: Literal["vggt_omega_jacobian"]
    patch_size: int = 16
    low_memory_init: bool = True


@register_model("vggt_omega_jacobian", cfg_cls=VggtOmegaJacobianFieldCfg)
class VggtOmegaJacobianField(VggtJacobianField):
    """Reuse VERA's DPT/Jacobian contract with the native Omega aggregator.

    Unlike the warp teacher's pair-token adapter, this encoder has no forced
    ``no_grad`` context. ``freeze_aggregator`` controls whether it is trained.
    The inherited decoder handles padding/cropping, simultaneous views,
    uncertainty, and the pixel displacement ``flow = J @ du``.
    """

    cfg: VggtOmegaJacobianFieldCfg

    def __init__(self, model_cfg: VggtOmegaJacobianFieldCfg):
        if model_cfg.pretrained_model_id:
            raise ValueError(
                "Omega requires a local checkpoint_path; pretrained_model_id is unsupported."
            )
        if not model_cfg.strict_checkpoint_load:
            raise ValueError("Omega aggregator checkpoints must be loaded strictly.")
        if int(model_cfg.patch_size) != 16:
            raise ValueError("The released Omega aggregator uses patch_size=16.")
        layers = tuple(int(index) for index in model_cfg.decoder_intermediate_layer_idx)
        if (
            len(layers) != 4
            or len(set(layers)) != 4
            or tuple(sorted(layers)) != layers
            or any(index < 0 or index >= 24 for index in layers)
        ):
            raise ValueError("Omega's DPT decoder needs four increasing layer indices in [0, 23].")
        if len(model_cfg.decoder_out_channels) != 4:
            raise ValueError("Omega's DPT decoder needs four decoder_out_channels values.")
        if model_cfg.checkpoint_path and not Path(model_cfg.checkpoint_path).expanduser().is_file():
            raise FileNotFoundError(f"Omega checkpoint not found: {model_cfg.checkpoint_path}")

        super().__init__(model_cfg)
        self.set_aggregator_trainable(not model_cfg.freeze_aggregator)

    def _build_vggt_model(self, model_cfg: VggtOmegaJacobianFieldCfg) -> nn.Module:
        # Optional dependency: importing VERA's original model registry does not
        # require vggt-omega-warp to be installed.
        try:
            from vggt_omega.models.aggregator import Aggregator
            from vggt_omega.models.layers.vision_transformer import init_weights_vit
        except ImportError as exc:
            raise ImportError(
                "Omega Jacobian requires the local vggt-omega-warp package. "
                "Install it with `pip install -e ./vggt-omega-warp` from the VERA root."
            ) from exc

        use_meta = bool(model_cfg.checkpoint_path and model_cfg.low_memory_init)
        with torch.device("meta") if use_meta else nullcontext():
            aggregator = Aggregator(
                patch_size=int(model_cfg.patch_size),
                embed_dim=int(model_cfg.embed_dim),
                cached_layer_indices=tuple(model_cfg.decoder_intermediate_layer_idx),
            )
        if not model_cfg.checkpoint_path:
            # Omega's released constructor expects a checkpoint: its attention
            # masks start as NaN and its LayerScale parameters as torch.empty.
            # Initialize only these otherwise-uninitialized attention stacks
            # when explicitly constructing a student from scratch.
            aggregator.frame_blocks.apply(init_weights_vit)
            aggregator.inter_frame_blocks.apply(init_weights_vit)
        # Retain the parent's vggt.aggregator parameter names so its optimizer
        # groups and forward implementation remain usable without modification.
        model = nn.Module()
        model.add_module("aggregator", aggregator)
        return model

    def _strip_unused_heads(self) -> None:
        # Only the aggregator is constructed; camera/depth heads never exist.
        pass

    def _load_pretrained_checkpoint(self, model_cfg: VggtOmegaJacobianFieldCfg) -> None:
        self.checkpoint_report = None
        if not model_cfg.checkpoint_path:
            return
        try:
            from vggt_omega_warp.backbone import (
                _materialize_vggt_nonpersistent_buffers,
                load_aggregator_checkpoint,
            )
        except ImportError as exc:
            raise ImportError("Omega checkpoint loading requires vggt-omega-warp.") from exc

        # Meta construction avoids an extra ~3.4 GiB initialized encoder copy.
        # The official loader verifies all keys/shapes, supports full-model or
        # aggregator-only PT/safetensors, and assigns parameters when on meta.
        _materialize_vggt_nonpersistent_buffers(self.vggt.aggregator, "cpu")
        self.checkpoint_report = load_aggregator_checkpoint(
            self.vggt.aggregator,
            Path(model_cfg.checkpoint_path).expanduser(),
            map_location="cpu",
        )

    def set_aggregator_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the encoder; rebuild the optimizer after changing it."""
        self.cfg.freeze_aggregator = not bool(trainable)
        self.vggt.aggregator.requires_grad_(bool(trainable))
        self.vggt.aggregator.train(bool(trainable) and self.training)

    def train(self, mode: bool = True) -> VggtOmegaJacobianField:
        super().train(mode)
        if self.cfg.freeze_aggregator:
            self.vggt.aggregator.eval()
        return self

    def get_optimizer_param_groups(
        self, *, base_lr: float
    ) -> list[nn.Parameter] | list[dict[str, object]]:
        if float(self.cfg.aggregator_lr_multiplier) == 1.0:
            return [parameter for parameter in self.parameters() if parameter.requires_grad]
        return super().get_optimizer_param_groups(base_lr=base_lr)
