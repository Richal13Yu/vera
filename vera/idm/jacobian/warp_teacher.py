"""Frozen Omega-Warp correspondence targets for the image Jacobian.

This module only imports PyTorch and the standard library. The optional
``vggt_omega_warp`` package and its checkpoints are loaded on the first call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass
class WarpTeacherCfg:
    config_path: str | None = None
    backbone_checkpoint: str | None = None
    head_checkpoint: str | None = None
    precision: str = "bf16"
    pair_batch_size: int = 1
    certainty_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("warp teacher precision must be fp32, bf16, or fp16")
        if type(self.pair_batch_size) is not int or self.pair_batch_size < 1:
            raise ValueError("warp teacher pair_batch_size must be a positive integer")
        _validate_threshold(self.certainty_threshold)


def _validate_threshold(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("warp teacher certainty_threshold must be finite and in [0,1]")


@torch.inference_mode(False)
@torch.no_grad()
def warp_output_to_targets(
    output: Mapping[str, Tensor],
    image_hw: tuple[int, int],
    certainty_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Convert native Omega-Warp output to finite, detached pixel-flow targets.

    ``flow_px`` is source-to-target displacement in ``(x,y)`` pixel units.
    Returns ``[N,2,H,W]`` flow and ``[N,1,H,W]`` confidence weights. Geometric
    validity uses target pixel centres, not the wider bilinear sampling domain.
    Masked labels are zeroed so downstream multiplication cannot retain NaNs.
    """
    _validate_threshold(certainty_threshold)
    flow, logits = output["flow_px"], output["certainty_logits"]
    height, width = image_hw
    if flow.ndim != 4 or tuple(flow.shape[1:]) != (height, width, 2):
        raise ValueError("teacher flow_px must have shape [N,H,W,2] at input resolution")
    if logits.shape != flow.shape[:-1] or logits.device != flow.device:
        raise ValueError("teacher certainty_logits must have shape [N,H,W] on the flow device")
    if not flow.is_floating_point() or not logits.is_floating_point():
        raise TypeError("teacher flow and certainty logits must be floating point")
    flow, logits = flow.detach().float(), logits.detach().float()
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=torch.float32),
        torch.arange(width, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    target_xy = flow + torch.stack((x, y), dim=-1)
    probability = logits.sigmoid()
    tolerance = 1e-4  # Omega-Warp correspondence_in_frame tolerance in pixels.
    valid = (
        torch.isfinite(flow).all(dim=-1)
        & torch.isfinite(logits)
        & (target_xy[..., 0] >= -tolerance)
        & (target_xy[..., 0] <= width - 1 + tolerance)
        & (target_xy[..., 1] >= -tolerance)
        & (target_xy[..., 1] <= height - 1 + tolerance)
        & (probability >= certainty_threshold)
    )
    flow = torch.where(valid[..., None], flow, 0.0)
    weights = torch.where(valid, probability, 0.0)
    return flow.permute(0, 3, 1, 2).contiguous(), weights[:, None].contiguous()


class OmegaWarpTeacher:
    """Lazy teacher kept outside the student's module tree and optimizer.

    Inputs are aligned ``[B,T,V,3,H,W]`` or ``[B,T,3,H,W]`` RGB pairs in
    ``[0,1]``. Flattening only the batch/time/view prefix pairs each observation
    with its next observation from the same camera. Outputs preserve that
    prefix, with channel dimensions 2 (pixel flow) and 1 (confidence weight).
    """

    def __init__(self, cfg: WarpTeacherCfg | Mapping[str, Any]) -> None:
        self.cfg = WarpTeacherCfg(**dict(cfg)) if isinstance(cfg, Mapping) else cfg
        self.model: torch.nn.Module | None = None
        self.provenance: dict[str, Any] | None = None
        self._model_device: torch.device | None = None

    @staticmethod
    def _require_file(value: str | None, name: str) -> Path:
        if not value:
            raise ValueError(f"Omega-Warp teacher requires warp_teacher.{name}")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Omega-Warp teacher {name} not found: {path}")
        return path

    def _ensure_model(self, device: torch.device) -> torch.nn.Module:
        if self.model is None:
            config_path = self._require_file(self.cfg.config_path, "config_path")
            backbone_path = self._require_file(
                self.cfg.backbone_checkpoint, "backbone_checkpoint"
            )
            head_path = self._require_file(self.cfg.head_checkpoint, "head_checkpoint")
            try:
                from vggt_omega_warp.checkpoint import load_checkpoint
                from vggt_omega_warp.cli.common import (
                    build_model,
                    load_toml,
                    validate_config_sections,
                )
                from vggt_omega_warp.provenance import (
                    assert_local_artifact_unchanged,
                    backbone_artifact_metadata,
                    expected_backbone_metadata,
                    local_artifact_stat_identity,
                    model_implementation_provenance,
                    stable_file_fingerprint,
                )
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "Omega-Warp teacher requires the local vggt-omega-warp package "
                    "and its dependencies; install it with pip install -e ./vggt-omega-warp"
                ) from exc

            config_identity, config_stat = stable_file_fingerprint(config_path)
            config = load_toml(config_path)
            validate_config_sections(config, command="infer")
            if config.get("model", {}).get("kind", "vggt_omega") != "vggt_omega":
                raise ValueError("OmegaWarpTeacher requires model.kind='vggt_omega'")
            config["implementation"] = model_implementation_provenance(config)
            # Teacher inference can move to another machine/runtime. Retain
            # exact source compatibility without requiring the training GPU,
            # local Git status, or deterministic-runtime flags to match.
            expected_implementation = {
                key: value for key, value in config["implementation"].items()
                if key not in {"runtime", "git"}
            }
            config.setdefault("model", {})["backbone_checkpoint"] = str(backbone_path)
            head_identity, head_stat = stable_file_fingerprint(head_path)
            backbone_metadata = backbone_artifact_metadata(
                config, backbone_path, include_hash=True
            )
            backbone_stat = local_artifact_stat_identity(backbone_path)
            model = build_model(
                config, backbone_checkpoint=backbone_path, map_location="cpu"
            )
            payload = load_checkpoint(
                head_path,
                model,
                map_location="cpu",
                strict=True,
                restore_rng=False,
                expected_backbone=expected_backbone_metadata(backbone_metadata),
                expected_implementation=expected_implementation,
            )
            assert_local_artifact_unchanged(config_path, config_stat)
            assert_local_artifact_unchanged(head_path, head_stat)
            assert_local_artifact_unchanged(backbone_path, backbone_stat)
            # Keep only metadata: the training payload also contains optimizer
            # tensors which are unnecessary for this frozen teacher.
            self.provenance = {
                "config": config_identity,
                "backbone": backbone_metadata,
                "head": head_identity,
                "global_step": payload.get("global_step"),
                "epoch": payload.get("epoch"),
                "feature_extraction_contract": payload.get("feature_extraction_contract"),
                "implementation": config["implementation"],
            }
            self.model = model
        if self._model_device != device:
            self.model.to(device).eval().requires_grad_(False)
            self._model_device = device
        return self.model

    def _autocast_context(self, device: torch.device):
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Omega-Warp teacher supports CPU and CUDA devices")
        if self.cfg.precision == "fp32":
            # Override an enclosing student autocast context as well.
            return torch.autocast(device_type=device.type, enabled=False)
        dtype = torch.bfloat16 if self.cfg.precision == "bf16" else torch.float16
        if device.type == "cpu" and dtype == torch.float16:
            raise ValueError("CPU Omega-Warp teacher supports fp32/bf16, not fp16")
        if device.type == "cuda" and dtype == torch.bfloat16:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("this CUDA device requires fp16 or fp32 teacher precision")
        return torch.autocast(device_type=device.type, dtype=dtype)

    @torch.inference_mode(False)
    @torch.no_grad()
    def __call__(self, rgb: Tensor, rgb_next: Tensor) -> tuple[Tensor, Tensor]:
        if rgb.ndim not in {5, 6} or rgb.shape[-3] != 3:
            raise ValueError("teacher RGB must have shape [B,T,3,H,W] or [B,T,V,3,H,W]")
        if rgb.shape != rgb_next.shape or rgb.device != rgb_next.device:
            raise ValueError("teacher RGB pairs must have identical shapes and devices")
        if min(rgb.shape) < 1 or min(rgb.shape[-2:]) < 16:
            raise ValueError("teacher requires nonempty RGB pairs and H,W >= 16")
        for image in (rgb, rgb_next):
            if not image.is_floating_point():
                raise TypeError("teacher RGB must be floating point in [0,1]")
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("teacher RGB must be finite and in [0,1]")

        prefix, height, width = rgb.shape[:-3], rgb.shape[-2], rgb.shape[-1]
        source = rgb.reshape(-1, 3, height, width)
        target = rgb_next.reshape(-1, 3, height, width)
        model = self._ensure_model(rgb.device)
        flows, weights = [], []
        for start in range(0, source.shape[0], self.cfg.pair_batch_size):
            end = start + self.cfg.pair_batch_size
            # Both the RGB refinement head and its Aggregator adapter require
            # [-1,1]. The adapter itself converts back to [0,1] for Omega.
            source_pair = source[start:end].float().mul(2).sub(1)
            target_pair = target[start:end].float().mul(2).sub(1)
            with self._autocast_context(rgb.device):
                output = model(source_pair, target_pair)
            flow, weight = warp_output_to_targets(
                output, (height, width), self.cfg.certainty_threshold
            )
            if flow.shape[0] != source_pair.shape[0] or flow.device != rgb.device:
                raise ValueError("teacher output batch size/device differs from its input")
            flows.append(flow)
            weights.append(weight)
        return (
            torch.cat(flows).reshape(*prefix, 2, height, width),
            torch.cat(weights).reshape(*prefix, 1, height, width),
        )


__all__ = ["WarpTeacherCfg", "OmegaWarpTeacher", "warp_output_to_targets"]
