"""Teacher contract tests without importing VERA's optional IDM stack."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from torch import nn


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "vera/idm/jacobian/warp_teacher.py"
)
_SPEC = importlib.util.spec_from_file_location("_vera_warp_teacher_test", _MODULE_PATH)
teacher_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = teacher_module
_SPEC.loader.exec_module(teacher_module)
WarpTeacherCfg = teacher_module.WarpTeacherCfg
OmegaWarpTeacher = teacher_module.OmegaWarpTeacher
warp_output_to_targets = teacher_module.warp_output_to_targets


class FakeOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_pairs = []

    def forward(self, source, target):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.seen_pairs.append((source.clone(), target.clone()))
        batch, _, height, width = source.shape
        # Input delta 0.1 becomes 0.2 after [-1,1] normalization, then a
        # +1 pixel source-to-target displacement. Reversing pairs changes sign.
        dx = (target[:, 0, 0, 0] - source[:, 0, 0, 0]) * 5 + self.anchor
        flow = torch.stack((dx, torch.zeros_like(dx)), dim=-1)
        return {
            "flow_px": flow[:, None, None].expand(batch, height, width, 2),
            "certainty_logits": source.new_full((batch, height, width), 2.0),
        }


@pytest.mark.parametrize("prefix", [(2, 2), (2, 2, 3)])
def test_pair_order_direction_and_frozen_labels(prefix):
    count = 1
    for size in prefix:
        count *= size
    rgb = (
        torch.arange(count, dtype=torch.float32)
        .mul(0.025)
        .reshape(*prefix, 1, 1, 1)
        .expand(*prefix, 3, 16, 18)
        .clone()
        .requires_grad_(True)
    )
    rgb_next = rgb + 0.1
    fake = FakeOmega()
    teacher = OmegaWarpTeacher(WarpTeacherCfg(precision="fp32", pair_batch_size=5))
    teacher.model = fake
    flow, weights = teacher(rgb, rgb_next)

    assert flow.shape == (*prefix, 2, 16, 18)
    assert weights.shape == (*prefix, 1, 16, 18)
    torch.testing.assert_close(
        torch.cat([pair[0] for pair in fake.seen_pairs]),
        rgb.detach().reshape(-1, 3, 16, 18) * 2 - 1,
    )
    torch.testing.assert_close(
        torch.cat([pair[1] for pair in fake.seen_pairs]),
        rgb_next.detach().reshape(-1, 3, 16, 18) * 2 - 1,
    )
    torch.testing.assert_close(flow[..., 0, :, :-1], torch.ones_like(flow[..., 0, :, :-1]))
    assert not flow[..., 1, :, :].any()
    assert not weights[..., -1].any()  # right edge maps outside target image
    assert (weights[..., :-1] > 0.8).all()
    assert flow.dtype == weights.dtype == torch.float32
    assert not flow.requires_grad and not weights.requires_grad
    assert not fake.anchor.requires_grad

    reverse, _ = teacher(rgb_next, rgb)
    torch.testing.assert_close(reverse[..., 0, :, 1:], -torch.ones_like(reverse[..., 0, :, 1:]))
    prediction = torch.zeros_like(flow, requires_grad=True)
    ((prediction - flow).square() * weights).mean().backward()
    assert prediction.grad.abs().sum() > 0
    assert rgb.grad is None and fake.anchor.grad is None

    student = nn.Module()
    student.register_parameter("student_weight", nn.Parameter(torch.zeros(())))
    student.teacher = teacher
    assert set(student.state_dict()) == {"student_weight"}


def test_masks_nonfinite_low_confidence_and_pixel_center_boundary():
    flow = torch.zeros(1, 3, 6, 2)
    logits = torch.full((1, 3, 6), torch.logit(torch.tensor(0.75)))
    flow[0, 0, 0, 0] = 1
    flow[0, 0, 1, 0] = float("nan")
    flow[0, 0, 2, 1] = float("inf")
    logits[0, 0, 3] = float("nan")
    logits[0, 0, 4] = float("inf")
    flow[0, 0, 5, 0] = 0.1  # sampleable, but outside the target pixel centres
    logits[0, 1, 0] = -2
    flow[0, 2, 0, 1] = 1  # y order and target height also matter

    targets, weights = warp_output_to_targets(
        {"flow_px": flow, "certainty_logits": logits}, (3, 6)
    )
    expected = torch.full((1, 1, 3, 6), 0.75)
    expected[0, 0, 0, 1:] = 0
    expected[0, 0, 1, 0] = 0
    expected[0, 0, 2, 0] = 0
    torch.testing.assert_close(weights, expected)
    assert targets[0, 0, 0, 0] == 1
    assert torch.isfinite(targets).all() and torch.isfinite(weights).all()
    assert not targets.masked_select(weights.expand_as(targets) == 0).any()


def test_labels_remain_ordinary_tensors_inside_inference_mode():
    teacher = OmegaWarpTeacher(WarpTeacherCfg(precision="fp32"))
    teacher.model = FakeOmega()
    with torch.inference_mode():
        rgb = torch.zeros(1, 1, 3, 16, 18)
        flow, weights = teacher(rgb, rgb + 0.1)
    assert not torch.is_inference(flow) and not torch.is_inference(weights)
    prediction = torch.zeros_like(flow, requires_grad=True)
    torch.nn.functional.mse_loss(prediction * weights, flow).backward()
    assert prediction.grad is not None


def test_missing_checkpoint_fails_without_constructing_a_model(tmp_path):
    teacher = OmegaWarpTeacher(WarpTeacherCfg())
    assert teacher.model is None
    rgb = torch.zeros(1, 1, 3, 16, 16)
    with pytest.raises(ValueError, match="config_path"):
        teacher(rgb, rgb)
    config = tmp_path / "teacher.toml"
    config.write_text("[model]\nkind='vggt_omega'\n")
    teacher = OmegaWarpTeacher(
        WarpTeacherCfg(config_path=str(config), backbone_checkpoint=str(tmp_path / "absent.pt"))
    )
    with pytest.raises(FileNotFoundError, match="backbone_checkpoint"):
        teacher(rgb, rgb)
    assert teacher.model is None


@pytest.mark.parametrize("reject_checkpoint", [False, True])
def test_lazy_loader_checks_hash_and_strict_checkpoint_contract(
    tmp_path, monkeypatch, reject_checkpoint
):
    calls = []
    model = FakeOmega()
    implementation = {"source_tree_sha256": "checked-code",
                      "runtime": {"gpu_devices": ["inference-machine"]},
                      "git": {"root": str(tmp_path)}}
    backbone = {"id": "facebook/VGGT-Omega", "sha256": "checked-backbone"}
    package = ModuleType("vggt_omega_warp")
    package.__path__ = []
    cli = ModuleType("vggt_omega_warp.cli")
    cli.__path__ = []
    common = ModuleType("vggt_omega_warp.cli.common")
    checkpoint = ModuleType("vggt_omega_warp.checkpoint")
    provenance = ModuleType("vggt_omega_warp.provenance")
    common.load_toml = lambda path: {"model": {"kind": "vggt_omega"}}
    common.validate_config_sections = lambda config, **kw: calls.append(("validate", kw))

    def build_model(config, **kwargs):
        calls.append(("build", kwargs))
        return model

    def load_checkpoint(path, loaded, **kwargs):
        assert loaded is model
        calls.append(("load", kwargs))
        if reject_checkpoint:
            raise RuntimeError("checkpoint contract mismatch")
        return {"global_step": 123, "epoch": 4}

    def fingerprint(path):
        return {"path": str(path), "sha256": path.name}, ("identity", path.name)

    def backbone_metadata(config, path, **kwargs):
        calls.append(("hash", kwargs))
        return backbone

    common.build_model = build_model
    checkpoint.load_checkpoint = load_checkpoint
    provenance.stable_file_fingerprint = fingerprint
    provenance.model_implementation_provenance = lambda config: implementation
    provenance.backbone_artifact_metadata = backbone_metadata
    provenance.expected_backbone_metadata = lambda metadata: metadata
    provenance.local_artifact_stat_identity = lambda path: ("identity", path.name)
    provenance.assert_local_artifact_unchanged = (
        lambda path, identity: calls.append(("unchanged", path.name))
    )
    for module in (package, cli, common, checkpoint, provenance):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    paths = {}
    for key in ("config_path", "backbone_checkpoint", "head_checkpoint"):
        path = tmp_path / key
        path.write_text("local test artifact")
        paths[key] = str(path)
    teacher = OmegaWarpTeacher(WarpTeacherCfg(**paths, precision="fp32"))
    assert not calls
    rgb = torch.zeros(1, 1, 3, 16, 18)
    if reject_checkpoint:
        with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
            teacher(rgb, rgb)
        assert teacher.model is None
    else:
        teacher(rgb, rgb)
        teacher(rgb, rgb)
        assert teacher.model is model and not model.training
        assert teacher.provenance["global_step"] == 123
        assert len([call for call in calls if call[0] == "unchanged"]) == 3
    assert ("hash", {"include_hash": True}) in calls
    loads = [call[1] for call in calls if call[0] == "load"]
    assert len(loads) == 1
    assert loads[0] == {
        "map_location": "cpu",
        "strict": True,
        "restore_rng": False,
        "expected_backbone": backbone,
        "expected_implementation": {"source_tree_sha256": "checked-code"},
    }


def test_invalid_rgb_is_not_silently_renormalized():
    teacher = OmegaWarpTeacher(WarpTeacherCfg(precision="fp32"))
    teacher.model = FakeOmega()
    rgb = torch.zeros(1, 1, 3, 16, 16)
    with pytest.raises(ValueError, match=r"finite and in \[0,1\]"):
        teacher(rgb - 0.1, rgb)
    with pytest.raises(TypeError, match="floating point"):
        teacher(rgb.to(torch.uint8), rgb.to(torch.uint8))


def test_fp32_teacher_overrides_enclosing_autocast():
    teacher = OmegaWarpTeacher(WarpTeacherCfg(precision="fp32"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with teacher._autocast_context(torch.device("cpu")):
            assert not torch.is_autocast_enabled("cpu")
