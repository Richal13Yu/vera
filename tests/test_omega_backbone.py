"""CPU contracts for the native Omega student, without downloaded weights.

The real 24-layer Omega encoder runs at width 64 to keep tests small. VERA's
unrelated import-time training dependencies are isolated. If VGGT is absent,
its DPT utility functions are replaced by their local Omega equivalents; the
actual VERA JacobianDptHead and forward implementation still run.
"""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


@pytest.fixture(scope="module")
def student_modules():
    root = Path(__file__).resolve().parents[1]
    old_threads = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(root / "vggt-omega-warp"))
        pytest.importorskip("vggt_omega.models.aggregator")
        package_name = "_vera_omega_student_test"
        for name in ("vera", "vera.utils", "vera.utils.convention", "vera.utils.geometry", package_name):
            module = types.ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        sys.modules["vera.utils.geometry"].project_world_coords_to_camera = lambda *args: None

        class Float:
            def __class_getitem__(cls, item):
                return torch.Tensor

        typing_module = types.ModuleType("jaxtyping")
        typing_module.Float = Float
        monkeypatch.setitem(sys.modules, "jaxtyping", typing_module)
        registry = types.ModuleType(f"{package_name}.registry")
        registry.register_model = lambda *args, **kwargs: lambda cls: cls
        monkeypatch.setitem(sys.modules, registry.__name__, registry)

        def load(stem):
            name = f"{package_name}.{stem}"
            path = root / "vera/idm/jacobian/models" / f"{stem}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            return module

        base = load("base")
        vggt = load("vggt_jacobian_field")
        if vggt.VGGT is None:
            from vggt_omega.models.heads import dense_head, utils

            def fusion_block(features, has_residual=True):
                block = dense_head._make_fusion_block(features, has_residual=has_residual)
                forward = block.forward

                def with_default_size(*xs, size=None):
                    # VGGT defaults to 2x upsampling; Omega supplies explicit sizes.
                    if size is None:
                        size = tuple(dimension * 2 for dimension in xs[0].shape[-2:])
                    return forward(*xs, size=size)

                block.forward = with_default_size
                return block

            vggt.VGGT = object()
            vggt._make_scratch = lambda channels, features, expand=False: dense_head._make_scratch(channels, features)
            vggt._make_fusion_block = fusion_block
            vggt.custom_interpolate = dense_head.custom_interpolate
            vggt.create_uv_grid = utils.create_uv_grid
            vggt.position_grid_to_embed = utils.position_grid_to_embed
        torch.set_num_threads(1)
        try:
            yield base, load("vggt_omega_jacobian_field")
        finally:
            torch.set_num_threads(old_threads)


def make_cfg(omega, **overrides):
    arguments = dict(
        name="vggt_omega_jacobian", command_dim=3, spatial_dim=2, image_size=32,
        embed_dim=64, decoder_features=16, decoder_out_channels=[16, 16, 16, 16],
        predict_uncertainty=True, aggregator_lr_multiplier=0.1,
    )
    arguments.update(overrides)
    return omega.VggtOmegaJacobianFieldCfg(**arguments)


@pytest.mark.parametrize("multiview", [False, True])
def test_real_encoder_current_view_shapes_flow_and_gradients(student_modules, multiview):
    base, omega = student_modules
    torch.manual_seed(5)
    model = omega.VggtOmegaJacobianField(make_cfg(omega))
    assert type(model.vggt.aggregator).__name__ == "Aggregator"
    assert model.vggt.aggregator.cached_layer_indices == {4, 11, 17, 23}
    assert model.vggt.aggregator.patch_token_start == 17
    prefix = (1, 2) if multiview else (1,)
    observation = base.InputObservation(rgb=torch.rand(*prefix, 3, 17, 19))
    command = base.InputCommand(du=torch.randn(1, 3))
    output = model(observation, command)
    assert output.jacobian.shape == (*prefix, 3, 2, 17, 19)
    assert output.optical_flow.shape == (*prefix, 2, 17, 19)
    assert output.flow_confidence.shape == (*prefix, 1, 17, 19)
    assert torch.isfinite(output.jacobian).all()
    assert (output.flow_confidence > 0).all()
    equation = "bvcdhw,bc->bvdhw" if multiview else "bcdhw,bc->bdhw"
    torch.testing.assert_close(output.optical_flow, torch.einsum(equation, output.jacobian, command.du))
    (output.optical_flow.square().mean() + output.flow_confidence.mean()).backward()
    for module in (model.vggt.aggregator, model.decoder):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in module.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters()
                   if parameter.grad is not None)


def test_freezing_keeps_decoder_trainable_and_optimizer_groups_correct(student_modules):
    base, omega = student_modules
    model = omega.VggtOmegaJacobianField(make_cfg(omega, freeze_aggregator=True))
    model.train()
    assert not model.vggt.aggregator.training
    assert not any(parameter.requires_grad for parameter in model.vggt.aggregator.parameters())
    assert model.decoder.training
    groups = model.get_optimizer_param_groups(base_lr=0.001)
    assert len(groups) == 1 and groups[0]["lr"] == 0.001
    output = model(base.InputObservation(rgb=torch.rand(1, 3, 17, 19)),
                   base.InputCommand(du=torch.randn(1, 3)))
    output.optical_flow.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.vggt.aggregator.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    model.set_aggregator_trainable(True)
    assert model.vggt.aggregator.training
    assert all(parameter.requires_grad for parameter in model.vggt.aggregator.parameters())
    groups = model.get_optimizer_param_groups(base_lr=0.001)
    assert [group["lr"] for group in groups] == [0.001, 0.0001]
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    model.set_aggregator_trainable(False)
    model.cfg.aggregator_lr_multiplier = 1.0
    assert all(parameter.requires_grad for parameter in model.get_optimizer_param_groups(base_lr=0.001))


@pytest.mark.parametrize("extension", ["pt", "safetensors"])
def test_strict_meta_checkpoint_load_and_missing_key_rejection(student_modules, tmp_path, extension):
    base, omega = student_modules
    if extension == "safetensors":
        safetensors = pytest.importorskip("safetensors.torch")
    from vggt_omega_warp.backbone import CheckpointFormatError

    model = omega.VggtOmegaJacobianField(make_cfg(omega))
    state = {"aggregator." + key: value for key, value in model.vggt.aggregator.state_dict().items()}
    state["dense_head.ignored"] = torch.ones(1)
    checkpoint = tmp_path / f"omega.{extension}"

    def save():
        if extension == "pt":
            torch.save({"model": state}, checkpoint)
        else:
            safetensors.save_file(state, str(checkpoint))

    save()
    loaded = omega.VggtOmegaJacobianField(make_cfg(omega, checkpoint_path=str(checkpoint)))
    assert loaded.checkpoint_report.tensor_count == len(model.vggt.aggregator.state_dict())
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    for key, value in loaded.vggt.aggregator.state_dict().items():
        torch.testing.assert_close(value, model.vggt.aggregator.state_dict()[key])
    output = loaded(base.InputObservation(rgb=torch.rand(1, 3, 17, 19)),
                    base.InputCommand(du=torch.randn(1, 3)))
    assert torch.isfinite(output.optical_flow).all()
    del state[next(key for key in state if key.startswith("aggregator."))]
    save()
    with pytest.raises(CheckpointFormatError):
        omega.VggtOmegaJacobianField(make_cfg(omega, checkpoint_path=str(checkpoint)))


@pytest.mark.parametrize("overrides", [
    {"patch_size": 14}, {"strict_checkpoint_load": False},
    {"decoder_intermediate_layer_idx": [0, 1, 2, 24]},
    {"decoder_intermediate_layer_idx": [4, 4, 17, 23]},
    {"pretrained_model_id": "remote/model"},
])
def test_incompatible_configs_fail_before_encoder_construction(student_modules, overrides):
    _, omega = student_modules
    with pytest.raises(ValueError):
        omega.VggtOmegaJacobianField(make_cfg(omega, **overrides))


def _ddp_student_worker(rank, init_file):
    """Two iterations catch parameters unused by DDP's previous reduction."""
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    fixture = student_modules.__wrapped__()
    base, omega = next(fixture)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(5)
        model = omega.VggtOmegaJacobianField(make_cfg(
            omega, freeze_aggregator=False, predict_uncertainty=False))
        wrapped = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.get_optimizer_param_groups(base_lr=1e-4))
        before = model.vggt.aggregator.camera_token.detach().clone()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(rank * 10 + step)
            output = wrapped(base.InputObservation(rgb=torch.rand(1, 2, 3, 17, 19)),
                             base.InputCommand(du=torch.randn(1, 3)))
            output.optical_flow.square().mean().backward()
            optimizer.step()
        updated = model.vggt.aggregator.camera_token.detach()
        assert not torch.equal(updated, before)
        gathered = [torch.empty_like(updated) for _ in range(2)]
        dist.all_gather(gathered, updated)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
        fixture.close()


def test_unfrozen_student_two_rank_ddp(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo distributed backend is unavailable')
    torch.multiprocessing.start_processes(
        _ddp_student_worker, args=(str(tmp_path / 'ddp_init'),),
        nprocs=2, join=True, start_method='spawn')
