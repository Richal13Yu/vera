"""Resume migration preserves the decoder's AdamW trajectory."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.unfreeze_omega_checkpoint import extend_optimizer_state


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.decoder = nn.Linear(3, 2)
        self.cfg = SimpleNamespace(freeze_aggregator=True, aggregator_lr_multiplier=1.)
        self.set_aggregator_trainable(False)

    def set_aggregator_trainable(self, trainable):
        self.cfg.freeze_aggregator = not trainable
        self.backbone.requires_grad_(trainable)


def test_migration_preserves_decoder_moments_and_next_update():
    torch.manual_seed(7)
    model = Student()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    for p in model.decoder.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    reference = copy.deepcopy(model)
    reference_opt = torch.optim.AdamW(reference.decoder.parameters(), lr=5e-5)
    reference_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    checkpoint = {'optimizer_states': [copy.deepcopy(opt.state_dict())], 'global_step': 407000}
    report = extend_optimizer_state(checkpoint, model)
    assert report == {'preserved_optimizer_states': 2, 'new_backbone_parameter_tensors': 2}
    resumed = torch.optim.AdamW(model.parameters(), lr=1.)
    resumed.load_state_dict(checkpoint['optimizer_states'][0])
    assert resumed.param_groups[0]['lr'] == 5e-5
    assert checkpoint['global_step'] == 407000
    assert all(p not in resumed.state for p in model.backbone.parameters())
    for p in model.parameters():
        p.grad = torch.full_like(p, 2.)
    for p in reference.decoder.parameters():
        p.grad = torch.full_like(p, 2.)
    resumed.step()
    reference_opt.step()
    for actual, expected in zip(model.decoder.parameters(), reference.decoder.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(resumed.state[p]['step'] == 1 for p in model.backbone.parameters())
    assert all(resumed.state[p]['step'] == 2 for p in model.decoder.parameters())


def test_migration_rejects_an_already_unfrozen_model():
    model = Student()
    model.set_aggregator_trainable(True)
    with pytest.raises(ValueError, match='frozen Omega'):
        extend_optimizer_state({}, model)
