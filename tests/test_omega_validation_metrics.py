"""Validation diagnoses action recovery without changing the training objective."""
from types import SimpleNamespace

import torch

from vera.idm.jacobian.omega_warp_jacobian import OmegaWarpJacobian


def test_validation_recovers_known_action_with_inverse_training_loss_disabled():
    jacobian = torch.zeros(1, 2, 2, 2, 3)
    jacobian[:, 0, 0] = 1.
    jacobian[:, 1, 1] = 1.
    action = torch.tensor([[2., -1.]])
    flow = torch.einsum('nashw,na->nshw', jacobian, action)
    weights = torch.ones(1, 1, 2, 3)
    output = SimpleNamespace(optical_flow=flow, jacobian=jacobian)
    algo = SimpleNamespace(
        cfg=SimpleNamespace(motion_aware_flow_weighting=False, flow_loss='charbonnier',
                            flow_charbonnier_eps=1e-3, log_flow_per_view=False,
                            view_balanced_flow_loss=True, flow_gradient_weight=0.,
                            jacobian_tv_weight=0., predict_uncertainty=False,
                            inverse_action_weight=0., inverse_action_damping=1e-6),
        warp_teacher=lambda *args: (flow, weights),
        dataset_metadata={'flow_normalization_mode': 'scale', 'oflow_scale': 1.},
        _prepare_model_inputs=lambda **kwargs: (kwargs['rgb'], action, flow, None, action, None),
        model=lambda *args: output,
    )
    def compute(namespace):
        batch = {'rgb': torch.zeros(1, 1, 3, 2, 3),
                 'rgb_next': torch.zeros(1, 1, 3, 2, 3), 'du': action}
        return OmegaWarpJacobian._compute_loss(algo, batch, namespace=namespace)
    training, validation = compute('training'), compute('validation/task')
    assert set(training[0]) == set(validation[0]) == {'flow'}
    torch.testing.assert_close(training[2], validation[2], rtol=0, atol=0)
    assert 'du_hat_flow_mse' not in training[4]
    assert validation[4]['du_hat_flow_mse'].item() < 1e-9
    assert validation[4]['warp_epe_px'].item() == 0.
