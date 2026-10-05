from types import SimpleNamespace
import torch
from torch import nn
from scripts.compare_mimicgen_idm import FlowUnitAdapter
from scripts.run_mimicgen_comparison import OmegaServingAdapter
from vera.idm.jacobian.models.base import InputObservation,InputCommand,JacobianFieldOutput


class KnownField(nn.Module):
    cfg=SimpleNamespace()
    def compute_jacobian(self,obs):
        assert obs.rgb.shape[1]==2  # camera axis must remain simultaneous
        return torch.ones(obs.rgb.shape[0],2,7,2,3,3)
    def forward(self,obs,cmd):
        j=self.compute_jacobian(obs)
        return JacobianFieldOutput(jacobian=j,optical_flow=torch.einsum('bvashw,ba->bvshw',j,cmd.du))


def test_official_flow_conversion_preserves_jacobian_action_relation():
    m=FlowUnitAdapter(KnownField(),[0.9,0.8])
    cmd=InputCommand(du=torch.arange(7.).reshape(1,7))
    out=m(InputObservation(rgb=torch.zeros(1,2,3,3,3)),cmd)
    torch.testing.assert_close(out.optical_flow,torch.einsum('bvashw,ba->bvshw',out.jacobian,cmd.du))
    torch.testing.assert_close(out.jacobian[0,0,0,:,0,0],torch.tensor([0.9,0.8]))


def test_omega_serving_groups_views_and_inverts_the_same_unit_conversion():
    adapter=OmegaServingAdapter(KnownField(),[9.,8.],0.1)
    j=adapter.compute_jacobian(InputObservation(rgb=torch.zeros(2,3,3,3)))
    assert j.shape==(2,7,2,3,3)
    torch.testing.assert_close(j[0,0,:,0,0],torch.tensor([1/0.9,1/0.8]))


def test_wan_motion_track_config_preserves_selected_backend():
    from vera.video_model.link.wan_pipeline import MotionTrackConfig
    from vera.policy.world_models.tracker_backends import tracker_backend_from_cfg
    assert tracker_backend_from_cfg(MotionTrackConfig(backend='cotracker')) == 'cotracker'
    assert tracker_backend_from_cfg(SimpleNamespace(tracker_backend='megaflow')) == 'megaflow'
    assert tracker_backend_from_cfg(SimpleNamespace()) == 'alltracker'


def test_cotracker_float_visibility_can_render_without_changing_tracks():
    from vera.policy.world_models.cotracker_inference import CoTrackerInference
    tracker=CoTrackerInference(device='cpu')
    tracks=torch.tensor([[[[1.,1.],[2.,2.]],[[2.,1.],[3.,2.]]]])
    visibility=torch.tensor([[[1.,0.],[1.,1.]]])
    tracker._model=lambda video,grid_size: (tracks,visibility)
    out=tracker.infer(torch.zeros(1,2,3,4,4),return_visualization=True)
    assert out.visualization.shape == (1,2,4,4,3)
    assert out.visualization.dtype == torch.uint8
    torch.testing.assert_close(out.motion_tracks.xy_src,tracks[:,:-1])
