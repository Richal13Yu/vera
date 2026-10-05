"""Compare the released IDM on the same pinned RGB/actions and Warp labels.

This is a common-target diagnostic, not a reproduction of MegaFlow validation.
Warp-label EPE may favor the model trained on those labels. Robot success is
measured separately with the official planner/controller.
"""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from omegaconf import OmegaConf
from vera.experiments.jacobian_learning import JacobianLearningExperiment
from vera.idm.registry import resolve_algorithm_cfg,resolve_algorithm_instance
from vera.idm.jacobian.models.base import JacobianFieldOutput


class FlowUnitAdapter(nn.Module):
    supports_joint_multiview = True
    def __init__(self, model, scales):
        super().__init__(); self.model=model
        self.register_buffer('scales',torch.tensor(scales,dtype=torch.float32))
    def forward(self, obs, cmd):
        out=self.model(obs,cmd)
        # Convert official flow / [9.5378, 8.3015] to flow * 0.1.
        scale=self.scales.view(*([1]*(out.jacobian.ndim-3)),2,1,1)
        j=out.jacobian.float()*scale
        scale_f=self.scales.view(*([1]*(out.optical_flow.ndim-3)),2,1,1)
        return JacobianFieldOutput(jacobian=j,optical_flow=out.optical_flow.float()*scale_f)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load('outputs/omega_warp_validation_20261005/latest/.hydra/config.yaml')
    official=OmegaConf.load('vera-ckpts/idm-mimicgen-285ouq1q/config.yaml')
    official.algorithm.model.pretrained_model_id=None  # full trained backbone is in the checkpoint
    official.algorithm.model.checkpoint_path=None
    official.algorithm.model.image_size=518  # pretrained VGGT positional grid (37x37 + CLS)
    model=resolve_algorithm_instance(resolve_algorithm_cfg(official.algorithm))
    ckpt=torch.load('vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt',map_location='cpu',weights_only=True,mmap=True)
    state=ckpt['state_dict']; state=model._adapt_state_dict_for_compile(state)
    model.load_state_dict(state,strict=True)
    step=ckpt.get('global_step'); del ckpt,state
    exp=JacobianLearningExperiment(cfg,logger=None)
    exp.algo=exp._build_algo(); exp._inject_dataset_metadata()
    # The two checkpoints use identical normalized action scales.
    assert list(official.dataset.action_abs_scale)==list(cfg.dataset.action_abs_scale)
    scale=[float(x)*float(cfg.dataset.oflow_scale) for x in official.dataset.oflow_abs_scale]
    exp.algo.model=FlowUnitAdapter(model.model,scale)
    trainer=exp._build_trainer('validation',[])
    results=trainer.validate(exp.algo,datamodule=exp.data_module)
    payload={'model':'official_285ouq1q','global_step':step,
             'protocol':'same RGB/actions/Warp labels as Omega validation, official J converted to common flow units',
             'warning':'Warp pseudo-label diagnostics are not independent ground-truth flow or robot success.',
             'scale_official_J_to_common':scale,
             'dataloader_names':exp.data_module.validation_dataloader_names,'results':results}
    (args.output/'validation_results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('OFFICIAL_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
