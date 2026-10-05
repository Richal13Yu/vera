"""Paired MimicGen rollouts through the official adapter, using local transport.

The planner, CoTracker, gated controller, states, seeds, and horizons are shared.
Omega receives its simultaneous camera input and an explicit conversion into the
released controller's flow units. No controller gains are tuned per model.
"""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf


class OmegaServingAdapter(nn.Module):
    def __init__(self, model, official_flow_scales, omega_flow_scale):
        super().__init__(); self.model=model; self.cfg=model.cfg
        self.register_buffer('factor',1/(torch.tensor(official_flow_scales)*omega_flow_scale))
    def compute_jacobian(self, obs):
        from vera.idm.jacobian.models.base import InputObservation
        rgb=obs.rgb
        if rgb.ndim!=4 or rgb.shape[0]%2:
            raise ValueError(f'Expected flattened pairs of simultaneous camera views, got {rgb.shape}')
        grouped=rgb.reshape(-1,2,*rgb.shape[1:])
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            j=self.model.compute_jacobian(InputObservation(rgb=grouped))
        j=j.flatten(0,1).float()
        return j*self.factor.view(1,1,2,1,1)


class LocalClient:
    def __init__(self,adapter):self.adapter=adapter
    def reset(self,info):return self.adapter.reset(info)
    def infer(self,obs):return self.adapter.infer(obs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['official','omega'],required=True)
    parser.add_argument('--num-demos',type=int,default=10)
    parser.add_argument('--horizon',type=int,default=200)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    # Bundle inference config with a link to the original immutable checkpoint.
    bundle=args.output/'idm';bundle.mkdir(exist_ok=True)
    official=OmegaConf.load(root/'vera-ckpts/idm-mimicgen-285ouq1q/config.yaml')
    if args.variant=='official':
        config=official.copy()
        config.algorithm.model.pretrained_model_id=None
        config.algorithm.model.checkpoint_path=None
        config.algorithm.model.image_size=518  # keep bundled pretrained positional grid
        ckpt=root/'vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt'
    else:
        config=OmegaConf.load(root/'outputs/omega_warp_validation_20261005/latest/.hydra/config.yaml')
        ckpt=root/'outputs/omega_warp_validation_20261005/latest_snapshot.ckpt'
        assert list(config.dataset.action_abs_scale)==list(official.dataset.action_abs_scale)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(config,resolve=True)),bundle/'config.yaml')
    link=bundle/'model.ckpt'
    if not link.exists():link.symlink_to(ckpt)
    os.environ['VERA_MIMICGEN_DYNAMICS_CKPT']=str(link.resolve().parent/'model.ckpt') if args.variant=='official' else str(link.absolute())
    # Keep the sidecar in this bundle for both models (avoid pretrained redownload).
    os.environ['VERA_MIMICGEN_DYNAMICS_CKPT']=str(link.absolute())
    from vera.server.protocol.adapter_factory import make_adapter
    from vera.server.start_vera_server import _enable_teacache
    from vera.controller.run_mimicgen_eval import RemotePolicy
    from vera.env_runner.mimicgen_runner import MimicgenRunner,MimicgenRunnerCfg
    adapter=make_adapter('mimicgen',device=torch.device('cuda:0'),
        algo_config_path=str(root/'vera-ckpts/mimicgen-wan-1.3b/algo_config.yaml'),
        text='A robot arm stacks one block on top of another block',run_dir=str(args.output))
    _enable_teacache(adapter._policy,0.10)
    adapter._policy.dynamics_model.eval()
    if args.variant=='omega':
        algo=adapter._policy.dynamics_model
        algo.model=OmegaServingAdapter(algo.model,list(official.dataset.oflow_abs_scale),float(config.dataset.oflow_scale)).to('cuda:0')
    context_frames=int(adapter.config.context_frames)
    cfg=MimicgenRunnerCfg(env_name='mimicgen',dataset_path=str(root/'data/mimicgen_raw/core/stack_d0.hdf5'),
        render_size=128,render_obs_key=list(adapter.config.view_keys),num_demos_to_run=1,
        max_episode_steps=args.horizon,n_repeat=1,action_scale=1.0,
        save_videos=True,save_trajectory=True,save_rrd=False,output_dir=str(args.output/'rollouts'),
        use_stored_model=False,demo_warmup_steps=context_frames-1,log_step_debug=False)
    runner=MimicgenRunner(cfg,device='cpu')
    remote=RemotePolicy(LocalClient(adapter),view_keys=list(adapter.config.view_keys),
        view_widths=[128,128],context_frames=context_frames,
        prompt='A robot arm stacks one block on top of another block')
    import h5py
    with h5py.File(cfg.dataset_path,'r') as f:keys=sorted(f['data'].keys())[:args.num_demos]
    report={'variant':args.variant,'task':'stack_d0','horizon':args.horizon,'demo_keys':keys,
        'context_frames':context_frames,'seed_base':args.seed,'episodes':[],
        'policy_cfg':runner._to_jsonable(adapter._policy.cfg),'status':'running',
        'notes':['Official RemotePolicy and VeraPolicyAdapter via local transport.',
                 'Official published planner has skip_text_encoder=true; both arms use its empty prompt embeddings.',
                 'No controller gains tuned; Omega J is converted into official flow normalization and cameras grouped as trained.']}
    def save():
        temp=args.output/'results.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.output/'results.json')
    save()
    for i,key in enumerate(keys):
        seed=args.seed+i;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        start=time.time();res=runner.run(remote,options={'demo_keys':[key]},run_tag=f'{args.variant}_{key}_seed{seed}')
        episode={'demo':key,'seed':seed,'success':bool(res['env_successes'][0]),
                 'relaxed_success':bool(res['relaxed_successes'][0]),'max_reward':float(res['max_rewards'][0]),
                 'return':float(res['demo_returns'][0]),'elapsed_s':time.time()-start,'save_dir':res['save_dir']}
        report['episodes'].append(episode);save();print('EPISODE_RESULT',json.dumps(episode),flush=True)
    report['status']='complete';report['success_rate']=np.mean([e['success'] for e in report['episodes']]).item();save()
    adapter.flush();runner.env.close();print('ROBOT_EVAL_COMPLETE',report['success_rate'],flush=True)


if __name__=='__main__':main()
