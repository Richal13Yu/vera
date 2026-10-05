"""Extend a frozen Omega run's AdamW state for full-backbone continuation.

Preserves weights, decoder moments, optimizer settings, and Lightning progress.
Newly unfrozen backbone parameters receive fresh AdamW state on their first step.
The input checkpoint is never overwritten; the frozen teacher is not in it.
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import vera.idm  # noqa: F401 -- register algorithms
from vera.idm.registry import resolve_algorithm_cfg, resolve_algorithm_instance


def extend_optimizer_state(checkpoint, model):
    if not model.cfg.freeze_aggregator or model.cfg.aggregator_lr_multiplier != 1.0:
        raise ValueError("Expected a frozen Omega student with aggregator_lr_multiplier=1")
    old_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizers = checkpoint['optimizer_states']
    if len(optimizers) != 1 or len(optimizers[0]['param_groups']) != 1:
        raise ValueError("Expected one AdamW optimizer with one parameter group")
    optimizer = optimizers[0]
    old_ids = optimizer['param_groups'][0]['params']
    if len(old_ids) != len(old_parameters):
        raise ValueError("Checkpoint optimizer does not match the frozen student's parameters")
    model.set_aggregator_trainable(True)
    new_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    new_ids = {name: index for index, (name, _) in enumerate(new_parameters)}
    new_state = {}
    for old_id, (name, parameter) in zip(old_ids, old_parameters):
        state = optimizer['state'].get(old_id)
        if state is not None:
            if any(state[key].shape != parameter.shape for key in ('exp_avg', 'exp_avg_sq')):
                raise ValueError(f"Optimizer moment shape mismatch for {name}")
            new_state[new_ids[name]] = state
    optimizer['state'] = new_state
    optimizer['param_groups'][0]['params'] = list(range(len(new_parameters)))
    return {'preserved_optimizer_states': len(new_state),
            'new_backbone_parameter_tensors': len(new_parameters) - len(old_parameters)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, required=True,
                        help='Source training run .hydra/config.yaml; export its VERA_* paths first')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.checkpoint.resolve():
        parser.error('output must be a new checkpoint path')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg = OmegaConf.load(args.config)
    if cfg.algorithm.model.name != 'vggt_omega_jacobian':
        parser.error('expected a vggt_omega_jacobian source config')
    if not cfg.algorithm.model.freeze_aggregator:
        parser.error('source config must describe a frozen backbone')
    algo = resolve_algorithm_instance(resolve_algorithm_cfg(cfg.algorithm))
    algo.load_state_dict(checkpoint['state_dict'], strict=True)
    report = extend_optimizer_state(checkpoint, algo.model)
    # Exercise PyTorch's loader before exporting the migrated state.
    algo.configure_optimizers()['optimizer'].load_state_dict(checkpoint['optimizer_states'][0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    report.update(source=str(args.checkpoint.resolve()), output=str(args.output.resolve()),
                  global_step=checkpoint['global_step'], teacher_unchanged=True)
    args.output.with_suffix('.migration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
