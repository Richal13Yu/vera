"""Summarize native validation JSON, excluding the duplicate `all` batch.

Usage: python scripts/summarize_mimicgen_validation.py OUTPUT_ROOT
Expects protocol.json plus latest/ and best_train/validation_results.json.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.root / 'protocol.json').read_text())
    tasks = list(protocol['task_pins'])
    metrics = {
        'val_loss': ('loss', 'total'),
        'warp_epe_px': ('diagnostics', 'warp_epe_px'),
        'action_mse_normalized': ('metrics', 'action_mse'),
        'warp_valid_ratio': ('diagnostics', 'warp_valid_ratio'),
        'warp_certainty_mean': ('diagnostics', 'warp_certainty_mean'),
    }
    rows, summary = [], {'protocol': protocol, 'runs': {}}
    for run in ('best_train', 'latest'):
        data = json.loads((args.root / run / 'validation_results.json').read_text())
        assert data['dataloader_names'] == tasks + ['all'], data['dataloader_names']
        assert len(data['results']) == len(data['dataloader_names'])
        step = protocol['latest_checkpoint_step'] if run == 'latest' else protocol['best_training_checkpoint_step']
        run_rows = []
        for task, result in zip(data['dataloader_names'], data['results']):
            values = {name: result[f'{prefix}/validation/{task}/{metric}']
                      for name, (prefix, metric) in metrics.items()}
            assert all(math.isfinite(v) for v in values.values()), (run, task, values)
            row = {'run': run, 'step': step, 'task': task, **values}
            run_rows.append(row)
        for name in metrics:
            assert run_rows[-1][name] == run_rows[0][name], 'Expected original all to repeat first task'
        rows.extend(run_rows)
        summary['runs'][run] = {
            'step': step,
            'macro_9_tasks': {name: mean(row[name] for row in run_rows[:-1]) for name in metrics},
            'video_count': len(list((args.root / run).rglob('*.mp4'))),
            'results_path': str((args.root / run / 'validation_results.json').resolve()),
        }
    before = summary['runs']['best_train']['macro_9_tasks']
    after = summary['runs']['latest']['macro_9_tasks']
    summary['latest_vs_269000_percent_change'] = {
        name: 100 * (after[name] / before[name] - 1)
        for name in ('val_loss', 'warp_epe_px', 'action_mse_normalized')
    }
    with (args.root / 'per_task.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary['runs'], indent=2))
    print(json.dumps(summary['latest_vs_269000_percent_change'], indent=2))


if __name__ == '__main__':
    main()
