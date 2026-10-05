"""Summarize completed, paired official-versus-Omega robot experiments."""
import argparse,csv,json
from pathlib import Path
from statistics import mean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    runs={v:json.loads((args.root/('robot_'+v)/'results.json').read_text()) for v in ['official','omega']}
    a,b=runs['official'],runs['omega']
    assert a['status']==b['status']=='complete','Both robot arms must finish first'
    for k in ['task','horizon','demo_keys','context_frames','seed_base']:
        assert a[k]==b[k],k
    for k in ['controller','motion_planner','adaptive_controller','n_action_steps','action_chunk_horizon',
              'action_scale_translation','action_scale_yaw','gripper_min_hold_steps']:
        assert a['policy_cfg'][k]==b['policy_cfg'][k],k
    assert len(a['episodes'])==len(b['episodes'])==len(a['demo_keys'])
    rows=[]
    for x,y in zip(a['episodes'],b['episodes']):
        assert (x['demo'],x['seed'])==(y['demo'],y['seed'])
        rows.append({'demo':x['demo'],'seed':x['seed'],'official_success':x['success'],
                     'omega_success':y['success'],'official_max_reward':x['max_reward'],
                     'omega_max_reward':y['max_reward'],'official_video_dir':str(Path(x['save_dir'])/'videos'),
                     'omega_video_dir':str(Path(y['save_dir'])/'videos')})
    report={'robot':{v:{'successes':sum(e['success'] for e in d['episodes']),
                         'episodes':len(d['episodes']),'success_rate':d['success_rate'],
                         'mean_max_reward':mean(e['max_reward'] for e in d['episodes'])}
                       for v,d in runs.items()},
            'offline_common_warp_target':json.loads((args.root/'offline_comparison.json').read_text()),
            'robot_protocol':json.loads((args.root/'robot_protocol.json').read_text()),
            'paired_rows':rows}
    (args.root/'comparison_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (args.root/'robot_per_episode.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report['robot'],indent=2))


if __name__=='__main__':main()
