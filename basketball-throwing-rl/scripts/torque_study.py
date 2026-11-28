"""Predeclared from-scratch factorial PPO study; never loads a trained policy for training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

SCRIPT = Path(__file__).with_name('torque_experiment.py')


def make_protocol(timesteps, seeds):
    if timesteps <= 0 or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('positive budget and at least 3 distinct seeds required')
    rng = np.random.default_rng(20261127)
    return {'training_mode': 'from_scratch', 'requested_timesteps': timesteps,
            'seeds': seeds, 'conditions': {'none': [0., 0.], 'torque': [.5, 0.],
                                           'work': [0., .02], 'both': [.5, .02]},
            'checkpoint_steps': [s for s in [100000,250000,500000,750000] if s < timesteps],
            'targets_xz': np.column_stack((rng.uniform(1,1.5,300), rng.uniform(.75,1.1,300))).tolist(),
            'target_seed': 20261127, 'evaluation_seed': 27000,
            'training_target_x_range': [1.,1.5], 'training_target_z_range': [.75,1.1],
            'selection': 'predeclared checkpoints and final, no best-checkpoint selection',
            'time_weight': .1, 'torque_reference_s': 2., 'work_reference_j': 50.}


def validate_initializations(configs):
    hashes = {}
    for cfg in configs:
        init = cfg['initialization']
        if (init['mode'] != 'from_scratch' or init['num_timesteps'] != 0
                or not np.isclose(init['observation_count'], .0001, rtol=0, atol=1e-12)):
            raise ValueError('warm start or previously updated normalization detected')
        seed = cfg['seed']
        if seed in hashes and hashes[seed] != init['policy_sha256']:
            raise ValueError('conditions do not share initial policy for a seed')
        hashes[seed] = init['policy_sha256']
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('distinct seeds have identical initial policy')


def run_one(out, protocol, condition, seed):
    name = f'{condition}-seed{seed}'
    run = out / name
    torque, work = protocol['conditions'][condition]
    env = {**os.environ, 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    start = time.monotonic()
    with (out / f'{name}.log').open('w') as log:
        subprocess.run([sys.executable,str(SCRIPT),'train','--out',str(run),
                        '--preset','time','--seed',str(seed),'--timesteps',str(protocol['requested_timesteps']),
                        '--effort-weight',str(torque),'--work-weight',str(work),'--torch-threads','1',
                        '--x-range','1','1.5','--z-range','.75','1.1'],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        cfg=json.loads((run/'config.json').read_text())
        validate_initializations([cfg])
        for step in [*protocol['checkpoint_steps'],None]:
            label = str(step) if step is not None else 'final'
            command=[sys.executable,str(SCRIPT),'evaluate','--run',str(run),
                     '--out',str(out/f'{name}-eval-{label}'),'--seed',str(protocol['evaluation_seed']),
                     '--targets',*[f'{x},{z}' for x,z in protocol['targets_xz']]]
            if step is not None:
                command += ['--checkpoint-steps',str(step)]
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return {'name':name,'condition':condition,'seed':seed,'wall_time_s':time.monotonic()-start}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timesteps',type=int,default=1000000)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    protocol=make_protocol(args.timesteps,args.seeds)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    done=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_one,args.out,protocol,c,s):(c,s) for s in args.seeds for c in protocol['conditions']}
        for job in as_completed(jobs):
            condition,seed=jobs[job]
            try:
                result=job.result();result['status']='complete'
            except Exception as exc:
                result={'condition':condition,'seed':seed,'status':'failed','error':str(exc)}
            done.append(result)
            (args.out/'status.json').write_text(json.dumps(done,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    if any(r['status']!='complete' for r in done):
        raise RuntimeError('Study incomplete; inspect status and per-run logs. No failed run is omitted.')
    configs=[json.loads((args.out/f'{c}-seed{s}'/'config.json').read_text()) for s in args.seeds for c in protocol['conditions']]
    validate_initializations(configs)
    if len({c['actual_timesteps'] for c in configs}) != 1:
        raise ValueError('unequal training budgets')
    (args.out/'initialization-audit.json').write_text(json.dumps({'passed':True,'runs':len(configs),'initializations':[{'seed':c['seed'],**c['initialization']} for c in configs]},indent=2)+'\n')


if __name__=='__main__':
    main()
