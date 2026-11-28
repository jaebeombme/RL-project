"""PPO motion-efficiency experiments. Run with --help; legacy CLI is unchanged."""
import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
import hashlib
import time
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.basketball_torque_env import BasketballTorqueEnv, TorqueConfig

METRICS = ('elapsed_steps', 'elapsed_time_s', 'release_time_s', 'success_time_s',
           'torque_squared_integral', 'normalized_effort', 'absolute_work_j')
PRESETS = {'baseline': (0., 0.), 'time': (.1, 0.), 'efficient': (.1, .05)}


def config_for_preset(preset, **overrides):
    time_weight, effort_weight = PRESETS[preset]
    values = dict(time_weight=time_weight, effort_weight=effort_weight)
    values.update(overrides)
    return TorqueConfig(**values)


def summarize(rows):
    if not rows:
        raise ValueError('cannot summarize zero episodes')
    successful = [row for row in rows if row['is_success']]
    def stats(group):
        result = {}
        for key in METRICS:
            values = [float(row[key]) for row in group if row[key] is not None]
            result[key] = {'count': len(values),
                           'mean': float(np.mean(values)) if values else None,
                           'std': float(np.std(values)) if values else None}
        return result
    return {'episodes': len(rows), 'successes': len(successful),
            'success_rate': len(successful) / len(rows),
            'all_episodes': stats(rows), 'successful_episodes': stats(successful)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def train(args):
    if args.timesteps <= 0 or args.n_steps < 2 or args.batch_size < 2:
        raise ValueError('positive timesteps and n_steps/batch_size >= 2 required')
    if args.batch_size > args.n_steps or args.n_steps % args.batch_size:
        raise ValueError('batch_size must divide n_steps')
    if args.torch_threads < 1:
        raise ValueError('torch_threads must be positive')
    torch.set_num_threads(args.torch_threads)
    weights = {'work_weight': args.work_weight}
    if args.effort_weight is not None:
        weights['effort_weight'] = args.effort_weight
    cfg = config_for_preset(args.preset, **weights, torque_limits=tuple(args.torque_limits),
                            target=tuple(args.target),
                            target_x_range=None if args.fixed_target else tuple(args.x_range),
                            target_z_range=None if args.fixed_target else tuple(args.z_range))
    args.out.mkdir(parents=True, exist_ok=False)
    probe = BasketballTorqueEnv(cfg)
    try:
        check_env(probe, warn=True)
    finally:
        probe.close()
    raw = Monitor(BasketballTorqueEnv(cfg), filename=str(args.out / 'train'),
                  info_keywords=('is_success', 'termination_reason', *METRICS))
    vec = VecNormalize(DummyVecEnv([lambda: raw]), norm_obs=True, norm_reward=False)
    vec.seed(args.seed)
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                text=True, capture_output=True, check=False)
        metadata = {'preset': args.preset, 'seed': args.seed,
                    'requested_timesteps': args.timesteps, 'environment': asdict(cfg),
                    'ppo': {'n_steps': args.n_steps, 'batch_size': args.batch_size,
                            'learning_rate': 3e-4, 'gamma': .99, 'gae_lambda': .95,
                            'n_epochs': 10, 'clip_range': .2, 'ent_coef': .01,
                            'net_arch': [128, 128], 'device': 'cpu'},
                    'observation_normalization': True, 'reward_normalization': False,
                    'git_commit': result.stdout.strip() if result.returncode == 0 else None,
                    'versions': {name: version(name) for name in
                                 ('mujoco', 'gymnasium', 'stable-baselines3', 'numpy', 'torch')}}
        metadata['torch_threads'] = args.torch_threads
        model = PPO('MlpPolicy', vec, seed=args.seed, verbose=1,
                    tensorboard_log=str(args.out / 'tb'),
                    policy_kwargs={'net_arch': [128, 128]},
                    **{k: v for k, v in metadata['ppo'].items() if k != 'net_arch'})
        digest = hashlib.sha256()
        for key, value in sorted(model.policy.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        metadata['initialization'] = {
            'mode': 'from_scratch', 'num_timesteps': model.num_timesteps,
            'observation_count': float(vec.obs_rms.count),
            'policy_sha256': digest.hexdigest()}
        write_json(args.out / 'config.json', metadata)
        started = time.monotonic()
        callback = CheckpointCallback(save_freq=50_000, save_path=str(args.out / 'checkpoints'),
                                      name_prefix='torque_ppo', save_vecnormalize=True)
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(args.out / 'final_model')
        vec.save(args.out / 'vecnormalize.pkl')
        metadata['actual_timesteps'] = model.num_timesteps
        metadata['wall_time_s'] = time.monotonic() - started
        write_json(args.out / 'config.json', metadata)
    finally:
        vec.close()


def checkpoint_paths(run, steps):
    if steps is None:
        return run / 'final_model.zip', run / 'vecnormalize.pkl'
    if steps <= 0:
        raise ValueError('checkpoint steps must be positive')
    return (run / 'checkpoints' / f'torque_ppo_{steps}_steps.zip',
            run / 'checkpoints' / f'torque_ppo_vecnormalize_{steps}_steps.pkl')


def evaluate(args):
    torch.set_num_threads(1)
    if args.episodes <= 0:
        raise ValueError('episodes must be positive')
    metadata = json.loads((args.run / 'config.json').read_text())
    cfg = TorqueConfig(**metadata['environment'])
    targets = [TorqueConfig.validate_target((x, 0., z)) for x, z in args.targets]
    checkpoint_steps = getattr(args, 'checkpoint_steps', None)
    model_path, normalization_path = checkpoint_paths(args.run, checkpoint_steps)
    # Check artifacts before creating output; never silently use fresh normalization.
    if not model_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError(f'missing matching model/normalization pair: {model_path}, {normalization_path}')
    args.out.mkdir(parents=True, exist_ok=False)
    raw = BasketballTorqueEnv(cfg, render_mode='rgb_array' if args.gif else None)
    vec = VecNormalize.load(normalization_path, DummyVecEnv([lambda: raw]))
    vec.training = False
    vec.norm_reward = False
    frames, rows = [], []
    try:
        model = PPO.load(model_path, env=vec, device='cpu')
        for target_index, target in enumerate(targets):
            for episode in range(args.episodes):
                seed = args.seed + target_index * args.episodes + episode
                obs, _ = raw.reset(seed=seed, options={'target': target})
                record_frames = args.gif and target_index == 0 and episode == 0
                if record_frames:
                    frames.append(raw.render())
                total_reward = 0.
                while True:
                    action, _ = model.predict(vec.normalize_obs(obs), deterministic=True)
                    obs, reward, terminated, truncated, info = raw.step(action)
                    total_reward += reward
                    if record_frames:
                        frames.append(raw.render())
                    if terminated or truncated:
                        break
                rows.append({'target_x': float(target[0]), 'target_z': float(target[2]),
                             'seed': seed, 'is_success': info['is_success'],
                             'termination_reason': info['termination_reason'],
                             'episode_return': total_reward,
                             **{key: info[key] for key in METRICS}})
        per_target = {}
        for target in targets:
            subset = [row for row in rows if row['target_x'] == target[0] and row['target_z'] == target[2]]
            per_target[f'{target[0]:g},{target[2]:g}'] = summarize(subset)
        report = {'training_config': metadata, 'evaluation_seed': args.seed,
                  'checkpoint_timesteps': checkpoint_steps if checkpoint_steps is not None else metadata.get('actual_timesteps'),
                  'deterministic_policy': True, 'episodes_per_target': args.episodes,
                  'summary': summarize(rows), 'per_target': per_target, 'episodes': rows}
        write_json(args.out / 'evaluation.json', report)
        with (args.out / 'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if frames:
            import imageio.v2 as imageio
            imageio.mimsave(args.out / 'first_episode.gif', frames, duration=raw.dt * 1000, loop=0)
        print(json.dumps(report['summary'], indent=2, allow_nan=False))
    finally:
        vec.close()


def target_pair(value):
    try:
        pair = tuple(float(item) for item in value.split(','))
        if len(pair) != 2:
            raise ValueError
        TorqueConfig.validate_target((pair[0], 0., pair[1]))
        return pair
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('target must be x,z with x in [0.9,3], z in [0.3,2]') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--preset', choices=PRESETS, default='efficient')
    training.add_argument('--out', type=Path, required=True, help='New directory, never overwritten')
    training.add_argument('--effort-weight', type=float, default=None, help='Override normalized torque penalty')
    training.add_argument('--work-weight', type=float, default=0., help='Absolute mechanical work penalty per joule')
    training.add_argument('--torch-threads', type=int, default=1)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--timesteps', type=int, default=1_000_000)
    training.add_argument('--n-steps', type=int, default=1024)
    training.add_argument('--batch-size', type=int, default=256)
    training.add_argument('--torque-limits', type=float, nargs=2, default=(8., 5.))
    training.add_argument('--target', type=float, nargs=3, default=(1.25, 0., .9))
    training.add_argument('--fixed-target', action='store_true', help='Disable per-reset target sampling')
    training.add_argument('--x-range', type=float, nargs=2, default=(1., 1.5))
    training.add_argument('--z-range', type=float, nargs=2, default=(.75, 1.1))
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--checkpoint-steps', type=int, default=None, help='Evaluate this exact saved checkpoint and matching normalization')
    evaluation.add_argument('--run', type=Path, required=True)
    evaluation.add_argument('--out', type=Path, required=True)
    evaluation.add_argument('--seed', type=int, default=10000)
    evaluation.add_argument('--episodes', type=int, default=1, help='Repeats per fixed target')
    evaluation.add_argument('--targets', type=target_pair, nargs='+',
                            default=[(x, z) for x in (1., 1.25, 1.5) for z in (.75, .9, 1.1)])
    evaluation.add_argument('--gif', action='store_true', help='Render first evaluation episode')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
