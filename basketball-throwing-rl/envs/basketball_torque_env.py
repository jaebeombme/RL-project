"""Direct torque + learned release, with whole-ball scoring and SI cost metrics."""
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .basketball_throw_env import BasketballThrowEnv


@dataclass
class TorqueConfig:
    episode_time: float = 2.0
    control_dt: float = 0.02
    target: tuple = (1.25, 0.0, 0.90)
    target_x_range: tuple | None = None
    target_z_range: tuple | None = None
    torque_limits: tuple = (8.0, 5.0)  # N m; simulation settings, not hardware ratings
    success_margin: float = 0.005
    success_reward: float = 10.0
    failure_penalty: float = 5.0
    distance_weight: float = 0.4
    time_weight: float = 0.1
    effort_weight: float = 0.05
    work_weight: float = 0.0  # reward penalty per joule of absolute mechanical work

    def __post_init__(self):
        for name in ('episode_time', 'control_dt', 'success_reward', 'failure_penalty'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('success_margin', 'distance_weight', 'time_weight', 'effort_weight', 'work_weight'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        limits = np.asarray(self.torque_limits, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
            raise ValueError('torque_limits must contain two positive finite values')
        self.validate_target(self.target)
        for name in ('target_x_range', 'target_z_range'):
            bounds = getattr(self, name)
            if bounds is not None:
                values = np.asarray(bounds, dtype=float)
                if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] > values[1]:
                    raise ValueError(f'{name} must be finite ordered bounds')
                for value in values:
                    target = list(self.target)
                    target[0 if name == 'target_x_range' else 2] = value
                    self.validate_target(target)

    @staticmethod
    def validate_target(target):
        target = np.asarray(target, dtype=float)
        if (target.shape != (3,) or not np.all(np.isfinite(target))
                or abs(target[1]) > 1e-9 or not .9 <= target[0] <= 3.
                or not .3 <= target[2] <= 2.):
            raise ValueError('target must be planar (y=0), x in [0.9, 3], z in [0.3, 2]')
        return target.copy()


class HoopPassage:
    """Conservative inner cylinder through the polygonal rim; no entry, no score.

    Segments are checked at every physics step. A downward center crossing arms
    the detector. Lateral escape or upward recrossing disarms it. Completion is
    only when the ball's top clears the bottom of the rim plus a safety margin.
    """
    def __init__(self, clearance, exit_depth):
        self.clearance = clearance
        self.exit_depth = exit_depth
        self.entered = False

    def update(self, prev, curr, center):
        prev, curr = np.asarray(prev) - center, np.asarray(curr) - center
        dz = curr[2] - prev[2]
        start = prev
        if not self.entered:
            if not (prev[2] >= 0 > curr[2]):
                return False
            start = prev + (-prev[2] / dz) * (curr - prev)
            if np.linalg.norm(start[:2]) > self.clearance:
                return False
            self.entered = True
        if curr[2] >= 0:
            self.entered = False
            return False
        complete = dz < 0 and prev[2] >= -self.exit_depth >= curr[2]
        end = (prev + ((-self.exit_depth - prev[2]) / dz) * (curr - prev)
               if complete else curr)
        if max(np.linalg.norm(start[:2]), np.linalg.norm(end[:2])) > self.clearance:
            self.entered = False
            return False
        return bool(complete)


class BasketballTorqueEnv(BasketballThrowEnv):
    """Actions: normalized shoulder/elbow torque and release (>0 means let go).

    Observation (16): q[2], dq[2], ball xyz[3], ball velocity[3], goal xyz[3],
    released, entered, remaining seconds. Legacy position environments are intact.
    """
    def __init__(self, config=None, render_mode=None):
        cfg = config or TorqueConfig()
        super().__init__(config=cfg, render_mode=render_mode,
                         xml_path=Path(__file__).resolve().parents[1] / 'assets' / 'planar_torque.xml')
        h = self.model.opt.timestep
        if not np.isclose(self.dt, cfg.control_dt, atol=1e-10, rtol=0):
            raise ValueError('control_dt must be a positive multiple of physics timestep')
        if not np.isclose(cfg.episode_time / h, round(cfg.episode_time / h)):
            raise ValueError('episode_time must be a multiple of physics timestep')
        self.max_physics_steps = round(cfg.episode_time / h)
        self.torque_limits = np.asarray(cfg.torque_limits, dtype=float)
        self.model.actuator_ctrlrange[:] = np.column_stack((-self.torque_limits, self.torque_limits))
        self.ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        rim_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'rim_00')
        self.ball_radius = float(self.model.geom_size[self.ball_geom, 0])
        self.rim_radius = float(self.model.geom_size[rim_geom, 0])
        # 12 chord segments, not a perfect circle. Their apothem bounds the opening.
        self.clearance = .16 * np.cos(np.pi / 12) - self.rim_radius - self.ball_radius - cfg.success_margin
        if self.clearance <= 0:
            raise ValueError('success_margin leaves no usable hoop aperture')
        self.exit_depth = self.ball_radius + self.rim_radius + cfg.success_margin
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        self._done = True
        self.passage = HoopPassage(self.clearance, self.exit_depth)
        self.physics_steps = 0
        self.release_time = None
        self.success_time = None
        self.effort = self.normalized_effort = self.work = 0.

    def _obs(self):
        return np.concatenate((
            self.data.qpos[self.qadr], self.data.qvel[self.vadr],
            self._ball_position(), self._ball_velocity(), self.target,
            [float(self.released), float(self.passage.entered),
             max(0., self.cfg.episode_time - self.data.time)],
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        target = np.asarray(self.cfg.target, dtype=float).copy()
        if options and 'target' in options:
            target = TorqueConfig.validate_target(options['target'])
        else:
            for index, bounds in ((0, self.cfg.target_x_range), (2, self.cfg.target_z_range)):
                if bounds is not None:
                    target[index] = self.np_random.uniform(*bounds)
        self.target = target
        self.passage = HoopPassage(self.clearance, self.exit_depth)
        # Reuse joint placement, then match the torque model's weld transform.
        # Its offset includes a gap so release cannot inject overlap energy.
        super().reset(seed=None)
        link2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link2')
        offset = self.model.eq_data[self.weld_id, 3:6]
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = (
            self.data.xpos[link2] + self.data.xmat[link2].reshape(3, 3) @ offset)
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = self.data.xquat[link2]
        self.data.ctrl[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.physics_steps = 0
        self.release_time = self.success_time = None
        self.effort = self.normalized_effort = self.work = 0.
        self.best_dist = float(np.linalg.norm(self._ball_position() - self.target))
        self._done = False
        return self._obs(), self._info(False, 'running')

    def _info(self, success, reason):
        return {
            'is_success': bool(success), 'released': self.released,
            'hoop_entered': self.passage.entered, 'termination_reason': reason,
            'target': self.target.copy(), 'distance': float(np.linalg.norm(self._ball_position() - self.target)),
            'best_distance': self.best_dist, 'elapsed_steps': self.step_count,
            'elapsed_time_s': float(self.data.time), 'release_time_s': self.release_time,
            'success_time_s': self.success_time,
            'torque_squared_integral': self.effort,
            'normalized_effort': self.normalized_effort, 'absolute_work_j': self.work,
        }

    def _floor_contact(self):
        return any({int(c.geom1), int(c.geom2)} == {self.ball_geom, self.floor_geom}
                   for c in self.data.contact[:self.data.ncon])

    def step(self, action):
        if self._done:
            raise RuntimeError('reset() is required before stepping a completed episode')
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError('action must contain three finite numbers')
        action = np.clip(action, -1., 1.)
        self.data.ctrl[:] = action[:2] * self.torque_limits
        if not self.released and action[2] > 0:
            self.released = True
            self.release_time = float(self.data.time)
            self.data.eq_active[self.weld_id] = 0
        start_time = self.data.time
        start_effort = self.normalized_effort
        start_work = self.work
        prev = self._ball_position()
        initial_distance = float(np.linalg.norm(prev - self.target))
        success, reason = False, 'running'
        h = self.model.opt.timestep
        for _ in range(min(self.frame_skip, self.max_physics_steps - self.physics_steps)):
            # Refresh derived state before measuring actual actuator torque/velocity.
            mujoco.mj_forward(self.model, self.data)
            tau = self.data.qfrc_actuator[self.vadr].copy()
            velocity = self.data.qvel[self.vadr].copy()
            self.effort += float(tau @ tau) * h
            self.normalized_effort += float(np.sum((tau / self.torque_limits) ** 2)) * h
            self.work += float(np.sum(np.abs(tau * velocity))) * h
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            # mj_step integrates qpos; derived xpos must correspond to the new qpos.
            mujoco.mj_forward(self.model, self.data)
            curr = self._ball_position()
            if self.released:
                self.best_dist = min(self.best_dist, float(np.linalg.norm(curr - self.target)))
                success = self.passage.update(prev, curr, self.target)
            if success:
                self.success_time = float(self.data.time)
                reason = 'success'
            elif self._floor_contact() or curr[2] <= self.ball_radius:
                reason = 'floor'
            elif curr[0] < -1.5 or curr[0] > 3.5 or abs(curr[1]) > 1.:
                reason = 'out_of_bounds'
            prev = curr
            if reason != 'running':
                break
        self.step_count += 1
        terminated = reason != 'running'
        truncated = not terminated and self.physics_steps >= self.max_physics_steps
        if truncated:
            reason = 'timeout' if self.released else 'no_release'
        self._done = terminated or truncated
        progress = (initial_distance - float(np.linalg.norm(prev - self.target))) if self.released else 0.
        distance_reward = self.cfg.distance_weight * progress
        time_cost = self.cfg.time_weight * (self.data.time - start_time)
        effort_cost = self.cfg.effort_weight * (self.normalized_effort - start_effort)
        work_cost = self.cfg.work_weight * (self.work - start_work)
        terminal_reward = self.cfg.success_reward if success else (-self.cfg.failure_penalty if self._done else 0.)
        reward = distance_reward + terminal_reward - time_cost - effort_cost - work_cost
        info = self._info(success, reason)
        info.update(distance_reward=distance_reward, time_cost=time_cost,
                    effort_cost=effort_cost, work_cost=work_cost, terminal_reward=terminal_reward)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
