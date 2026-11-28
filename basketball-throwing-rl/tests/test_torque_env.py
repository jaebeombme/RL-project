import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from envs.basketball_torque_env import BasketballTorqueEnv, TorqueConfig, HoopPassage


def test_agent_controls_release_and_reset_restores_hold():
    env = BasketballTorqueEnv()
    env.reset(seed=1)
    # Upright zero-torque equilibrium: no hidden 0.5s release timer.
    for _ in range(30):
        _, _, terminated, truncated, info = env.step([0, 0, -1])
        assert not terminated and not truncated and not info['released']
    _, _, _, _, info = env.step([0, 0, 1])
    assert info['released'] and info['release_time_s'] == pytest.approx(.6)
    assert env.data.eq_active[env.weld_id] == 0
    env.step([0, 0, -1])
    assert env.data.eq_active[env.weld_id] == 0
    env.reset()
    assert env.data.eq_active[env.weld_id] == 1
    assert env.release_time is None


def test_actual_torque_and_integrated_effort():
    env = BasketballTorqueEnv(TorqueConfig(torque_limits=(2., 3.)))
    env.reset()
    _, _, _, _, info = env.step([1, -1, -1])
    np.testing.assert_allclose(env.data.qfrc_actuator[env.vadr], [2, -3])
    assert info['torque_squared_integral'] == pytest.approx(.26)
    assert info['normalized_effort'] == pytest.approx(.04)
    assert info['absolute_work_j'] >= 0
    assert info['elapsed_time_s'] == pytest.approx(.02)
    assert info['elapsed_steps'] == 1


def test_hold_timeout_is_failure_not_free_zero_reward():
    env = BasketballTorqueEnv(TorqueConfig(episode_time=.04))
    env.reset()
    env.step([0, 0, -1])
    _, reward, term, trunc, info = env.step([0, 0, -1])
    assert trunc and not term and not info['is_success']
    assert info['termination_reason'] == 'no_release'
    assert reward < 0
    with pytest.raises(RuntimeError):
        env.step([0, 0, 1])


def test_whole_ball_passage_not_only_center():
    detector = HoopPassage(clearance=.09, exit_depth=.062)
    center = np.array([1.25, 0, .9])
    def point(z, x=0):
        return center + [x, 0, z]
    assert not detector.update(point(.1), point(-.01), center)
    assert detector.entered
    assert not detector.update(point(-.01), point(-.04), center)
    assert detector.update(point(-.04), point(-.07), center)


@pytest.mark.parametrize('start,end', [
    ([1.25, 0, .8], [1.25, 0, .7]),  # Already below: never entered.
    ([1.25, 0, .8], [1.25, 0, 1.0]), # Upward crossing.
    ([1.5, 0, 1.0], [1.5, 0, .7]),   # Outside aperture.
])
def test_invalid_passages(start, end):
    detector = HoopPassage(.09, .062)
    assert not detector.update(np.array(start), np.array(end), np.array([1.25, 0, .9]))


def test_fast_crossing_and_lateral_escape():
    center = np.array([0., 0., 1.])
    detector = HoopPassage(.09, .062)
    assert detector.update(np.array([0., 0., 1.2]), np.array([0., 0., .8]), center)
    detector = HoopPassage(.09, .062)
    assert not detector.update(np.array([0., 0., 1.1]), np.array([0., 0., .99]), center)
    assert not detector.update(np.array([0., 0., .99]), np.array([.2, 0., .9]), center)
    assert not detector.entered


def test_seeded_targets_and_override_move_collision_geometry():
    env = BasketballTorqueEnv(TorqueConfig(target_x_range=(1., 1.5), target_z_range=(.8, 1.1)))
    obs, _ = env.reset(seed=42)
    first = env.target.copy()
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs, obs2)
    np.testing.assert_array_equal(first, env.target)
    obs, _ = env.reset(options={'target': (1.4, 0, 1.)})
    np.testing.assert_allclose(env.data.xpos[env.target_id], [1.4, 0, 1.])
    np.testing.assert_allclose(obs[10:13], [1.4, 0, 1.])
    with pytest.raises(ValueError):
        env.reset(options={'target': (1.4, .5, 1.)})


@pytest.mark.parametrize('action', [[0, 0], [0, 0, np.nan], [0, np.inf, 1]])
def test_invalid_actions_do_not_advance_simulation(action):
    env = BasketballTorqueEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.data.time == 0


def test_env_contract_and_finite_rollouts():
    env = BasketballTorqueEnv()
    check_env(env, skip_render_check=True)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        for _ in range(110):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            if term or trunc:
                break
        assert term or trunc


def test_real_physics_drop_requires_whole_ball_clearance():
    env = BasketballTorqueEnv()
    env.reset()
    env.released = True
    env.data.eq_active[env.weld_id] = 0
    env.data.qpos[env.ball_qadr:env.ball_qadr+3] = env.target + [0, 0, .10]
    env.data.qvel[env.ball_vadr:env.ball_vadr+6] = [0, 0, -1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    for _ in range(30):
        _, _, term, trunc, info = env.step([0, 0, -1])
        if term or trunc:
            break
    assert info['is_success']
    assert env._ball_position()[2] + env.ball_radius < env.target[2] - env.rim_radius
    assert info['success_time_s'] > 0


def test_costs_integrate_same_physical_duration_across_control_rates():
    results = []
    for dt in (.01, .02):
        env = BasketballTorqueEnv(TorqueConfig(control_dt=dt, torque_limits=(2., 3.)))
        env.reset()
        for _ in range(round(.04/dt)):
            _, _, term, trunc, info = env.step([.5, -1, -1])
            assert not term and not trunc
        results.append(info)
    assert results[0]['torque_squared_integral'] == pytest.approx(.4)
    assert results[1]['torque_squared_integral'] == pytest.approx(.4)
    assert results[0]['absolute_work_j'] == pytest.approx(results[1]['absolute_work_j'])
    assert results[0]['elapsed_steps'] == 4 and results[1]['elapsed_steps'] == 2


def test_cost_weights_change_reward_not_dynamics_or_metrics():
    baseline = BasketballTorqueEnv(TorqueConfig(time_weight=0., effort_weight=0.))
    efficient = BasketballTorqueEnv(TorqueConfig(time_weight=.1, effort_weight=.05))
    baseline.reset(seed=0)
    efficient.reset(seed=0)
    obs1, r1, _, _, i1 = baseline.step([.5, .5, -1])
    obs2, r2, _, _, i2 = efficient.step([.5, .5, -1])
    np.testing.assert_array_equal(obs1, obs2)
    assert r1 - r2 == pytest.approx(.0025)
    assert i1['torque_squared_integral'] == i2['torque_squared_integral']
    assert i1['absolute_work_j'] == i2['absolute_work_j']


@pytest.mark.parametrize('kwargs', [
    {'control_dt': .0001}, {'control_dt': .003}, {'episode_time': 0},
    {'torque_limits': (0, 1)}, {'time_weight': -1}, {'target': (1, 1, 1)},
    {'target_x_range': (2, 1)}, {'target_z_range': (.1, 1)},
    {'success_margin': .2}, {'effort_weight': float('nan')},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BasketballTorqueEnv(TorqueConfig(**kwargs))


def test_zero_torque_release_has_no_overlap_or_artificial_launch_impulse():
    env = BasketballTorqueEnv()
    env.reset()
    for contact in env.data.contact[:env.data.ncon]:
        assert env.ball_geom not in (contact.geom1, contact.geom2) or contact.dist >= 0
    start = env._ball_position().copy()
    env.step([0, 0, 1])
    np.testing.assert_allclose(env._ball_velocity(), [0, 0, -.1962], atol=1e-5)
    np.testing.assert_allclose(env._ball_position() - start, [0, 0, -.001962], atol=1e-6)


def test_split_costs_change_rewards_not_physics_and_use_step_deltas():
    configs = [(0., 0.), (.5, 0.), (0., .02), (.5, .02)]
    envs = [BasketballTorqueEnv(TorqueConfig(effort_weight=t, work_weight=w)) for t, w in configs]
    try:
        for env in envs:
            env.reset(seed=17)
        previous_c = previous_w = 0.
        for _ in range(5):
            results = [env.step([.7, -.4, -1]) for env in envs]
            base = results[0]
            c = base[4]['normalized_effort'] - previous_c
            w = base[4]['absolute_work_j'] - previous_w
            assert w > 0
            for (t_weight, w_weight), result in zip(configs, results):
                np.testing.assert_array_equal(result[0], base[0])
                assert result[1] == pytest.approx(base[1] - t_weight*c - w_weight*w)
                assert result[4]['work_cost'] == pytest.approx(w_weight*w)
                assert result[4]['effort_cost'] == pytest.approx(t_weight*c)
            previous_c = base[4]['normalized_effort']
            previous_w = base[4]['absolute_work_j']
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_work_weight_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        TorqueConfig(work_weight=value)
