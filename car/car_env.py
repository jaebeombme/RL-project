"""이륜차 차선 따라가기 Gymnasium 환경 → 이름 "MyCar-v0"

오른쪽 모터가 20% 약한 차(right_gear=0.16)가 y = 0 차선을 따라 똑바로 가도록 배우는 과업.
같은 명령 (1, 1)을 계속 주면 오른쪽으로 휘어 버린다 (drive_test.py 참고).

  상태 s  = [y, ψ, ψ̇]   차선에서 벗어난 거리 [m], 진행 방향 각도 [rad], 회전 속도 [rad/s]
  행동 a  = 0 왼쪽으로 (0.5, 1) / 1 직진 (1, 1) / 2 오른쪽으로 (1, 0.5)   ← [왼 모터, 오른 모터]
  보상 r  = 1 - 2|y|     (차선 위면 1, 0.5 m 벗어나면 0)
  종료    = |y| > 0.5 m  또는  |ψ| > 90°  → terminated
  시간 제한 = 200 step (0.1초 × 200 = 20초) → truncated
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from car_model import make_car_xml

ACTIONS = np.array([[0.5, 1.0],    # 0: 왼쪽으로 꺾기
                    [1.0, 1.0],    # 1: 직진 명령
                    [1.0, 0.5]])   # 2: 오른쪽으로 꺾기


def yaw_from_quat(q):
    """쿼터니언 [w, x, y, z] → 위에서 본 회전 각도 ψ (z축 회전)"""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MyCarEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, right_gear=0.16):
        self.model = mujoco.MjModel.from_xml_string(make_car_xml(right_gear=right_gear))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 20                     # 0.005초 × 20 = 0.1초마다 행동
        self.y_limit = 0.5
        big = np.finfo(np.float32).max
        high = np.array([2 * self.y_limit, np.pi, big], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self._renderer = None

    def _obs(self):
        y = self.data.qpos[1]                    # freejoint qpos = [x, y, z, 쿼터니언 4개]
        psi = yaw_from_quat(self.data.qpos[3:7])
        psi_dot = self.data.qvel[5]              # freejoint qvel = [선속도 3, 각속도 3]
        return np.array([y, psi, psi_dot], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        y0 = self.np_random.uniform(-0.05, 0.05)        # 차선 근처에서
        psi0 = self.np_random.uniform(-0.1, 0.1)        # 약간 비틀어진 방향으로 시작
        self.data.qpos[1] = y0
        self.data.qpos[3:7] = [np.cos(psi0 / 2), 0, 0, np.sin(psi0 / 2)]
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = ACTIONS[action]
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = self._obs()
        y, psi = obs[0], obs[1]
        terminated = bool(abs(y) > self.y_limit or abs(psi) > np.pi / 2)
        reward = 1.0 - 2.0 * min(abs(y), self.y_limit)
        return obs, float(reward), terminated, False, {}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=640)
        self._renderer.update_scene(self.data, camera="follow")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="MyCar-v0", entry_point=MyCarEnv, max_episode_steps=200)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = gym.make("MyCar-v0")
    check_env(env.unwrapped, skip_render_check=True)
    print("[OK] Gymnasium 규칙 검사 통과")
    for name, policy in [("항상 직진 명령 (1, 1)", lambda s: 1),
                         ("무작위 행동", lambda s: env.action_space.sample())]:
        s, info = env.reset(seed=0)
        env.action_space.seed(0)
        total, steps = 0.0, 0
        while True:
            s, r, terminated, truncated, info = env.step(policy(s))
            total, steps = total + r, steps + 1
            if terminated or truncated:
                break
        print(f"{name:20s}: {steps:3d} step 버팀, 보상 합 {total:6.1f}, 마지막 y = {s[0]:+.2f} m")
