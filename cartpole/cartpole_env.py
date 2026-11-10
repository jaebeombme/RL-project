"""MuJoCo CartPole을 Gymnasium 환경으로 감싼 파일 → 이름 "MyCartPole-v0"

다른 파일에서  import cartpole_env  한 줄이면 gym.make("MyCartPole-v0")를 쓸 수 있다.
직접 실행하면(python cartpole_env.py) 무작위 행동으로 한 판 돌려 본다.

  상태 s  = [x, θ, ẋ, θ̇]  (qpos 2개 + qvel 2개)
  행동 a  = 0 (왼쪽 -10 N) / 1 (오른쪽 +10 N)
  보상 r  = 버틴 step마다 +1
  종료    = |θ| > 12°  또는  |x| > 2.4 m   → terminated
  시간 제한 = 500 step (10초)               → truncated (gym.make가 붙여 줌)
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cartpole.xml")


class MyCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(XML)   # 설계도 (변하지 않음)
        self.data = mujoco.MjData(self.model)            # 지금 상태 (매 step 바뀜)
        self.frame_skip = 2                  # 0.01초 × 2 = 0.02초마다 행동 (CartPole-v1과 같음)
        self.theta_limit = 12 * np.pi / 180  # 12° = 0.2095 rad
        self.x_limit = 2.4
        big = np.finfo(np.float32).max       # 속도에는 한계가 없으므로 아주 큰 값
        high = np.array([2 * self.x_limit, 2 * self.theta_limit, big, big], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.render_mode = render_mode
        self._renderer = None

    def _obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)             # self.np_random 시드 설정
        mujoco.mj_resetData(self.model, self.data)
        # CartPole-v1처럼 네 값 모두 ±0.05 안에서 무작위로 시작
        self.data.qpos[:] = self.np_random.uniform(-0.05, 0.05, size=2)
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, size=2)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[0] = 1.0 if action == 1 else -1.0   # 힘 = ctrl × gear(10)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)           # 물리 계산은 MuJoCo가
        obs = self._obs()
        x, theta = obs[0], obs[1]
        terminated = bool(abs(theta) > self.theta_limit or abs(x) > self.x_limit)
        reward = 1.0                                        # 보상·종료는 우리가 정한다
        return obs, reward, terminated, False, {}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=640)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# import 하는 순간 이름이 등록된다 → gym.make("MyCartPole-v0")
gym.register(id="MyCartPole-v0", entry_point=MyCartPoleEnv, max_episode_steps=500)


if __name__ == "__main__":
    env = gym.make("MyCartPole-v0")
    s, info = env.reset(seed=0)
    print("첫 상태 [x, θ, ẋ, θ̇] =", s)
    total = 0
    while True:
        s, r, terminated, truncated, info = env.step(env.action_space.sample())
        total += r
        if terminated or truncated:
            break
    print(f"무작위 행동으로 {int(total)} step 버팀 (terminated={terminated})")
