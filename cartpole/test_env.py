"""환경 점검 파일: 학습 전에 환경이 규칙대로 동작하는지 확인

실행:  python test_env.py
  1) Gymnasium 규칙 검사 (check_env)
  2) reset · step이 돌려주는 값 출력
  3) 무작위 행동 20판 평균 길이 → CartPole-v1과 비교
"""
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import cartpole_env  # noqa: F401  ← 이 import가 "MyCartPole-v0"을 등록한다

env = gym.make("MyCartPole-v0")

# 1) 규칙 검사: reset/step 반환값의 모양·타입이 맞는지
check_env(env.unwrapped, skip_render_check=True)
print("[OK] Gymnasium 규칙 검사 통과")

# 2) 한 step 주고받기
print("관측 공간:", env.observation_space)
print("행동 공간:", env.action_space)
s, info = env.reset(seed=0)
s1, r, terminated, truncated, info = env.step(1)
print("s  =", s)
print("s' =", s1, " r =", r, " terminated =", terminated, " truncated =", truncated)


# 3) 무작위 행동 20판
def random_lengths(env_id, n=20):
    e = gym.make(env_id)
    lengths = []
    for k in range(n):
        e.reset(seed=k)
        e.action_space.seed(k)
        length = 0
        while True:
            _, _, terminated, truncated, _ = e.step(e.action_space.sample())
            length += 1
            if terminated or truncated:
                break
        lengths.append(length)
    return np.mean(lengths)


print(f"무작위 행동 평균 길이  MyCartPole-v0: {random_lengths('MyCartPole-v0'):.1f}"
      f"   CartPole-v1: {random_lengths('CartPole-v1'):.1f}")
