"""이륜차 차선 따라가기를 Q-network로 학습 (CartPole의 train_qnet.py와 같은 학습 코드)

실행:  python train_car.py            (시간이 몇 분 걸린다)
결과:  car_qnet_best.pt, figures/car_learning_curve.png, figures/car_paths.png
CartPole과 다른 곳: 환경 이름, 상태 3개·행동 3개, 점수 = 보상 합(최대 200), 시간 제한 200 step
"""
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import car_env  # noqa: F401  ← "MyCar-v0" 등록

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GAMMA = 0.99
EPS_START, EPS_DECAY, EPS_MIN = 1.0, 0.9995, 0.05   # 에피소드가 짧아 ε를 조금 빨리 줄임
LR = 0.001
N_EPISODE = 1000
GOAL = 190.0                                  # 평가 보상 합 평균 (최대 200)
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(HERE, "car_qnet_best.pt")

env = gym.make("MyCar-v0")
s_dim = env.observation_space.shape[0]        # 3: [y, ψ, ψ̇]
a_dim = env.action_space.n                    # 3: 왼쪽 / 직진 / 오른쪽


def build_network(s_dim, a_dim):
    return nn.Sequential(
        nn.Linear(s_dim, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, a_dim),
    )


def to_tensor(s):
    return torch.as_tensor(s, dtype=torch.float32)


def run_greedy(model, seed, record=False):
    """탐험 없이 한 판 → (보상 합, 지나간 x·y 좌표)"""
    e = gym.make("MyCar-v0")
    s, info = e.reset(seed=seed)
    total, path = 0.0, []
    while True:
        with torch.no_grad():
            a = int(torch.argmax(model(to_tensor(s))))
        s, r, terminated, truncated, info = e.step(a)
        total += r
        if record:
            path.append(e.unwrapped.data.body("car").xpos[:2].copy())
        if terminated or truncated:
            break
    return total, np.array(path)


def evaluate(model, n=5):
    return float(np.mean([run_greedy(model, 100 + k)[0] for k in range(n)]))


model = build_network(s_dim, a_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

eps, best = EPS_START, -1.0
scores = []
start = time.time()
s, info = env.reset(seed=SEED)
for episode in range(1, N_EPISODE + 1):
    if episode > 1:
        s, info = env.reset()
    total = 0.0
    while True:
        with torch.no_grad():
            y_ = model(to_tensor(s))                          # ①
        a = np.random.randint(a_dim) if np.random.random() < eps else int(torch.argmax(y_))  # ②
        s1, r, terminated, truncated, info = env.step(a)      # ③
        x, y = to_tensor(s), y_.clone()                       # ④
        with torch.no_grad():
            y1_ = model(to_tensor(s1))                        # ⑤
        y[a] = r if terminated else r + GAMMA * y1_.max()     # ⑥
        loss = loss_fn(model(x), y)                           # ⑦
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s = s1                                                # ⑧
        total += r
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if terminated or truncated:
            break
    scores.append(total)
    if episode % 10 == 0:
        print(f"{episode}번째 에피소드 보상 합: {np.mean(scores[-10:]):6.1f}  (ε = {eps:.3f})")
    if truncated:                                 # 20초를 버텼으면 탐험 없이 평가
        score = evaluate(model)
        print(f"   {episode}번째 에피소드 후 탐욕 평가: {score:.1f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), SAVE_PATH)
        if score >= GOAL:
            print(f"{episode}번째 에피소드에서 목표 달성 (평가 평균 {score:.1f} ≥ {GOAL})")
            break

print(f"학습 시간 {time.time() - start:.0f}초, 최고 평가 {best:.1f}")
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

plt.figure(figsize=(7, 4))
plt.plot(scores, alpha=0.4, label="episode")
if len(scores) >= 10:
    plt.plot(range(9, len(scores)), np.convolve(scores, np.ones(10) / 10, mode="valid"),
             label="10-episode mean")
plt.xlabel("Episode")
plt.ylabel("Return (max 200)")
plt.title("Q-network on MyCar-v0 (lane keeping, weak right motor)")
plt.legend()
plt.savefig(os.path.join(HERE, "figures", "car_learning_curve.png"), dpi=150, bbox_inches="tight")

# 학습 전 (항상 직진 명령) vs 학습 후 (저장한 최고 신경망) 궤적 비교
if os.path.exists(SAVE_PATH):
    model.load_state_dict(torch.load(SAVE_PATH))
    total, path = run_greedy(model, seed=1000, record=True)
    e = gym.make("MyCar-v0")
    e.reset(seed=1000)
    naive = []
    while True:
        _, _, te, tr, _ = e.step(1)
        naive.append(e.unwrapped.data.body("car").xpos[:2].copy())
        if te or tr:
            break
    naive = np.array(naive)
    print(f"시험(seed 1000): 학습한 신경망 보상 합 {total:.1f}, 최종 x = {path[-1, 0]:.2f} m, "
          f"|y| 최대 {np.abs(path[:, 1]).max():.2f} m")
    plt.figure(figsize=(7, 3))
    plt.axhline(0, color="green", lw=3, alpha=0.4, label="lane (y = 0)")
    plt.plot(naive[:, 0], naive[:, 1], label="always (1, 1)")
    plt.plot(path[:, 0], path[:, 1], label="trained Q-network")
    plt.ylim(-0.6, 0.6)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(HERE, "figures", "car_paths.png"), dpi=150, bbox_inches="tight")
    print("그림 저장: figures/car_learning_curve.png, figures/car_paths.png")
