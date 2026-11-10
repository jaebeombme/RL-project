"""Q-network 학습 파일 (8강 예시 코드 q_learning_cartpole.py와 같은 코드, 환경만 MuJoCo)

실행:  python train_qnet.py                  ← MuJoCo로 만든 MyCartPole-v0
       python train_qnet.py --env CartPole-v1 ← 비교: Gymnasium 내장 CartPole
       python train_qnet.py --seed 1          ← 시드 바꾸기
결과:  qnet_best.pt (평가가 가장 좋았던 신경망), figures/learning_curve.png
"""
import argparse
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
import cartpole_env  # noqa: F401  ← "MyCartPole-v0" 등록 (8강 코드와 다른 곳 ①)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MyCartPole-v0")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 하이퍼 매개변수 (8강과 같음) ──
GAMMA = 0.99
EPS_START, EPS_DECAY, EPS_MIN = 1.0, 0.9999, 0.05
LR = 0.01
N_EPISODE = 2000
GOAL = 475.0
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(HERE, "qnet_best.pt")

# ── 환경 (8강 코드와 다른 곳 ②: 이름 한 줄) ──
env = gym.make(args.env)
s_dim = env.observation_space.shape[0]   # 4
a_dim = env.action_space.n               # 2


# ── Q-network: 상태 4개 → 32 → 32 → 행동별 가치 2개 ──
def build_network(s_dim, a_dim):
    return nn.Sequential(
        nn.Linear(s_dim, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, a_dim),
    )


def to_tensor(s):
    return torch.as_tensor(s, dtype=torch.float32)


def select_action(q, eps):
    if np.random.random() < eps:
        return np.random.randint(a_dim)
    return int(torch.argmax(q))


def evaluate(model, n=5):
    eval_env = gym.make(args.env)
    scores = []
    for k in range(n):
        s, info = eval_env.reset(seed=100 + k)
        length, done = 0, False
        while not done:
            with torch.no_grad():
                a = int(torch.argmax(model(to_tensor(s))))
            s, r, terminated, truncated, info = eval_env.step(a)
            length += 1
            done = terminated or truncated
        scores.append(length)
    return float(np.mean(scores))


model = build_network(s_dim, a_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ── 학습: ①~⑧ (8강 그림 6-9) ──
eps, best = EPS_START, -1.0
epi_length = []
start = time.time()
s, info = env.reset(seed=SEED)
for episode in range(1, N_EPISODE + 1):
    if episode > 1:
        s, info = env.reset()
    length = 0
    while True:
        with torch.no_grad():
            y_ = model(to_tensor(s))                          # ① ŷ = q_w(s)
        a = select_action(y_, eps)                            # ② ε-greedy
        s1, r, terminated, truncated, info = env.step(a)      # ③ 실행
        x, y = to_tensor(s), y_.clone()                       # ④ x = s, y = ŷ
        with torch.no_grad():
            y1_ = model(to_tensor(s1))                        # ⑤ ŷ' = q_w(s')
        if terminated:                                        # ⑥ y[a] ← target
            y[a] = r
        else:
            y[a] = r + GAMMA * y1_.max()
        loss = loss_fn(model(x), y)                           # ⑦ 한 번 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s = s1                                                # ⑧ 다음 순간
        length += 1
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if terminated or truncated:
            break
    epi_length.append(length)
    if episode % 10 == 0:
        print(f"{episode}번째 에피소드 길이: {np.mean(epi_length[-10:]):.1f}  (ε = {eps:.3f})")
    if length == 500:
        score = evaluate(model)
        print(f"   {episode}번째 에피소드 후 탐욕 평가: {score:.1f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), SAVE_PATH)
        if score >= GOAL:
            print(f"{episode}번째 에피소드에서 수렴 (평가 평균 {score:.1f} ≥ {GOAL})")
            break

print(f"학습 시간 {time.time() - start:.0f}초, 총 {sum(epi_length)} step, 최고 평가 {best:.1f}")
print("저장:", os.path.relpath(SAVE_PATH, HERE))

# ── 학습 곡선 ──
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
plt.figure(figsize=(7, 4))
plt.plot(epi_length, alpha=0.4, label="episode")
w = 10
if len(epi_length) >= w:
    plt.plot(range(w - 1, len(epi_length)),
             np.convolve(epi_length, np.ones(w) / w, mode="valid"), label="10-episode mean")
plt.xlabel("Episode")
plt.ylabel("Score (steps)")
plt.title(f"Q-network on {args.env} (seed {SEED})")
plt.legend()
plt.savefig(os.path.join(HERE, "figures", "learning_curve.png"), dpi=150, bbox_inches="tight")
print("그림 저장: figures/learning_curve.png")
