"""학습한 Q-network 시험 파일: 탐험 없이(argmax) 5판

실행:  python evaluate.py                     ← 시험 5판 + 첫 판을 GIF로 저장
       python evaluate.py --view              ← MuJoCo 창을 띄워 실제 속도로 보기
       python evaluate.py --env CartPole-v1   ← 내장 CartPole로 학습했다면 (--view 가능)
       (macOS에서 --view는  mjpython evaluate.py --view)
결과:  시험 에피소드 길이 5개, figures/qnet_test.gif
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cartpole_env  # noqa: F401  ← "MyCartPole-v0" 등록

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MyCartPole-v0")
parser.add_argument("--model", default="qnet_best.pt")
parser.add_argument("--view", action="store_true", help="시뮬레이션 창을 띄워서 보기")
parser.add_argument("--no-gif", action="store_true")
args = parser.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
MUJOCO_ENV = args.env == "MyCartPole-v0"


def build_network(s_dim, a_dim):            # train_qnet.py와 같은 구조여야 불러올 수 있다
    return nn.Sequential(
        nn.Linear(s_dim, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, a_dim),
    )


# ── 환경 만들기 ──
make_gif = (not args.no_gif) and (not args.view) and MUJOCO_ENV
if args.view and not MUJOCO_ENV:
    env = gym.make(args.env, render_mode="human")        # 내장 CartPole은 자체 창(pygame)
else:
    env = gym.make(args.env, render_mode="rgb_array" if make_gif else None)

model = build_network(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load(os.path.join(HERE, args.model)))   # 저장한 신경망 불러오기
model.eval()

# ── MuJoCo 창 열기 (--view) ──
viewer = None
if args.view and MUJOCO_ENV:
    import mujoco
    import mujoco.viewer
    sim = env.unwrapped                                  # MyCartPoleEnv (model, data를 가짐)
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data,   # 창만 열고 계산은 우리가
                                          show_left_ui=False, show_right_ui=False)
    viewer.cam.lookat[:] = [0, 0, 1.0]                   # 옆에서 보는 시점 (마우스로 돌려도 됨)
    viewer.cam.distance = 4.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10
    print("창에서: 마우스로 시점 회전 · 막대 더블클릭 후 Ctrl+오른쪽 드래그로 밀기 · Tab으로 메뉴 · 창 닫으면 종료")

lengths, frames = [], []
for k in range(5):
    s, info = env.reset(seed=1000 + k)       # 학습·평가에 쓰지 않은 초기 상태
    length = 0
    if viewer is not None:
        viewer.sync()
        time.sleep(0.5)                       # 시작 자세를 잠깐 보여 주기
    while True:
        t0 = time.time()
        with torch.no_grad():
            q = model(torch.as_tensor(s, dtype=torch.float32))
        a = int(torch.argmax(q))             # 탐험 없이 탐욕 선택
        s, r, terminated, truncated, info = env.step(a)
        length += 1
        if make_gif and k == 0 and length % 2 == 0:
            frames.append(env.render())       # 첫 판만, 2 step마다 한 장
        if viewer is not None:
            if not viewer.is_running():       # 창을 닫으면 멈춤
                break
            viewer.sync()                     # 지금 상태를 창에 그리기
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))   # 1 step = 0.02초 → 실제 속도
        if terminated or truncated:
            break
    lengths.append(length)
    print(f"{k + 1}번째 판: {length} step")
    if viewer is not None and not viewer.is_running():
        break
print("시험 에피소드 길이:", lengths)

if frames:
    from PIL import Image
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    path = os.path.join(HERE, "figures", "qnet_test.gif")
    images = [Image.fromarray(f[::2, ::2]) for f in frames]     # 절반 크기로 줄이기
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=40, loop=0)                         # 40 ms × 2 step = 실제 속도
    print("GIF 저장: figures/qnet_test.gif")
if viewer is not None:
    viewer.close()
env.close()
