"""실습 준비 점검: simlab 폴더에서  python check_setup.py

파이썬 · 패키지 · 파일 · MuJoCo · Gymnasium 환경 · Q-network를 차례로 검사한다.
모두 [OK]면 준비 끝. [X]가 나오면 그 줄의 안내를 따른다.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, fn, hint=""):
    try:
        msg = fn()
        print(f"[OK] {name}" + (f" — {msg}" if msg else ""))
        results.append(True)
    except Exception as e:
        print(f"[X]  {name} — {type(e).__name__}: {e}")
        if hint:
            print(f"     → {hint}")
        results.append(False)


def python_version():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} (3.10 이상 필요)")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def in_env():
    prefix = os.environ.get("CONDA_DEFAULT_ENV") or ("venv" if sys.prefix != sys.base_prefix else "")
    if not prefix or prefix == "base":
        raise RuntimeError("base 환경 또는 시스템 파이썬에서 실행 중")
    return f"환경 이름: {prefix}"


def version(mod):
    def f():
        m = __import__(mod)
        return m.__version__
    return f


def files():
    need = ["cartpole/assets/cartpole.xml", "cartpole/cartpole_env.py", "cartpole/train_qnet.py",
            "car/car_model.py", "car/car_env.py"]
    missing = [p for p in need if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return f"{len(need)}개 확인"


def mujoco_step():
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "cartpole", "assets", "cartpole.xml"))
    d = mujoco.MjData(m)
    for _ in range(30):
        d.ctrl[0] = 1.0
        mujoco.mj_step(m, d)
    return f"0.3초 뒤 카트 x = {d.qpos[0]:.3f} m (정답 0.436)"


def make_env():
    sys.path.insert(0, os.path.join(HERE, "cartpole"))
    import gymnasium as gym
    import cartpole_env  # noqa: F401
    env = gym.make("MyCartPole-v0")
    s, _ = env.reset(seed=0)
    env.step(0)
    return f"관측 {s.shape}, 행동 {env.action_space}"


def builtin_cartpole():
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    env.step(0)
    return "CartPole-v1 OK"


def qnet_speed():
    import numpy as np
    import torch
    import torch.nn as nn
    import gymnasium as gym
    net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    env = gym.make("MyCartPole-v0")
    s, _ = env.reset(seed=0)
    t = time.time()
    for _ in range(500):
        q = net(torch.as_tensor(s))
        s, r, te, tr, _ = env.step(int(np.random.randint(2)))
        loss = (q.max() - 1.0) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if te or tr:
            s, _ = env.reset()
    return f"{(time.time() - t) / 500 * 1000:.2f} ms/step"


def render():
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "cartpole", "assets", "cartpole.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, height=120, width=160) as r:
        r.update_scene(d, camera="side")
        img = r.render()
    return f"그림 {img.shape[1]}×{img.shape[0]}"


print("=== simlab 실습 준비 점검 ===")
check("파이썬 버전", python_version, "conda create -n simlab python=3.12 -y")
check("실습 전용 환경에서 실행 중", in_env, "conda activate simlab 을 먼저 실행")
for mod in ["numpy", "mujoco", "gymnasium", "torch", "matplotlib"]:
    check(mod, version(mod), "python -m pip install -r requirements.txt")
check("실습 파일", files, "simlab 폴더 안에서 실행했는지 확인")
check("MuJoCo 물리 계산", mujoco_step)
check("MyCartPole-v0 만들기", make_env)
check("Gymnasium 내장 CartPole-v1", builtin_cartpole, "python -m pip install \"gymnasium[classic-control]\"")
check("Q-network 학습 속도", qnet_speed)
check("MuJoCo 그림 그리기", render, "그림·GIF만 안 됨. 학습은 그대로 가능")

n_ok = sum(results)
print(f"\n{'전부 통과' if n_ok == len(results) else '확인 필요'} ({n_ok}/{len(results)})")
