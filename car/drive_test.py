"""이륜차 주행 시험: 모터 명령을 고정하고 5초 동안 달려 본다 (학습 없음)

실행:  python drive_test.py
  (1, 1)            두 바퀴 같은 명령 → 직진
  (1, 0.6)          왼쪽이 더 빠름   → 오른쪽으로 회전
  (1, 1) 약한 모터  오른쪽 모터 20% 약함 → 같은 명령인데 오른쪽으로 휜다
결과 숫자를 출력하고, 위에서 본 궤적을 figures/drive_paths.png로 저장한다.
"""
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from car_model import make_car_xml


def drive(ctrl, seconds=5.0, **design):
    model = mujoco.MjModel.from_xml_string(make_car_xml(**design))
    data = mujoco.MjData(model)
    xy = []
    while data.time < seconds:
        data.ctrl[:] = ctrl                      # [왼쪽, 오른쪽] 모터 명령
        mujoco.mj_step(model, data)
        xy.append(data.body("car").xpos[:2].copy())
    return np.array(xy)


runs = {
    "(1, 1)": drive([1, 1]),
    "(1, 0.6)": drive([1, 0.6]),
    "(1, 1) weak right motor": drive([1, 1], right_gear=0.16),   # 20% 약함
}

plt.figure(figsize=(6, 4.5))
for name, xy in runs.items():
    print(f"{name:24s} 5초 뒤 x = {xy[-1, 0]:5.2f} m,  y = {xy[-1, 1]:5.2f} m")
    plt.plot(xy[:, 0], xy[:, 1], label=name)
plt.axhline(0, color="green", lw=3, alpha=0.4, label="lane (y = 0)")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("5-second paths (top view)")
plt.legend(fontsize=8)
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
plt.savefig(os.path.join(here, "figures", "drive_paths.png"), dpi=150, bbox_inches="tight")
print("그림 저장: figures/drive_paths.png")
