"""XML 네 단계로 CartPole을 쌓아 보는 파일 (슬라이드: CartPole 구현 1~4단계)

실행:  python build_steps.py
하는 일: assets/ 의 XML 4개를 차례로 읽어
  (1) 몸체·관절·모터 개수와 qpos 길이를 출력하고
  (2) 단계마다 짧은 시험을 돌려 숫자로 확인하고
  (3) figures/stepN.png 그림을 저장한다.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    """assets/name.xml → (MjModel 설계도, MjData 현재 상태)"""
    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "assets", name + ".xml"))
    data = mujoco.MjData(model)
    return model, data


def info(name, model):
    print(f"\n[{name}.xml] body {model.nbody}  joint {model.njnt}  "
          f"qpos {model.nq}  actuator {model.nu}")


def save_png(model, data, filename):
    """옆 카메라("side")로 지금 장면을 그려 PNG로 저장"""
    try:
        from PIL import Image
        mujoco.mj_forward(model, data)           # 시간은 그대로, 위치만 다시 계산
        with mujoco.Renderer(model, height=360, width=640) as r:
            r.update_scene(data, camera="side")
            Image.fromarray(r.render()).save(os.path.join(FIG, filename))
        print("  그림 저장:", os.path.join("figures", filename))
    except Exception as e:                       # 그림이 안 돼도 숫자 시험은 계속
        print("  (그림 저장 건너뜀:", type(e).__name__, ")")


# 1단계: 빈 세상 ─ 움직일 것이 없다
model, data = load("step1_world")
info("step1_world", model)
print("  qpos =", data.qpos, "← 관절이 없어서 상태가 없다")
save_png(model, data, "step1_world.png")

# 2단계: 카트 ─ 처음 속도 1 m/s로 밀고 1초
model, data = load("step2_cart")
info("step2_cart", model)
data.qvel[0] = 1.0
while data.time < 1.0 - 1e-9:
    mujoco.mj_step(model, data)
print(f"  1초 뒤 x = {data.qpos[0]:.3f} m  (damping이 없으면 1.000)")
save_png(model, data, "step2_cart.png")

# 3단계: 막대 ─ 0.05 rad(2.9°) 기울여 놓고 손을 뗀다
model, data = load("step3_pole")
info("step3_pole", model)
data.qpos[1] = 0.05
cross = None
while data.time < 0.8 - 1e-9:
    mujoco.mj_step(model, data)
    if cross is None and abs(data.qpos[1]) > np.deg2rad(12):
        cross = data.time
    if abs(data.time - 0.6) < 1e-9:
        print(f"  0.6초: θ = {np.rad2deg(data.qpos[1]):.1f}°")
print(f"  0.8초: θ = {np.rad2deg(data.qpos[1]):.1f}°,  12°를 넘은 시각 {cross:.2f} s")
save_png(model, data, "step3_pole.png")

# 4단계: 모터 ─ ctrl = +1 (= +10 N)을 0.3초
model, data = load("cartpole")
info("cartpole", model)
for _ in range(30):                              # 0.01초 × 30 = 0.3초
    data.ctrl[0] = 1.0
    mujoco.mj_step(model, data)
print(f"  0.3초 뒤 x = {data.qpos[0]:.3f} m,  θ = {np.rad2deg(data.qpos[1]):.1f}°")
save_png(model, data, "cartpole.png")
