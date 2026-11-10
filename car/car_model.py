"""이륜차(2륜 차동구동) MJCF를 만드는 파일

make_car_xml(...) 함수가 설계 변수(바퀴 반지름, 폭, 차체 길이, 모터 세기 ...)를 받아
XML 문자열을 돌려준다. 숫자만 바꿔서 여러 차를 만들 수 있다.

직접 실행하면(python car_model.py) 세 가지 설계를 만들어
몸체·자유도 정보를 출력하고 figures/car_designs.png 그림을 저장한다.
"""
import os
import numpy as np
import mujoco


def make_car_xml(wheel_radius=0.06, track=0.26, body_len=0.30, body_mass=1.0,
                 left_gear=0.2, right_gear=0.2, wheel_damping=0.02):
    r = wheel_radius          # 바퀴 반지름 [m]
    hx = body_len / 2         # 차체 길이의 절반
    half = track / 2          # 바퀴 사이 거리(폭)의 절반
    rc = 0.015                # 캐스터(앞 받침 공) 반지름
    return f"""
<mujoco model="car">
  <option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".85 .92 1" rgb2="1 1 1" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".92 .92 .92" rgb2=".82 .82 .82" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="20 20"/>
  </asset>
  <worldbody>
    <light pos="0 -2 6" dir="0 .3 -1" directional="true"/>
    <geom name="floor" type="plane" size="20 20 .1" material="grid"/>
    <!-- 차선: y = 0 을 따라 그린 초록 선 (보이기만, 충돌 끔) -->
    <geom name="lane" type="box" pos="10 0 .001" size="10 .015 .001"
          rgba=".2 .75 .3 1" contype="0" conaffinity="0"/>
    <camera name="top" pos="2 -1 9" xyaxes="1 0 0 0 1 0" fovy="45"/>

    <!-- 차체: freejoint → 바닥 위를 자유롭게 (위치 3 + 자세 3) -->
    <body name="car" pos="0 0 {r}">
      <freejoint/>
      <geom name="chassis" type="box" size="{hx} .09 .03" mass="{body_mass}" rgba=".2 .45 .85 1"/>
      <geom name="nose" type="box" pos="{hx - 0.02} 0 .032" size=".02 .07 .004"
            rgba="1 .8 .1 1" mass="0" contype="0" conaffinity="0"/>
      <camera name="follow" mode="track" pos="-.9 -.9 .7" xyaxes=".7 -.7 0 .3 .3 .9"/>

      <!-- 바퀴: z축을 세계 y축으로 눕힌 body + hinge (body 기준 z축 = 차축) -->
      <body name="left_wheel" pos="{-hx / 3} {half} 0" zaxis="0 1 0">
        <joint name="left" type="hinge" axis="0 0 1" damping="{wheel_damping}"/>
        <geom type="ellipsoid" size="{r} {r} .02" mass=".1" rgba=".1 .1 .1 1"/>
      </body>
      <body name="right_wheel" pos="{-hx / 3} {-half} 0" zaxis="0 1 0">
        <joint name="right" type="hinge" axis="0 0 1" damping="{wheel_damping}"/>
        <geom type="ellipsoid" size="{r} {r} .02" mass=".1" rgba=".1 .1 .1 1"/>
      </body>

      <!-- 캐스터: 앞쪽을 받치는 마찰 없는 공 (condim=1 → 미끄러짐) -->
      <geom name="caster" type="sphere" pos="{hx * 0.8} 0 {-(r - rc)}" size="{rc}"
            mass=".03" condim="1" priority="1" rgba=".6 .6 .6 1"/>
    </body>
  </worldbody>

  <!-- 모터 두 개: 바퀴 토크 = ctrl(-1~1) x gear [N·m] -->
  <actuator>
    <motor name="left"  joint="left"  gear="{left_gear}"  ctrlrange="-1 1" ctrllimited="true"/>
    <motor name="right" joint="right" gear="{right_gear}" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


DESIGNS = {
    "기본": dict(),
    "큰 바퀴·넓은 폭": dict(wheel_radius=0.09, track=0.40),
    "긴 차체·작은 바퀴": dict(body_len=0.50, wheel_radius=0.045),
}

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "figures"), exist_ok=True)
    images = []
    for name, kw in DESIGNS.items():
        model = mujoco.MjModel.from_xml_string(make_car_xml(**kw))
        data = mujoco.MjData(model)
        mass = model.body_subtreemass[model.body("car").id]
        print(f"[{name}] 질량 {mass:.2f} kg  자유도(nv) {model.nv}  "
              f"qpos {model.nq}개  모터 {model.nu}개")
        try:
            mujoco.mj_forward(model, data)
            with mujoco.Renderer(model, height=300, width=400) as rd:
                rd.update_scene(data, camera="follow")
                images.append(rd.render())
        except Exception as e:
            print("  (그림 건너뜀:", type(e).__name__, ")")
    if images:
        from PIL import Image
        Image.fromarray(np.concatenate(images, axis=1)).save(os.path.join(here, "figures", "car_designs.png"))
        print("그림 저장: figures/car_designs.png")
