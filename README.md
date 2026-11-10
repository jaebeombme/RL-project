# simlab — MuJoCo로 CartPole·이륜차 만들고 Q-network로 학습하기

```
simlab/
├─ requirements.txt      필요한 패키지 목록
├─ check_setup.py        준비 점검 (13개 항목)
├─ cartpole/             ← CartPole: 모델링 + 환경 + 학습 + 시험 (이 폴더에서 바로 실행)
│   ├─ assets/           로봇 설계도 XML 4단계
│   │   ├─ step1_world.xml   빈 세상 (시간 간격·빛·카메라·바닥)
│   │   ├─ step2_cart.xml    + 카트 (slide 관절)
│   │   ├─ step3_pole.xml    + 막대 (hinge 관절)
│   │   └─ cartpole.xml      + 모터 = 완성본
│   ├─ build_steps.py    XML 4단계를 차례로 불러 시험하고 그림 저장
│   ├─ cartpole_env.py   MuJoCo CartPole → Gymnasium 환경 "MyCartPole-v0"
│   ├─ test_env.py       학습 전 환경 점검 (규칙 검사, 무작위 행동)
│   ├─ train_qnet.py     Q-network 학습 (8강 코드와 같음) → qnet_best.pt
│   └─ evaluate.py       저장한 신경망 시험 → figures/qnet_test.gif
└─ car/                  ← 이륜차 (2륜 차동구동)
    ├─ car_model.py      이륜차 XML을 만드는 함수 make_car_xml(), 설계 3종 그림
    ├─ drive_test.py     고정 명령으로 5초 주행 → 궤적 그림
    ├─ car_env.py        차선 따라가기 Gymnasium 환경 "MyCar-v0"
    └─ train_car.py      (도전) 이륜차 Q-network 학습
```

## 0. 설치 (conda, PC당 한 번)

Windows는 시작 메뉴의 **Miniforge Prompt**(또는 Anaconda Prompt), macOS는 터미널.
conda가 없으면 https://github.com/conda-forge/miniforge 에서 Miniforge를 설치합니다.

```bash
conda create -n simlab python=3.12 -y
conda activate simlab                 # 프롬프트 앞이 (simlab)으로 바뀌면 성공 — 터미널 열 때마다
cd simlab
python -m pip install -r requirements.txt
python check_setup.py                 # 전부 [OK]면 준비 끝
```

## 1. CartPole (cartpole 폴더에서)

```bash
cd cartpole
python build_steps.py     # XML 4단계 확인 + figures/*.png
python cartpole_env.py    # 환경 한 판 (무작위 행동)
python test_env.py        # 환경 점검
python train_qnet.py      # 학습 (약 20초) → qnet_best.pt, figures/learning_curve.png
python evaluate.py        # 시험 5판 + figures/qnet_test.gif
```

- `python train_qnet.py --env CartPole-v1` : 같은 코드로 Gymnasium 내장 CartPole 학습 (비교용)
- 모델을 마우스로 돌려 보기: `python -m mujoco.viewer --mjcf assets/cartpole.xml`
  (macOS는 `mjpython -m mujoco.viewer --mjcf assets/cartpole.xml`)

## 2. 이륜차 (car 폴더에서)

```bash
cd car
python car_model.py       # 설계 3종 정보 + figures/car_designs.png
python drive_test.py      # 5초 주행 3가지 + figures/drive_paths.png
python car_env.py         # 차선 따라가기 환경 점검
python train_car.py       # (도전) 학습 약 30초 → figures/car_paths.png
```

## 확인한 결과 (준비 PC: Linux, CPU 2코어, Python 3.13, MuJoCo 3.17.0·4.0.0 모두 같은 값)

| 실행 | 결과 |
|---|---|
| build_steps.py | 카트 1초 뒤 x = 0.975 m · 막대 12°를 넘는 시각 0.55 s · ctrl +1 0.3초 뒤 x = 0.436 m, θ = −39.8° |
| test_env.py | 무작위 행동 평균 길이 MyCartPole-v0 22.4, CartPole-v1 22.1 |
| train_qnet.py | 369번째 에피소드에서 수렴, 총 24,711 step, 약 20초 |
| evaluate.py | 시험 5판 모두 500 |
| drive_test.py | (1,1) x 2.92 m·y 0.00 m / (1,0.6) x −0.53·y −0.97 / 약한 모터 x 1.18·y −1.91 |
| train_car.py | 380번째 에피소드에서 목표(평가 평균 ≥ 190) 달성, 시험 보상 합 197.6 / 200 |

## 자주 나는 문제

- `ModuleNotFoundError` → `conda activate simlab` 을 했는지(프롬프트 앞 괄호) 확인 후 다시 설치
- `No registered env with id: MyCartPole-v0` → `cartpole` 폴더 안에서 실행했는지 확인
- 그림·GIF 저장만 실패 → 원격 서버 등 화면이 없는 PC. 학습은 그대로 됩니다 (Linux 서버는 `MUJOCO_GL=egl`)
- Intel Mac: 최신 mujoco·torch 휠이 없음 → `conda create -n simlab python=3.11 -y` 후
  `python -m pip install "mujoco==3.10.0" "torch==2.2.2" "numpy<2" "gymnasium[classic-control]==1.0.0" matplotlib`
