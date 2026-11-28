# Basketball Throwing RL

2-DoF planar robot arm with a free ball and fixed-time release. MuJoCo physics, Gymnasium environment, Stable-Baselines3 PPO/SAC, headless GIF export.

## 실행
### 환경 설정

```bash
conda create -n basketball_throwing_rl python=3.12 -y
conda activate basketball_throwing_rl
python -m pip install -r requirements.txt
```

### 학습
#### Tasks
1. target: 특정 점 위치로 공 던지기
2. hoop: 농구슛 모델

```bash
python ./scripts/train.py --algo [ppo,sac] --task [hoop,target] --timesteps [1_000_000]
```

### 평가

```bash
python scripts/evaluate.py --algo [ppo,sac] --model [path_to_parameter] --task [hoop,target] --episodes 100 --seed 10000 --out [path_to_output]
```

### Random policy 기반 action 시각화

```bash
MUJOCO_GL=egl python scripts/play_random.py --episodes 5 --out results/random_policy.gif
```

![random_policy_example.gif](docs/random_policy.gif)

### 학습 결과 시각화

```bash
MUJOCO_GL=egl python scripts/play.py --algo ppo --model [path_to_parameter] --task [hoop,target] --episodes 3 --out [path_to_output]
```

![ppo_result.gif](docs/hoop_ppo.gif)
## 직접 토크 제어와 동작 효율 (PPO)

기존 위치 제어 실험과 별도로, 두 관절의 **토크 + 공 방출 시점**을 학습하는 환경을 추가했습니다. 성공은 림 안쪽 진입 후 공 전체가 림 아래로 통과하는 경우입니다. 골대 거리·높이를 변경하고 성공률, 완료 시간, 누적 토크 제곱, 절대 기계적 일량을 함께 평가합니다.

```bash
python scripts/torque_experiment.py train --preset efficient --out runs/torque_seed42 --seed 42
python scripts/torque_experiment.py evaluate --run runs/torque_seed42 --out runs/torque_eval42
```

`baseline`, `time`, `efficient`의 보상 비교, 지표 단위와 논문과의 차이는 [동작 효율 실험 안내](docs/torque_efficiency.md)를 참고하세요. 테스트는 `python -m pip install pytest` 후 `python -m pytest tests -q`로 실행합니다. 기존 2차원 위치 제어 모델과 새 3차원 토크·방출 모델은 호환되지 않습니다.
