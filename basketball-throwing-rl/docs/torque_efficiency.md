# 토크 제어 PPO: 동작 효율 실험

이 실험의 질문은 **성공률을 유지하면서 더 짧은 시간과 더 적은 제어 노력으로 던질 수 있는가?** 입니다. 일정 성공률까지 필요한 학습 스텝(학습 효율)을 주된 성과로 주장하지 않습니다. 현재 코드는 실험 도구이며, 효율 개선 결과를 미리 보장하지 않습니다.

## 환경

기존 `train.py`, `BasketballThrowEnv`, `BasketballHoopEnv`와 위치 제어 XML은 그대로 유지합니다. 새 `BasketballTorqueEnv`는 `assets/planar_torque.xml`의 두 motor actuator를 사용합니다.

- 물리 주기: 0.002 s, 기본 제어 주기: 0.02 s. 최대 2 s / 100 제어 스텝.
- 행동 `Box(-1, 1, (3,))`: `[어깨 정규화 토크, 팔꿈치 정규화 토크, 방출]`.
- 토크: `tau = action[:2] * [8, 5]` N m. gear=1 직접 제어이며 위치 서보가 없습니다. 한계는 조정 가능한 시뮬레이션 초기값이며 실제 하드웨어 사양이 아닙니다.
- 부착: 공 중심을 팔꿈치 링크의 로컬 x=0.43 m에 배치합니다. 팔 끝과 공 표면 사이에 5 mm 간격을 두고 weld의 위치·자세를 일치시켜, 초기 겹침이 방출 때 인위적인 반발력을 만들지 않게 합니다.
- 방출: 세 번째 출력이 `> 0`이면 해당 제어 스텝 **시작 시점**에 weld를 해제합니다. 고정 타이머, 강제 방출, 재부착은 없습니다. 공의 위치·속도를 인위적으로 바꾸지 않습니다.
- 관측 16개: 관절 각도(2), 각속도(2), 공 위치(3), 공 선속도(3), 골대 위치(3), 방출 여부(1), 림 진입 여부(1), 남은 초(1). 모든 위치·속도는 월드 좌표입니다. 공 회전 상태는 관측에 포함하지 않으므로 접촉 후 회전 효과까지 완전한 상태 표현은 아닙니다.
- 학습 CLI 기본 골대: 매 에피소드 `x ~ U(1.0,1.5)`, `z ~ U(0.75,1.1)`, `y=0`. 범위와 고정 골대는 CLI로 선택합니다. 목표 body를 이동하므로 표시와 림 충돌 형상이 함께 이동합니다.
- 설정 허용 범위 `x∈[0.9,3], z∈[0.3,2]`는 공간 검증 범위이며 모든 점의 도달 가능성을 보장하지 않습니다. 더 넓은 범위는 별도 도달성 확인 후 사용하세요.

### 공 전체 통과 판정

물리 스텝마다 공의 이동 구간을 검사합니다.

1. 방출된 공 중심이 림 평면을 **위에서 아래로** 가로지르고, 보수적으로 정한 내부 구멍 안에 있어야 진입으로 기록합니다.
2. 공이 내부 통과 영역에 머무르는 동안만 진입 상태를 유지합니다. 옆으로 빠져나가거나 위로 다시 올라오면 무효입니다.
3. 공 중심이 `rim_z - (ball_radius + rim_tube_radius + margin)` 아래에 도달하면 성공합니다. 즉 공 윗부분이 림 아랫면과 여유 간격 아래에 있습니다.

림은 반지름 0.16 m의 12각형 capsule입니다. 통과 허용 중심 반경은 `0.16*cos(pi/12) - 0.012 - 0.045 - 0.005 ≈ 0.09255 m`입니다. 원형 림으로 잘못 가정하지 않으며, 실제 농구의 모든 림 튕김/가장자리 성공을 인정하는 정교한 심판 모델은 아닙니다. 내부 수직 원기둥을 사용하는 보수적 기준입니다. 네트와 백보드는 없습니다.

바닥에 먼저 닿거나 영역을 벗어나면 실패 종료(`terminated`). 시간 제한은 `truncated`; 그때까지 방출하지 않았으면 `no_release` 실패입니다. 실패에도 명시적 패널티가 있어 공을 놓지 않거나 빨리 실패해 비용만 줄이는 전략을 억제합니다. 성공 이후 팔의 정지·회복 동작은 현재 과제에 포함되지 않습니다.

## 보상과 독립 평가 지표

제어 스텝 보상:

```
r = 10 * success - 5 * terminal_failure
    + 0.4 * released * (distance_before - distance_after)
    - time_weight * elapsed_seconds_this_step
    - effort_weight * integral(sum((actual_tau / torque_limit)^2), dt)
```

거리 항은 방출 후에만 적용하며 공을 들고 골대에 접근하는 보상 악용을 줄입니다. 보상 계수는 실험 시작값입니다. 거리 shaping은 PPO 할인율에 대해 정책 불변성을 보장하는 potential shaping은 아니며, 세 실험에 동일하게 적용합니다. PPO의 `gamma=0.99`도 조기 성공을 선호할 수 있으므로 baseline은 '시간 선호가 전혀 없는 모델'이 아니라 **명시적 시간·노력 비용이 없는 모델**입니다.

| preset | 시간 비용 가중치 | 제어 노력 비용 가중치 |
|---|---:|---:|
| baseline | 0 | 0 |
| time | 0.1 | 0 |
| efficient | 0.1 | 0.05 |

세 조건은 같은 토크 환경, 성공 판정, PPO 설정을 사용합니다. 성공률과 효율의 trade-off를 보여주려면 후속 실험에서 비용 계수도 변화시키세요.

보상 가중치가 0이어도 다음 지표는 항상 기록합니다.

| 필드 | 의미 / 단위 |
|---|---|
| is_success | 공 전체 통과 여부 |
| elapsed_steps | 실행한 제어 호출 횟수, 마지막 호출은 조기 종료 가능 |
| elapsed_time_s | 실제 시뮬레이션 시간, s |
| success_time_s | 성공 시점, 실패는 null |
| release_time_s | 방출 시점, 미방출은 null; 즉시 방출은 0 |
| torque_squared_integral | `∫sum(tau_i²)dt`, (N m)² s |
| normalized_effort | `∫sum((tau_i/limit_i)²)dt`, s |
| absolute_work_j | `∫sum(abs(tau_i*dq_i))dt`, J |

제어 노력과 일량은 **실제 actuator 토크**를 물리 스텝 시작 상태에서 읽어 좌측 직사각형 적분합니다. 토크 제곱은 에너지가 아닙니다. absolute_work_j도 회생·전기 손실을 모델링한 배터리 소모가 아니라 절대 기계적 일량의 근사입니다. 시간은 정확한 누적 물리 스텝이며 비용 적분은 수치 근사입니다. 방출 이후에도 종료까지 팔 제어 비용을 포함합니다.

## 실행

Python 3.10 이상 권장. 저장소의 `basketball-throwing-rl` 디렉터리에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests -q

# 빠른 설치/학습 경로 점검: 성능 실험이 아닙니다.
python scripts/torque_experiment.py train --out runs/torque_smoke \
  --timesteps 2048 --n-steps 128 --batch-size 64 --seed 42

# 고정 골대부터 학습하는 경우
python scripts/torque_experiment.py train --out runs/fixed_efficient_seed42 \
  --preset efficient --fixed-target --target 1.25 0 0.9 \
  --timesteps 1000000 --seed 42

# 본 비교: 세 preset과 여러 독립 학습 seed에 같은 예산/골대 범위를 사용
for seed in 0 1 2; do
  for preset in baseline time efficient; do
    python scripts/torque_experiment.py train \
      --out "runs/${preset}_seed${seed}" --preset "$preset" \
      --seed "$seed" --timesteps 1000000
    python scripts/torque_experiment.py evaluate \
      --run "runs/${preset}_seed${seed}" --out "runs/eval_${preset}_seed${seed}" \
      --seed 10000
  done
done

# 별도 미학습 위치에서 평가 + 첫 에피소드 GIF
python scripts/torque_experiment.py evaluate \
  --run runs/efficient_seed0 --out runs/eval_heldout_seed0 \
  --targets 1.1,0.82 1.35,1.02 1.45,0.95 --gif
```

출력 디렉터리는 덮어쓰지 않습니다. 학습 중 관측만 VecNormalize로 정규화하고 보상은 정규화하지 않습니다. `final_model.zip`과 `vecnormalize.pkl`은 반드시 한 쌍으로 보관합니다. `config.json`에 환경·PPO 설정, seed, 버전, 실제 학습 스텝을 저장합니다. PPO rollout 크기로 인해 실제 학습 스텝은 요청 예산보다 약간 커질 수 있습니다. 중간 checkpoint에는 같은 스텝 번호의 VecNormalize 파일이 함께 저장됩니다. 기본 평가 CLI는 최종 모델 쌍을 사용합니다.

평가 결과는 `episodes.csv`, `evaluation.json`입니다. 기본 3×3 골대 격자에서 deterministic policy로 평가하며, 각 위치별 결과와 전체 결과를 모두 저장합니다. 실패를 포함한 성공률·비용과 **성공한 에피소드에 한정한** 시간·비용 평균을 함께 제공합니다. 성공이 없으면 성공 조건부 평균은 0이 아니라 null입니다. JSON의 std는 에피소드 값의 모집단 표준편차이며 신뢰구간이 아닙니다.

현재 초기 자세와 동역학은 고정이므로 같은 모델·같은 골대의 deterministic 반복은 사실상 중복입니다. `--episodes`를 늘려 이것을 독립 표본으로 해석하면 안 됩니다. 여러 독립 **학습 seed**와 다양한 평가 골대가 중요합니다. 세 모델 모두 성공한 공통 목표 집합에서도 시간·비용을 비교하고, 각 모델이 실패한 목표를 반드시 따로 보고하세요. 저장된 per-episode 행으로 공통 성공 집합을 구성할 수 있습니다. 평균 return은 preset마다 보상 함수가 달라 직접적인 효율 비교 지표가 아닙니다.

## 참고 자료와 차이

- 강의 자료 `RL_8강_실습_Simulation.pdf`: MJCF로 형상을 만들고 MuJoCo 물리를 Gymnasium reset/step으로 감싸는 구성 참고. 강의의 CartPole Q-network를 이 실험의 알고리즘으로 사용하지 않습니다.
- Pahič et al., *Robot skill learning in latent space of a deep autoencoder neural network*, Robotics and Autonomous Systems 135 (2021), 103690. DOI: https://doi.org/10.1016/j.robot.2020.103690. 논문은 MuJoCo의 평면 투척, 거리·높이 목표, DMP/AE 잠재공간과 PoWER 계열 학습을 다룹니다. 3개 관절을 사용하며 공은 홀더에서 수동적으로 이탈합니다. 본 구현의 **2관절 직접 토크 PPO, 명시적 방출 행동, 동작 효율 비용은 별도 설계**입니다. 논문의 표본 효율 결과를 본 코드의 동작 효율 근거로 가져오지 않습니다.
- PPO / 지원 행동 공간: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- MuJoCo actuator/제약: https://mujoco.readthedocs.io/en/stable/XMLreference.html

본 PR 검증은 환경 동작과 학습 파이프라인의 실행 가능성까지입니다. 수렴, 높은 성공률, 동작 효율 개선은 충분한 학습과 위 비교 실험 후 판단해야 합니다.

## From-scratch split-cost study (2026-11-27)

PR #1's historical fine-tuning results do not answer a from-scratch reward comparison. The new study initializes PPO weights, optimizer and VecNormalize from zero for **every** run. Three seeds (42,43,44) are crossed with four conditions:

| condition | normalized torque weight | absolute work weight (per J) |
|---|---:|---:|
| none | 0 | 0 |
| torque | 0.5 | 0 |
| work | 0 | 0.02 |
| both | 0.5 | 0.02 |

The time penalty remains 0.1 in all conditions (`time` preset). Weights correspond to a reward penalty of one at C=2 s or W_abs=50 J; they are fixed initial choices, not tuned optima. Reward subtracts each physics-integrated cost's increment for the current control interval, not the episode cumulative total again. `work_weight=0` preserves older run configurations.

```bash
python scripts/torque_study.py --out runs/scratch-study --workers 6 --timesteps 1000000 --seeds 42 43 44
```

Each run trains **a total** of 1,000,448 steps (1,000,000 rounded to a 1,024-step PPO rollout), never an additional budget on pretrained weights. Goals are randomized on every reset: x=[1,1.5], z=[0.75,1.1], y=0. `protocol.json` freezes 300 evaluation targets before starting. Evaluate saved checkpoints at 100k,250k,500k,750k and final using each checkpoint's matching VecNormalize. No best-checkpoint selection, and no evaluation-driven reward changes.

`config.json` records initial policy SHA256, observation count=0.0001, and num_timesteps=0. `initialization-audit.json` verifies matched initial parameters across conditions within a seed, distinct parameters across seeds, and no warmed normalization. Existing run directories are never overwritten. Any failed run remains explicit in `status.json` and makes the study fail; no seed is silently excluded.

Report per-seed success and mean/sample SD across training seeds. Do not treat all episodes pooled across seeds as independent training replications. Compare costs on common successful goals within each seed as well as all-episode and individual-success summaries; no common successes means missing efficiency evidence, not zero cost. Absolute mechanical work is not electrical energy consumption.

After all runs finish, summarize all predeclared checkpoints without selection:

```bash
python scripts/summarize_torque_study.py runs/scratch-study
```

The summarizer requires the complete initialization audit and writes summary.json/CSV, a learning curve, and report.md. Missing runs/checkpoints fail rather than being excluded.
