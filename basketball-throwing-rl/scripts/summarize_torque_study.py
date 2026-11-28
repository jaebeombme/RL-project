"""Report every predeclared seed/condition/checkpoint, never select best runs."""
import csv,json,sys
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rl-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if len(sys.argv) != 2:
    raise SystemExit('Usage: python summarize_torque_study.py STUDY_DIRECTORY')
out=Path(sys.argv[1])
p=json.loads((out/'protocol.json').read_text())
conditions=list(p['conditions']);seeds=p['seeds'];steps=[*p['checkpoint_steps'],'final']
audit=json.loads((out/'initialization-audit.json').read_text())
if not audit['passed'] or audit['runs'] != len(conditions)*len(seeds):
    raise ValueError('complete fresh-initialization audit required before reporting')
metrics=['elapsed_steps','elapsed_time_s','normalized_effort','torque_squared_integral','absolute_work_j']
def mean(values):return float(np.mean(values)) if values else None
def stats(values):
 values=[v for v in values if v is not None]
 return {'n_seeds':len(values),'mean':mean(values),'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None}
result={'protocol':p,'checkpoints':{}}
flat=[]
for step in steps:
 records={}
 for s in seeds:
  reports={c:json.loads((out/f'{c}-seed{s}-eval-{step}'/'evaluation.json').read_text()) for c in conditions}
  rows={c:r['episodes'] for c,r in reports.items()}
  reference=[(r['target_x'],r['target_z'],r['seed']) for r in rows[conditions[0]]]
  assert len(reference)==300
  assert all([(r['target_x'],r['target_z'],r['seed']) for r in rows[c]]==reference for c in conditions)
  common=[i for i in range(300) if all(rows[c][i]['is_success'] for c in conditions)]
  pair=[i for i in range(300) if rows['none'][i]['is_success'] and rows['both'][i]['is_success']]
  for c,r in reports.items():
   su=r['summary'];entry={'seed':s,'successes':su['successes'],'success_rate':su['success_rate'],
     'actual_checkpoint_steps':r['checkpoint_timesteps'],
     'all_episodes':{m:su['all_episodes'][m]['mean'] for m in metrics},
     'successful_episodes':{m:su['successful_episodes'][m]['mean'] for m in metrics},
     'common_success_count':len(common),
     'common_success':{m:mean([rows[c][i][m] for i in common]) for m in metrics},
     'none_both_common_count':len(pair),
     'none_both_common':{m:mean([rows[c][i][m] for i in pair]) for m in metrics} if c in ['none','both'] else None}
   records.setdefault(c,[]).append(entry)
   flat.append({'checkpoint':step,'condition':c,'seed':s,'successes':su['successes'],'success_rate':su['success_rate'],**entry['successful_episodes']})
 result['checkpoints'][str(step)]={c:{'seeds':rs,'success_rate':stats([r['success_rate'] for r in rs]),
   'common_success':{m:stats([r['common_success'][m] for r in rs]) for m in metrics},
   'successful_episodes':{m:stats([r['successful_episodes'][m] for r in rs]) for m in metrics}} for c,rs in records.items()}
(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
with (out/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
fig,ax=plt.subplots(figsize=(9,5))
colors=['#62748a','#227cc4','#dd8c21','#3b9f70']
x=[result['checkpoints'][str(st)][conditions[0]]['seeds'][0]['actual_checkpoint_steps'] for st in steps]
for c,color in zip(conditions,colors):
 y=[result['checkpoints'][str(st)][c]['success_rate']['mean']*100 for st in steps]
 sd=[result['checkpoints'][str(st)][c]['success_rate']['sample_std']*100 for st in steps]
 ax.plot(x,y,'o-',label=c,color=color);ax.fill_between(x,np.maximum(0,np.array(y)-sd),np.minimum(100,np.array(y)+sd),color=color,alpha=.12)
ax.set(xlabel='Total training control steps from scratch',ylabel='Success (%) on fixed 300 targets',ylim=(0,102),title=f'Mean ± sample SD across {len(seeds)} training seeds')
ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'learning-curve.png',dpi=160);plt.close(fig)
lines=['# 처음부터 학습한 토크·기계적 일 보상 비교','', f'기존 미세조정 모델을 재사용하지 않은 {len(conditions)}조건 × {len(seeds)}seed 실험. 모든 run은 새 PPO/optimizer/VecNormalize에서 시작하여 총 {x[-1]:,}스텝 학습했다. 골대는 매 에피소드 x=[1,1.5], z=[.75,1.1], y=0에서 무작위 배치된다. 초기 가중치 해시 및 정규화 통계 감사 결과는 initialization-audit.json에 있다.','', '## 최종 성공률 (각 seed 300개 동일 목표)','', '| 조건 | '+' | '.join(f'seed{s}' for s in seeds)+' | 평균 ± seed 간 표본 표준편차 |','|---|'+''.join('---:|' for _ in range(len(seeds)+1))]
final=result['checkpoints']['final']
for c in conditions:
 z=final[c];cells=[f'{r["successes"]}/300 ({r["success_rate"]*100:.1f}%)' for r in z['seeds']];st=z['success_rate']
 lines.append('| '+c+' | '+' | '.join(cells)+f' | {st["mean"]*100:.2f} ± {st["sample_std"]*100:.2f}% |')
lines+=['','## 고정 시점별 성공률','', '| 총 스텝 | none | torque | work | both |','|---|---:|---:|---:|---:|']
for st in steps:
 cells=[]
 for c in conditions:
  z=result['checkpoints'][str(st)][c]['success_rate'];cells.append(f'{z["mean"]*100:.2f} ± {z["sample_std"]*100:.2f}%')
 lines.append('| '+str(st)+' | '+' | '.join(cells)+' |')
lines+=['','## 동작 효율: 동일 seed의 4조건 공통 성공 목표','', 'seed별 공통 성공 목표에서 평균을 먼저 계산한 다음, seed별 평균을 동등한 가중치로 집계했다. 공통 성공이 없는 seed는 null이며 0으로 대체하지 않는다. 표본 수가 줄면 효율 비교의 근거가 제한된다.','', '| 조건 | 유효 seed | 시간(s) | 정규화 토크 비용(s) | 절대 기계적 일(J) |','|---|---:|---:|---:|---:|']
def format_stat(z):return 'NA' if z['mean'] is None else f'{z["mean"]:.4f} ± {z["sample_std"]:.4f}' if z['sample_std'] is not None else f'{z["mean"]:.4f} (n=1)'
for c in conditions:
 z=final[c]['common_success']; lines.append(f'| {c} | {z["elapsed_time_s"]["n_seeds"]} | '+ ' | '.join(format_stat(z[m]) for m in ['elapsed_time_s','normalized_effort','absolute_work_j'])+' |')
lines+=['','공통 성공 목표 수: '+', '.join(f'seed{r["seed"]}: {r["common_success_count"]}/300' for r in final['none']['seeds'])+'.','', '비용 없음과 둘 다 조건의 쌍별 공통 성공 비교 및 각 조건의 전체·성공 에피소드 평균은 summary.json에 별도 저장했다. 실패/성공 대상 구성이 다르면 단순 평균 비용 차이를 효율 개선으로 해석하지 않는다.','', '## 보상·재현·한계','', '- 보상: 성공 +10, 실패 -5, 방출 후 거리 감소 ×0.4, 시간 벌점 0.1/s 유지. 토크 가중치 0 또는 0.5; 절대 기계적 일 가중치 0 또는 0.02/J. 설정을 결과에 맞춰 사후 변경하지 않았다.','- 제어 스텝 0.02s, 물리 스텝 0.002s, 최대2s. 재타격 허용 및 공 전체 림 하향 통과 성공 기준 유지.','- 가중치·optimizer·정규화의 warm start 없음. 100k/250k/500k/750k/최종 체크포인트를 각각 대응하는 정규화 파일과 로드했다. 마지막 성능만으로 비교하지 않으며 best checkpoint/seed 선택 없음.','- 동일 평가 목표를 고정 시점들에 재사용했으므로 학습 경과 진단용 고정 평가 세트다. 이후 튜닝한다면 독립 최종 테스트가 추가로 필요하다.','- 학습 seed는 3개뿐이며 일반적 통계 우월성을 주장하기 어렵다. 900개 에피소드를 독립 학습 반복으로 취급하지 않는다.','- 절대 기계적 일은 전기 소비 에너지가 아니다. 모든 비용은 에피소드 종료까지 적분한다.','', '[학습 곡선](learning-curve.png) · [집계 JSON](summary.json) · [집계 CSV](summary.csv) · [고정 프로토콜](protocol.json)']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({c:final[c]['success_rate'] for c in conditions},indent=2))
