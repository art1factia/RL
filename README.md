# RL Particle Control

Deep Q-Network (DQN)을 사용한 2D 파티클 분수 제어 시스템

## 프로젝트 개요

본 프로젝트는 강화학습(DQN)을 사용하여 2D 공간에서 파티클 분수의 발사 각도와 속도를 제어함으로써 지정된 타겟 영역을 최대한 커버하는 에이전트를 개발합니다.

**주요 특징:**
- 물리 기반 시뮬레이션 (중력, 마찰)
- 4가지 실제 시나리오 (원형, 벽면, 바닥, 분수 아치)
- DQN 기반 강화학습
- 다층 보상 함수로 안정적인 학습

**성능:**
- Ground 시나리오: ~40% 커버리지
- Wall 시나리오: ~30% 커버리지
- Circle 시나리오: ~25% 커버리지
- Arch 시나리오: ~20% 커버리지

**[상세 보고서 (REPORT.pdf)](./REPORT.pdf)** - 전체 설계, 실험 결과, 분석 내용

---

## 빠른 시작

### 설치

```bash
pip install -r requirements.txt
```

### 학습 실행

**기본 학습 (Circle 시나리오, 500 에피소드):**
```bash
python run_scenario_training.py --scenario circle --episodes 500
```

**다른 시나리오 학습:**
```bash
# 벽면 시나리오
python run_scenario_training.py --scenario wall --episodes 500

# 바닥 관개 시나리오
python run_scenario_training.py --scenario ground --episodes 300

# 분수 아치 시나리오
python run_scenario_training.py --scenario arch --episodes 500
```

### 학습 결과 확인

학습이 완료되면 `figures/run_YYYYMMDD_HHMMSS_[scenario]_seed[N]_ep[E]/` 디렉토리에 다음 파일들이 생성됩니다:

- `learning_curves.png` - 학습 곡선 (보상, 커버리지)
- `particle_animation.gif` - 학습된 에이전트의 행동 애니메이션
- `dqn_model.pth` - 학습된 DQN 모델 (PyTorch 체크포인트)

---

## 학습된 모델 사용

### 모델 다운로드

학습된 모델은 다음 위치에서 다운로드할 수 있습니다:

**사전 학습된 모델 (시나리오 별):**
[model download](https://github.com/art1factia/RL/releases/tag/dqn_model)

### 모델 로드 및 테스트

```bash
# 학습된 모델로 평가 및 애니메이션 생성
python load_and_test_model.py figures/run_*/dqn_model.pth \
  --scenario ground \
  --num-eval 10 \
  --save-animation \
  --animation-steps 500
```

**옵션 설명:**
- `--scenario`: 테스트할 시나리오 (circle, wall, ground, arch)
- `--num-eval`: 평가 에피소드 수 (기본값: 10)
- `--save-animation`: 애니메이션 GIF 생성
- `--animation-steps`: 애니메이션 길이 (기본값: 300)

---

## 프로젝트 구조

```
rl_particle/
├── env/
│   ├── __init__.py
│   └── particle_env.py          # 파티클 환경 (물리 시뮬레이션)
├── agent/
│   ├── __init__.py
│   └── dqn.py                   # DQN 에이전트
├── experiments/
│   ├── __init__.py
│   └── train.py                 # 학습 스크립트
├── figures/                     # 학습 결과 저장 디렉토리
│   ├── report_*.png            # 보고서용 시각화
│   └── run_*/                  # 개별 학습 실행 결과
│       ├── learning_curves.png
│       ├── particle_animation.gif
│       └── dqn_model.pth       # 학습된 모델
├── run_scenario_training.py     # 시나리오별 학습 실행
├── load_and_test_model.py      # 모델 로드 및 테스트
├── requirements.txt
├── README.md                    # 프로젝트 개요 (본 문서)
└── REPORT.pdf                   # 상세 보고서
```

---

## 주요 기술 스택

- **강화학습**: Deep Q-Network (DQN)
- **프레임워크**: PyTorch 2.x
- **환경**: NumPy 기반 물리 시뮬레이션
- **시각화**: Matplotlib, Pillow (GIF)

---

## 시나리오 설명

| 시나리오 | 설명 | 물리적 응용 |
|---------|------|-----------|
| **circle** | 원형 타겟 | 단순한 영역 커버 |
| **wall** | 벽면 페인팅/소화 | 수직 벽면 제어 (소방 호스, 페인트 스프레이) |
| **ground** | 바닥 관개 | 땅 위 영역 커버 (스프링클러, 정원 관개) |
| **arch** | 분수 아치 | 포물선 궤적 제어 (장식용 분수) |

---

## 알고리즘 개요

### State (67차원)
- 시간 정규화 (1차원)
- 각도/속도 정규화 (2차원)
- 8×8 파티클 분포 히스토그램 (64차원)

### Action (9개 이산 행동)
- 각도 조정: ↑, ↓
- 속도 조정: ↑, ↓
- 조합 행동 4가지
- 유지 행동

### Reward
- 커버리지 증가분 × 500.0
- 현재 커버리지 × 100.0
- 근접 보상 × 1.0

### Hyperparameters
- Learning rate: 3e-4
- Gamma: 0.99
- Replay buffer: 100,000
- Batch size: 128
- Epsilon decay: 20,000 steps

---

## 실험 결과 요약

### 시나리오별 성능

| 시나리오 | 평균 최종 커버리지 | 수렴 에피소드 |
|---------|------------------|-------------|
| Ground | 38.2% ± 3.5% | ~150 |
| Wall | 29.7% ± 4.2% | ~200 |
| Circle | 24.5% ± 3.8% | ~250 |
| Arch | 19.3% ± 5.1% | ~300 |

상세한 실험 결과 및 분석은 [REPORT.pdf](./REPORT.pdf)**를 참조하세요.

---

## 참고 자료

- **상세 보고서**: [REPORT.pdf](./REPORT.pdf)
- **DQN 논문**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **PyTorch 문서**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

---
