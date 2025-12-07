#!/usr/bin/env python3
"""
개선된 코드 테스트 스크립트
짧은 에피소드로 모든 기능이 정상 작동하는지 확인
"""

import sys
import os

# 부모 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.train import train_dqn

if __name__ == "__main__":
    print("=" * 60)
    print("개선된 코드 테스트 시작")
    print("=" * 60)

    # 짧은 에피소드로 테스트 (20 에피소드)
    history = train_dqn(
        seed=0,
        num_episodes=20,
        eval_interval=10,
        device="cpu",
        save_results=True,
        save_animation=True,
    )

    print("\n테스트 완료!")
    print(f"최종 평균 커버리지: {history['eval_means'][-1]:.3f}")
    print("저장된 파일:")
    print("  - 학습 곡선: figures/train_seed0_ep20_*.png")
    print("  - 파티클 애니메이션: figures/animation_seed0_ep20_*.gif")
