#!/usr/bin/env python3
"""
빠른 테스트: 50 에피소드로 개선 사항 검증
"""

from experiments.train import train_dqn

if __name__ == "__main__":
    print("=" * 60)
    print("빠른 테스트 (50 에피소드)")
    print("=" * 60)

    history = train_dqn(
        seed=0,
        num_episodes=50,
        eval_interval=10,
        device="cpu",
        save_results=True,
        save_animation=True,
    )

    print("\n테스트 완료!", flush=True)
    if len(history['eval_means']) > 0:
        print(f"최종 평균 커버리지: {history['eval_means'][-1]:.3f}", flush=True)
    print(f"결과 저장 위치: {history.get('result_dir', 'N/A')}", flush=True)
