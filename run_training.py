#!/usr/bin/env python3
"""
정식 학습 스크립트
개선된 DQN으로 파티클 제어 학습
"""

import argparse
from experiments.train import train_dqn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN 파티클 제어 학습")
    parser.add_argument("--episodes", type=int, default=500, help="학습 에피소드 수 (기본: 500)")
    parser.add_argument("--seed", type=int, default=0, help="랜덤 시드 (기본: 0)")
    parser.add_argument("--eval-interval", type=int, default=25, help="평가 주기 (기본: 25)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="디바이스 (기본: cpu)")
    parser.add_argument("--no-animation", action="store_true", help="애니메이션 생성 건너뛰기 (학습 속도 향상)")
    parser.add_argument("--animation-steps", type=int, default=300, help="애니메이션 길이 (스텝 수, 기본: 300)")

    args = parser.parse_args()

    print("=" * 70)
    print("DQN 파티클 제어 학습 시작")
    print("=" * 70)
    print(f"설정:")
    print(f"  - 에피소드 수: {args.episodes}")
    print(f"  - 시드: {args.seed}")
    print(f"  - 평가 주기: {args.eval_interval}")
    print(f"  - 디바이스: {args.device}")
    print(f"  - 애니메이션: {'생성 안 함' if args.no_animation else f'생성 ({args.animation_steps} 스텝)'}")
    print("=" * 70)
    print(flush=True)

    history = train_dqn(
        seed=args.seed,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        device=args.device,
        save_results=True,
        save_animation=not args.no_animation,
        animation_steps=args.animation_steps,
    )

    print("\n" + "=" * 70, flush=True)
    print("학습 완료!", flush=True)
    print("=" * 70, flush=True)

    if len(history['eval_means']) > 0:
        print(f"최종 평가 커버리지: {history['eval_means'][-1]:.3f}", flush=True)
        print(f"최대 커버리지: {max(history['episode_coverages']):.3f}", flush=True)

    print(f"\n결과 저장 위치: {history.get('result_dir', 'N/A')}", flush=True)
    print("\n저장된 파일:", flush=True)
    print(f"  - 학습 곡선: {history.get('result_dir', 'N/A')}/learning_curves.png", flush=True)
    if not args.no_animation:
        print(f"  - 파티클 애니메이션: {history.get('result_dir', 'N/A')}/particle_animation.gif", flush=True)
    print("=" * 70, flush=True)
