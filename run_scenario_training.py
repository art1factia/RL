#!/usr/bin/env python3
"""
시나리오별 학습 스크립트
물리적으로 의미 있는 다양한 타겟 영역에서 학습
"""

import argparse
from experiments.train import train_dqn
from env.particle_env import (
    make_circle_mask,
    make_wall_target_mask,
    make_ground_target_mask,
    make_parabolic_arch_mask,
)

SCENARIOS = {
    "circle": {
        "name": "원형 타겟 (기본)",
        "description": "단순한 원형 영역",
        "func": lambda: make_circle_mask(size=64, center=(0.5, 0.5), radius=0.25),
    },
    "wall": {
        "name": "벽면 페인팅/소화",
        "description": "수직 벽면의 특정 영역 (소방 호스, 페인트 스프레이)",
        "func": lambda: make_wall_target_mask(size=64, wall_x=0.7, y_range=(0.3, 0.8), width=0.15),
    },
    "ground": {
        "name": "바닥 관개",
        "description": "땅 위의 여러 영역 (농업 스프링클러, 정원 관개)",
        "func": lambda: make_ground_target_mask(size=64, regions=[(0.3, 0.08), (0.5, 0.06), (0.7, 0.09)]),
    },
    "arch": {
        "name": "분수 아치",
        "description": "포물선 궤적을 따르는 영역 (장식용 분수)",
        "func": lambda: make_parabolic_arch_mask(size=64, height=0.6, width=0.5, thickness=0.08),
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="시나리오별 파티클 제어 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 가능한 시나리오:
  circle  - 원형 타겟 (기본)
  wall    - 벽면 페인팅/소화 (수직 벽면)
  ground  - 바닥 관개 (여러 영역)
  arch    - 분수 아치 (포물선 궤적)

예시:
  python run_scenario_training.py --scenario wall --episodes 500
  python run_scenario_training.py --scenario ground --episodes 300 --no-animation
  python run_scenario_training.py --scenario arch --episodes 500 --animation-steps 500
        """,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="wall",
        choices=list(SCENARIOS.keys()),
        help="시나리오 선택 (기본: wall)",
    )
    parser.add_argument("--episodes", type=int, default=500, help="학습 에피소드 수")
    parser.add_argument("--seed", type=int, default=0, help="랜덤 시드")
    parser.add_argument("--eval-interval", type=int, default=25, help="평가 주기")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="디바이스")
    parser.add_argument("--no-animation", action="store_true", help="애니메이션 생성 건너뛰기")
    parser.add_argument("--animation-steps", type=int, default=300, help="애니메이션 길이 (스텝 수, 기본: 300)")
    parser.add_argument("--list-scenarios", action="store_true", help="사용 가능한 시나리오 목록 표시")

    args = parser.parse_args()

    # 시나리오 목록 표시
    if args.list_scenarios:
        print("\n사용 가능한 시나리오:\n")
        for key, info in SCENARIOS.items():
            print(f"  {key:8s} - {info['name']}")
            print(f"             {info['description']}\n")
        return

    scenario = SCENARIOS[args.scenario]

    print("=" * 70)
    print(f"시나리오: {scenario['name']}")
    print(f"설명: {scenario['description']}")
    print("=" * 70)
    print(f"설정:")
    print(f"  - 에피소드 수: {args.episodes}")
    print(f"  - 시드: {args.seed}")
    print(f"  - 평가 주기: {args.eval_interval}")
    print(f"  - 디바이스: {args.device}")
    print(f"  - 애니메이션: {'생성 안 함' if args.no_animation else f'생성 ({args.animation_steps} 스텝)'}")
    print("=" * 70)
    print(flush=True)

    # 타겟 마스크 생성
    target_mask = scenario["func"]()

    # 학습 실행
    print("\n학습 시작...\n", flush=True)
    history = train_dqn(
        seed=args.seed,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        device=args.device,
        save_results=True,
        save_animation=not args.no_animation,
        target_mask=target_mask,
        scenario_name=args.scenario,
        animation_steps=args.animation_steps,
    )

    print("\n" + "=" * 70)
    print(f"학습 완료!")
    print(f"최종 평균 커버리지: {history['eval_means'][-1] if len(history['eval_means']) > 0 else history['episode_coverages'][-1]:.3f}")
    print(f"결과 저장 위치: {history.get('result_dir', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
