#!/usr/bin/env python3
"""
학습된 DQN 모델을 로드하고 테스트하는 스크립트
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from agent import DQNAgent
from env import ParticleEnv, ParticleEnvConfig, make_circle_mask, make_wall_target_mask, make_ground_target_mask, make_parabolic_arch_mask
from experiments.train import save_particle_animation, evaluate_policy


SCENARIOS = {
    "circle": lambda: make_circle_mask(size=64, center=(0.5, 0.5), radius=0.25),
    "wall": lambda: make_wall_target_mask(size=64, wall_x=0.7, y_range=(0.3, 0.8), width=0.15),
    "ground": lambda: make_ground_target_mask(size=64, regions=[(0.3, 0.08), (0.5, 0.06), (0.7, 0.09)]),
    "arch": lambda: make_parabolic_arch_mask(size=64, height=0.6, width=0.5, thickness=0.08),
}


def load_model(model_path: str, device: str = "cpu"):
    """
    저장된 DQN 모델을 로드

    Args:
        model_path: .pth 모델 파일 경로
        device: 'cpu' 또는 'cuda'

    Returns:
        agent: 로드된 DQNAgent
        checkpoint: 저장된 체크포인트 정보
    """
    checkpoint = torch.load(model_path, map_location=device)

    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']

    # 에이전트 생성
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **checkpoint.get('hyperparameters', {})
    )

    # 모델 가중치 로드
    agent.q_net.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    agent.q_net.eval()
    agent.target_net.eval()

    print(f"✓ Model loaded from {model_path}")
    print(f"  - State dim: {state_dim}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Trained for {checkpoint.get('num_episodes', 'N/A')} episodes")
    print(f"  - Final coverage: {checkpoint.get('final_coverage', 'N/A'):.3f}")
    print(f"  - Mean eval coverage: {checkpoint.get('mean_eval_coverage', 'N/A'):.3f}")

    return agent, checkpoint


def main():
    parser = argparse.ArgumentParser(description="학습된 DQN 모델 테스트")
    parser.add_argument("model_path", type=str, help="모델 파일 경로 (.pth)")
    parser.add_argument("--scenario", type=str, default="circle",
                       choices=list(SCENARIOS.keys()),
                       help="테스트할 시나리오")
    parser.add_argument("--num-eval", type=int, default=10,
                       help="평가 에피소드 수")
    parser.add_argument("--save-animation", action="store_true",
                       help="애니메이션 GIF 생성")
    parser.add_argument("--animation-steps", type=int, default=300,
                       help="애니메이션 길이 (스텝)")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="디바이스")

    args = parser.parse_args()

    # 모델 로드
    agent, checkpoint = load_model(args.model_path, args.device)

    # 환경 생성
    target_mask = SCENARIOS[args.scenario]()
    cfg = ParticleEnvConfig(device=args.device)
    env = ParticleEnv(target_mask=target_mask, config=cfg)

    # 평가
    print(f"\nEvaluating on {args.scenario} scenario ({args.num_eval} episodes)...")
    metrics = evaluate_policy(env, agent, num_episodes=args.num_eval)
    print(f"  Mean coverage: {metrics['mean_coverage']:.3f} ± {metrics['std_coverage']:.3f}")

    # 애니메이션 생성
    if args.save_animation:
        model_dir = Path(args.model_path).parent
        anim_path = model_dir / f"test_animation_{args.scenario}.gif"
        print(f"\nCreating animation ({args.animation_steps} steps)...")
        save_particle_animation(env, agent, str(anim_path),
                              max_steps=args.animation_steps, sample_every=5)
        print(f"✓ Animation saved to {anim_path}")


if __name__ == "__main__":
    main()
