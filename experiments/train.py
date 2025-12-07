# experiments/train.py

import os
import random
from typing import List, Dict
from datetime import datetime
import io

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from PIL import Image

from env import (
    ParticleEnv,
    ParticleEnvConfig,
    make_circle_mask,
    make_wall_target_mask,
    make_ground_target_mask,
    make_parabolic_arch_mask,
)
from agent import DQNAgent

def save_particle_animation(
    env: ParticleEnv,
    agent: DQNAgent,
    save_path: str,
    max_steps: int = None,
    fps: int = 10,
    sample_every: int = 5,  # 개선 20: 매 N 스텝마다만 프레임 저장 (속도 향상)
):
    """
    개선 13: 파티클 움직임을 GIF로 저장하여 학습된 정책의 행동을 시각화
    에이전트가 어떻게 파티클을 제어하는지 직관적으로 확인 가능

    개선 20: sample_every를 사용하여 일부 프레임만 저장 (GIF 생성 속도 향상)
    """
    print(f"Generating animation... (sampling every {sample_every} steps)", flush=True)

    if max_steps is None:
        max_steps = env.cfg.max_steps

    # 에피소드 실행하면서 프레임 수집
    frames = []
    state = env.reset()
    done = False
    step_count = 0

    # Figure를 미리 생성하여 재사용 (속도 향상)
    fig, ax = plt.subplots(figsize=(6, 6))

    while not done and env.step_count < max_steps:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        state = next_state

        # sample_every 스텝마다만 프레임 저장
        if step_count % sample_every == 0:
            ax.clear()
            ax.imshow(
                env.target_mask,
                origin="lower",
                extent=[0, 1, 0, 1],
                alpha=0.3,
                cmap="Greens",
            )
            if env.p_pos.shape[0] > 0:
                ax.scatter(env.p_pos[:, 0], env.p_pos[:, 1], s=5, c="C0", alpha=0.6)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(
                f"Step {env.step_count}/{max_steps} | Coverage: {info['coverage']:.3f}\n"
                f"Angle: {env.theta:.1f}° | Speed: {env.speed:.2f}",
                fontsize=10,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            # 현재 각도/속도 표시
            ax.text(
                0.02,
                0.98,
                f"Action: {action}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            # 개선 25: matplotlib 버전 호환성 문제 해결
            # tostring_rgb() 대신 PIL을 사용한 안정적인 방법
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            frame = np.array(Image.open(buf).convert('RGB'))
            buf.close()
            frames.append(frame)

        step_count += 1

    plt.close(fig)

    if len(frames) == 0:
        print(f"No frames captured!", flush=True)
        return

    # GIF로 저장
    print(f"Saving {len(frames)} frames to GIF...", flush=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Animation saved to {save_path}", flush=True)


def evaluate_policy(env: ParticleEnv, agent: DQNAgent, num_episodes: int = 10) -> Dict[str, float]:
    coverages = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        last_cov = 0.0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            last_cov = info["coverage"]
        coverages.append(last_cov)
    coverages = np.array(coverages)
    return {
        "mean_coverage": float(coverages.mean()),
        "std_coverage": float(coverages.std()),
    }

def train_dqn(
    seed: int = 0,
    num_episodes: int = 500,
    eval_interval: int = 20,
    device: str = "cpu",
    save_results: bool = True,  # 개선 15: 자동 저장 옵션
    save_animation: bool = True,  # 개선 16: 애니메이션 저장 옵션
    target_mask=None,  # 개선 26: 커스텀 타겟 마스크 지원
    scenario_name: str = "default",  # 개선 26: 시나리오 이름 (결과 폴더명에 포함)
    animation_steps: int = 300,  # 개선 27: 애니메이션 길이 제어 (기본 100 -> 300)
):
    # 시드 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # 타겟 마스크 생성
    # 개선 23: 타겟 위치를 더 도달하기 쉬운 곳으로 이동 (0.7 -> 0.5)
    # 중력이 -3.0이므로, 너무 높으면 도달하기 어려움
    # 개선 26: target_mask가 제공되지 않으면 기본 원형 타겟 사용
    if target_mask is None:
        target_mask = make_circle_mask(size=64, center=(0.5, 0.5), radius=0.25)

    cfg = ParticleEnvConfig(device=device)
    env = ParticleEnv(target_mask=target_mask, config=cfg)

    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = len(env.action_deltas)

    # 하이퍼파라미터 설정 (개선된 값으로 업데이트)
    agent_params = {
        "lr": 3e-4,
        "gamma": 0.99,
        "buffer_capacity": 100000,
        "batch_size": 128,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_steps": 20000,
        "target_update_interval": 500,
        "warmup_steps": 1000,
    }

    # 전체 하이퍼파라미터 (저장 및 시각화용)
    hyperparameters = {
        **agent_params,
        "coverage_reward_scale": cfg.coverage_reward_scale,
        "current_coverage_scale": cfg.current_coverage_scale,
        "proximity_reward_scale": cfg.proximity_reward_scale,
    }

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **agent_params,
    )

    episode_rewards = []
    episode_coverages = []
    eval_steps = []
    eval_means = []

    print(f"\nStarting training for {num_episodes} episodes...", flush=True)
    print(f"State dim: {state_dim}, Action dim: {action_dim}", flush=True)
    print(f"Warmup steps: {agent_params['warmup_steps']}", flush=True)

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        last_cov = 0.0

        while not done:
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, info = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)
            loss = agent.update()

            state = next_state
            ep_reward += reward
            last_cov = info["coverage"]

        episode_rewards.append(ep_reward)
        episode_coverages.append(last_cov)

        if (ep + 1) % 10 == 0:
            print(
                f"[Seed {seed}] Episode {ep+1}/{num_episodes} "
                f"Reward={ep_reward:.3f}, Final coverage={last_cov:.3f}, eps={agent.epsilon():.3f}",
                flush=True  # 실시간 출력을 위한 버퍼링 해제
            )

        if (ep + 1) % eval_interval == 0:
            metrics = evaluate_policy(env, agent, num_episodes=10)
            eval_steps.append(ep + 1)
            eval_means.append(metrics["mean_coverage"])
            print(
                f"  >> Eval after {ep+1} episodes: "
                f"mean coverage={metrics['mean_coverage']:.3f} ± {metrics['std_coverage']:.3f}",
                flush=True  # 실시간 출력을 위한 버퍼링 해제
            )

    history = {
        "episode_rewards": np.array(episode_rewards),
        "episode_coverages": np.array(episode_coverages),
        "eval_steps": np.array(eval_steps),
        "eval_means": np.array(eval_means),
        "agent": agent,
        "env_cfg": cfg,
        "target_mask": target_mask,
        "hyperparameters": hyperparameters,  # 개선 17: 하이퍼파라미터 저장
    }

    # 개선 18: 학습 완료 후 자동으로 결과 저장
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 별도 폴더에 저장 (개선 26: 시나리오 이름 포함)
        result_dir = f"figures/run_{timestamp}_{scenario_name}_seed{seed}_ep{num_episodes}"
        os.makedirs(result_dir, exist_ok=True)
        print(f"\nSaving results to: {result_dir}/", flush=True)

        # 학습 곡선 저장
        print("Saving learning curves...", flush=True)
        curve_path = f"{result_dir}/learning_curves.png"
        plot_learning_curves(history, label_prefix="DQN ", save_path=curve_path)

        # 파티클 애니메이션 저장
        if save_animation:
            print(f"Creating particle animation ({animation_steps} steps)...", flush=True)
            anim_path = f"{result_dir}/particle_animation.gif"
            save_particle_animation(env, agent, anim_path, max_steps=animation_steps, sample_every=5)

        # 학습된 모델 저장
        print("Saving trained model...", flush=True)
        model_path = f"{result_dir}/dqn_model.pth"
        torch.save({
            'q_network_state_dict': agent.q_net.state_dict(),
            'target_network_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'hyperparameters': hyperparameters,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'seed': seed,
            'num_episodes': num_episodes,
            'final_coverage': episode_coverages[-1] if episode_coverages else 0.0,
            'mean_eval_coverage': eval_means[-1] if eval_means else 0.0,
        }, model_path)
        print(f"Model saved to {model_path}", flush=True)

        print(f"\n결과 저장 완료: {result_dir}/", flush=True)
        history["result_dir"] = result_dir
        history["model_path"] = model_path

    return history

def plot_learning_curves(history, label_prefix: str = "", save_path: str = None):
    """
    개선 14: 학습 곡선에 하이퍼파라미터 정보를 포함하여 실험 추적 용이성 향상
    저장된 이미지만 보고도 어떤 설정으로 학습했는지 파악 가능
    """
    rewards = history["episode_rewards"]
    coverages = history["episode_coverages"]
    eval_steps = history["eval_steps"]
    eval_means = history["eval_means"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward
    axes[0].plot(rewards, label=f"{label_prefix}reward", alpha=0.7)
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    # Coverage (per episode + eval)
    axes[1].plot(coverages, alpha=0.5, label=f"{label_prefix}final coverage")
    if len(eval_steps) > 0:
        axes[1].plot(eval_steps, eval_means, "o-", label=f"{label_prefix}eval mean coverage")
    axes[1].set_title("Final coverage per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Coverage")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # 하이퍼파라미터 정보 추가
    if "hyperparameters" in history:
        hp = history["hyperparameters"]
        info_text = (
            f"Hyperparameters:\n"
            f"  Learning rate: {hp.get('lr', 'N/A')}\n"
            f"  Gamma: {hp.get('gamma', 'N/A')}\n"
            f"  Batch size: {hp.get('batch_size', 'N/A')}\n"
            f"  Buffer capacity: {hp.get('buffer_capacity', 'N/A')}\n"
            f"  Eps decay: {hp.get('eps_decay_steps', 'N/A')} steps\n"
            f"  Warmup: {hp.get('warmup_steps', 'N/A')} steps\n"
            f"  Target update: {hp.get('target_update_interval', 'N/A')} steps\n"
            f"\nReward config:\n"
            f"  Coverage scale: {hp.get('coverage_reward_scale', 'N/A')}\n"
            f"  Current cov scale: {hp.get('current_coverage_scale', 'N/A')}\n"
            f"  Proximity scale: {hp.get('proximity_reward_scale', 'N/A')}\n"
            f"\nFinal mean coverage: {eval_means[-1] if len(eval_means) > 0 else coverages[-1]:.3f}"
        )
        fig.text(
            0.98,
            0.5,
            info_text,
            fontsize=8,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            family="monospace",
        )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to {save_path}", flush=True)
        plt.close(fig)  # 개선 21: 저장 후 닫기 (애니메이션 생성 계속 진행)
    else:
        plt.show()  # save_path가 없을 때만 화면에 표시

def run_multiple_seeds(
    seeds: List[int],
    num_episodes: int = 300,
    device: str = "cpu",
):
    histories = []
    final_coverages = []

    for s in seeds:
        print(f"=== Training with seed {s} ===")
        h = train_dqn(seed=s, num_episodes=num_episodes, eval_interval=20, device=device)
        histories.append(h)
        final_coverages.append(h["episode_coverages"][-1])

    final_coverages = np.array(final_coverages)
    mean_cov = final_coverages.mean()
    std_cov = final_coverages.std()
    print(f">>> Final coverage across seeds: {mean_cov:.3f} ± {std_cov:.3f}")

    # 막대 그래프
    plt.figure(figsize=(5, 4))
    x = np.arange(len(seeds))
    plt.bar(x, final_coverages)
    plt.xticks(x, [f"seed {s}" for s in seeds])
    plt.ylabel("Final coverage")
    plt.title("Final coverage per seed")
    plt.grid(axis="y")
    plt.show()

    return histories, final_coverages

def visualize_episode_trajectories(env: ParticleEnv, agent: DQNAgent, max_steps: int = None):
    """
    하나의 에피소드에서 파티클 궤적을 기록한 뒤,
    전체 궤적을 한 그림에 시각화.
    """
    if max_steps is None:
        max_steps = env.cfg.max_steps

    # 궤적 기록용 리스트
    all_positions = []

    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        all_positions.append(env.p_pos.copy())
        if env.step_count >= max_steps:
            break

    # 그림 그리기
    fig, ax = plt.subplots(figsize=(4, 4))

    # 타겟 마스크
    ax.imshow(env.target_mask, origin="lower", extent=[0, 1, 0, 1], alpha=0.3, cmap="Greens")

    # 각 스텝의 파티클 위치를 연하게
    for t, pos in enumerate(all_positions):
        if pos.shape[0] == 0:
            continue
        alpha = 0.1 + 0.9 * (t / len(all_positions))  # 뒤로 갈수록 진하게
        ax.scatter(pos[:, 0], pos[:, 1], s=5, c="C0", alpha=alpha)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Particle trajectories over one episode")
    plt.show()

if __name__ == "__main__":
    device = "cpu"

    # 개선 19: 학습 파라미터 설정
    # num_episodes를 늘려서 충분한 학습 시간 확보 (500 -> 1000)
    num_episodes = 1000
    eval_interval = 50

    print("=" * 60)
    print("개선된 DQN 학습 시작")
    print("=" * 60)
    print(f"총 에피소드: {num_episodes}")
    print(f"평가 주기: {eval_interval}")
    print(f"디바이스: {device}")
    print("=" * 60)

    # 1) 단일 seed 학습 + 자동 저장
    # save_results=True로 학습 곡선과 애니메이션 자동 저장
    history = train_dqn(
        seed=0,
        num_episodes=num_episodes,
        eval_interval=eval_interval,
        device=device,
        save_results=True,
        save_animation=True,
    )

    print("\n학습 완료!")
    print(f"최종 평균 커버리지: {history['eval_means'][-1]:.3f}")
