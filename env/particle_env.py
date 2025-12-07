# env/particle_env.py

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def make_circle_mask(size=64, center=(0.5, 0.6), radius=0.45) -> np.ndarray:
    """
    size x size 이진 마스크에서 원형 영역을 1로 표시.
    center, radius는 [0,1] 좌표계 기준.
    """
    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cx, cy = center
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist2 <= radius**2
    return mask.astype(np.bool_)


def make_wall_target_mask(size=64, wall_x=0.7, y_range=(0.3, 0.8), width=0.15) -> np.ndarray:
    """
    벽면 페인팅/소화 시나리오: 수직 벽면의 특정 영역

    물리적 의미:
    - 발사 위치에서 벽면까지 포물선 궤적으로 물/페인트를 발사
    - 중력 때문에 높은 곳을 맞추려면 강한 발사 필요
    - 현실적인 응용: 소방 호스, 페인트 스프레이, 세차

    Args:
        size: 그리드 크기
        wall_x: 벽의 x 위치 (0~1)
        y_range: 타겟의 y 범위 (하단, 상단)
        width: 타겟의 두께
    """
    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # 벽면 영역: wall_x ± width/2, y_range 내
    x_mask = np.abs(xx - wall_x) <= width / 2
    y_mask = (yy >= y_range[0]) & (yy <= y_range[1])

    return (x_mask & y_mask).astype(np.bool_)


def make_ground_target_mask(size=64, regions=None, ground_y=0.25) -> np.ndarray:
    """
    바닥 관개 시나리오: 땅 위의 여러 영역 (식물, 화재 등)

    물리적 의미:
    - 중력으로 떨어지는 물이 바닥의 특정 영역을 커버
    - 포물선 궤적의 낙하 지점 제어
    - 현실적인 응용: 농업 스프링클러, 정원 관개, 소화

    Args:
        size: 그리드 크기
        regions: [(x_center, radius), ...] 형태의 원형 영역 리스트
        ground_y: 바닥 높이
    """
    if regions is None:
        # 기본: 3개의 영역
        regions = [(0.3, 0.08), (0.5, 0.06), (0.7, 0.09)]

    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    mask = np.zeros((size, size), dtype=bool)

    # 각 영역을 원형으로 추가
    for x_center, radius in regions:
        dist2 = (xx - x_center) ** 2 + (yy - ground_y) ** 2
        mask |= (dist2 <= radius ** 2)

    return mask.astype(np.bool_)


def make_parabolic_arch_mask(size=64, height=0.6, width=0.5, thickness=0.08) -> np.ndarray:
    """
    분수 아치 시나리오: 포물선 궤적을 따르는 영역

    물리적 의미:
    - 물리적으로 자연스러운 포물선 궤적
    - 중력에 의한 자연스러운 경로
    - 현실적인 응용: 장식용 분수, 물줄기 쇼

    Args:
        size: 그리드 크기
        height: 아치의 최대 높이
        width: 아치의 너비
        thickness: 궤적의 두께
    """
    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # 포물선: y = -4*height/width^2 * (x - 0.5)^2 + height
    # x를 중심(0.5)으로 정규화
    x_normalized = (xx - 0.5) / (width / 2)
    parabola_y = height * (1 - x_normalized ** 2)

    # 포물선 근처의 영역만 마스크
    dist_to_parabola = np.abs(yy - parabola_y)
    mask = dist_to_parabola <= thickness

    # 유효 범위 (x: 0.5 ± width/2, y > 0)
    valid_x = np.abs(xx - 0.5) <= width / 2
    valid_y = parabola_y >= 0

    return (mask & valid_x & valid_y).astype(np.bool_)


@dataclass
class ParticleEnvConfig:
    max_steps: int = 200
    dt: float = 0.02
    gravity: float = -3.0  # y축 위가 + 방향, 음수면 아래로 끌어당김
    friction: float = 0.99
    emitter_pos: Tuple[float, float] = (0.5, 0.05)
    angle_deg_range: Tuple[float, float] = (30.0, 150.0)
    speed_range: Tuple[float, float] = (1.5, 4.0)
    particles_per_step: int = 50
    state_grid_size: int = 8
    coverage_grid_size: int = 64
    angle_step_deg: float = 5.0
    speed_step_frac: float = 0.1  # 전체 범위의 몇 퍼센트씩 움직일지
    penalty_lambda: float = 0.0  # 필요하면 패널티 넣기
    device: str = "cpu"

    # 개선 1: 보상 함수 설정
    # - coverage_reward_scale: 커버리지 증가분에 대한 보상 스케일
    # - current_coverage_scale: 현재 커버리지 자체에 대한 보상 (희소성 완화)
    # - proximity_reward_scale: 타겟 근처에 파티클이 있을 때 추가 보상
    # 개선 22: 보상 재조정 - proximity 감소, coverage 증가
    coverage_reward_scale: float = 500.0  # 100.0 -> 500.0 (커버리지 증가분 강조)
    current_coverage_scale: float = 100.0  # 10.0 -> 100.0 (현재 커버리지 강조)
    proximity_reward_scale: float = 1.0  # 5.0 -> 1.0 (근접 보상 약화)


class ParticleEnv:
    """
    2D 파티클 분수 환경.
    - action: 발사 각도/속도를 조정하는 이산 행동(0~8)
    - state: [time_norm, angle_norm, speed_norm, 8x8 파티클 분포 히스토그램]
    - reward: coverage_t - coverage_{t-1} - penalty
    """

    def __init__(self, target_mask: np.ndarray, config: ParticleEnvConfig):
        assert target_mask.shape[0] == target_mask.shape[1], "Mask must be square"
        self.cfg = config
        self.target_mask = target_mask.astype(bool)
        self.mask_size = target_mask.shape[0]

        # 내부 상태
        self.theta = None  # 현재 각도(도)
        self.speed = None  # 현재 속도
        self.step_count = 0

        # 파티클 (N x 2)
        self.p_pos = np.zeros((0, 2), dtype=np.float32)
        self.p_vel = np.zeros((0, 2), dtype=np.float32)

        self.prev_coverage = 0.0

        # 액션 -> (d_angle_sign, d_speed_sign) 매핑
        # -1, 0, +1
        self.action_deltas = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (0, 0),
        ]

    @property
    def angle_min(self):
        return self.cfg.angle_deg_range[0]

    @property
    def angle_max(self):
        return self.cfg.angle_deg_range[1]

    @property
    def speed_min(self):
        return self.cfg.speed_range[0]

    @property
    def speed_max(self):
        return self.cfg.speed_range[1]

    def reset(self):
        self.step_count = 0
        # 가운데 정도에서 시작
        self.theta = 0.5 * (self.angle_min + self.angle_max)
        self.speed = 0.5 * (self.speed_min + self.speed_max)
        self.p_pos = np.zeros((0, 2), dtype=np.float32)
        self.p_vel = np.zeros((0, 2), dtype=np.float32)
        self.prev_coverage = 0.0
        state = self._get_state()
        return state

    def _spawn_particles(self):
        n = self.cfg.particles_per_step
        theta_rad = math.radians(self.theta)
        # 방향 벡터
        dir_vec = np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float32)
        pos_new = np.repeat(
            np.array(self.cfg.emitter_pos, dtype=np.float32)[None, :], n, axis=0
        )
        vel_new = np.repeat((dir_vec * self.speed)[None, :], n, axis=0)

        if self.p_pos.shape[0] == 0:
            self.p_pos = pos_new
            self.p_vel = vel_new
        else:
            self.p_pos = np.vstack([self.p_pos, pos_new])
            self.p_vel = np.vstack([self.p_vel, vel_new])

    def _update_particles(self):
        if self.p_pos.shape[0] == 0:
            return 0.0  # lost fraction
        # 중력 + 마찰
        self.p_vel[:, 1] += self.cfg.gravity * self.cfg.dt
        self.p_pos += self.p_vel * self.cfg.dt
        self.p_vel *= self.cfg.friction

        # 화면 밖으로 나간 파티클 제거
        x = self.p_pos[:, 0]
        y = self.p_pos[:, 1]
        mask = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
        lost_frac = 1.0 - mask.mean()
        self.p_pos = self.p_pos[mask]
        self.p_vel = self.p_vel[mask]
        return lost_frac

    def _compute_grids(self):
        """8x8 coarse grid (state용) + 64x64 occupancy grid (coverage용)"""
        sg = self.cfg.state_grid_size
        cg = self.cfg.coverage_grid_size

        coarse = np.zeros((sg, sg), dtype=np.float32)
        occ = np.zeros((cg, cg), dtype=bool)

        if self.p_pos.shape[0] == 0:
            return coarse, occ

        x = np.clip(self.p_pos[:, 0], 0.0, 0.9999)
        y = np.clip(self.p_pos[:, 1], 0.0, 0.9999)

        # coarse grid index
        ix_s = (x * sg).astype(int)
        iy_s = (y * sg).astype(int)
        np.add.at(coarse, (iy_s, ix_s), 1.0)
        if coarse.sum() > 0:
            coarse /= coarse.sum()

        # coverage grid index
        ix_c = (x * cg).astype(int)
        iy_c = (y * cg).astype(int)
        occ[iy_c, ix_c] = True

        return coarse, occ

    def _compute_coverage(self, occ_grid: np.ndarray) -> float:
        target_pixels = self.target_mask.sum()
        if target_pixels == 0:
            return 0.0
        overlap = occ_grid & self.target_mask
        return overlap.sum() / target_pixels

    def _compute_proximity_reward(self) -> float:
        """
        개선 2: 타겟 근처에 파티클이 있으면 거리 기반 보상을 제공
        이를 통해 보상의 희소성을 완화하고, 초기 탐색을 개선

        개선 24: 근접 보상 계산 방식 개선
        - 평균 대신 최대값 사용 (가장 가까운 파티클만 보상)
        - 임계값 축소 (0.3 -> 0.2)
        """
        if self.p_pos.shape[0] == 0:
            return 0.0

        # 타겟 마스크에서 타겟 영역의 중심 계산
        y_indices, x_indices = np.where(self.target_mask)
        if len(y_indices) == 0:
            return 0.0

        # 정규화된 좌표로 변환 (0~1 범위)
        target_y = y_indices.mean() / self.mask_size
        target_x = x_indices.mean() / self.mask_size
        target_center = np.array([target_x, target_y])

        # 각 파티클에서 타겟 중심까지의 거리
        distances = np.linalg.norm(self.p_pos - target_center, axis=1)

        # 거리 기반 보상: 가까울수록 높은 보상
        # 0.2 이내의 거리에서만 보상 제공 (더 엄격하게)
        proximity_mask = distances < 0.2
        if proximity_mask.sum() == 0:
            return 0.0

        # 가장 가까운 파티클의 보상만 반환 (평균 대신)
        proximity_rewards = np.maximum(0, 0.2 - distances[proximity_mask])
        return proximity_rewards.max()  # mean() -> max()

    def _get_state(self):
        coarse, occ = self._compute_grids()
        t_norm = self.step_count / self.cfg.max_steps
        theta_norm = (self.theta - self.angle_min) / (self.angle_max - self.angle_min)
        speed_norm = (self.speed - self.speed_min) / (self.speed_max - self.speed_min)

        state_vec = np.concatenate(
            [
                np.array([t_norm, theta_norm, speed_norm], dtype=np.float32),
                coarse.flatten().astype(np.float32),
            ]
        )
        return state_vec

    def step(self, action: int):
        # action -> angle/speed update
        da_sign, dv_sign = self.action_deltas[action]
        angle_step = self.cfg.angle_step_deg
        speed_step = (self.speed_max - self.speed_min) * self.cfg.speed_step_frac

        self.theta = np.clip(
            self.theta + da_sign * angle_step, self.angle_min, self.angle_max
        )
        self.speed = np.clip(
            self.speed + dv_sign * speed_step, self.speed_min, self.speed_max
        )

        # 파티클 생성 + 업데이트
        self._spawn_particles()
        lost_frac = self._update_particles()

        coarse, occ = self._compute_grids()
        coverage = self._compute_coverage(occ)
        delta_cov = coverage - self.prev_coverage

        # 개선 3: 다층 보상 함수로 학습 안정화
        # 1) 커버리지 증가분 보상: 새로운 영역을 커버할 때 큰 보상
        # 2) 현재 커버리지 보상: 높은 커버리지를 유지하도록 유도
        # 3) 근접 보상: 타겟 근처에 파티클이 있으면 보상 (초기 탐색 개선)
        proximity_reward = self._compute_proximity_reward()

        reward = (
            self.cfg.coverage_reward_scale * delta_cov  # 커버리지 증가분
            + self.cfg.current_coverage_scale * coverage  # 현재 커버리지
            + self.cfg.proximity_reward_scale * proximity_reward  # 근접 보상
            - self.cfg.penalty_lambda * lost_frac  # 파티클 손실 페널티
        )

        self.prev_coverage = coverage

        self.step_count += 1
        done = self.step_count >= self.cfg.max_steps

        state = self._get_state()
        info = {
            "coverage": coverage,
            "proximity_reward": proximity_reward,
            "delta_coverage": delta_cov,
        }
        return state, reward, done, info

    def render(self, ax=None, show=True):
        """
        현재 파티클 분포 + 타겟 마스크를 2D로 시각화.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        # 타겟 마스크
        ax.imshow(
            self.target_mask,
            origin="lower",
            extent=[0, 1, 0, 1],
            alpha=0.3,
            cmap="Greens",
        )
        # 파티클
        if self.p_pos.shape[0] > 0:
            ax.scatter(self.p_pos[:, 0], self.p_pos[:, 1], s=10, c="C0", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Step {self.step_count}, coverage={self.prev_coverage:.3f}")
        if show:
            plt.show()
        return ax
