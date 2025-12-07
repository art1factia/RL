# agent/dqn.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.mem_s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.mem_a = np.zeros((capacity,), dtype=np.int64)
        self.mem_r = np.zeros((capacity,), dtype=np.float32)
        self.mem_ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.mem_d = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, s, a, r, ns, d):
        i = self.idx
        self.mem_s[i] = s
        self.mem_a[i] = a
        self.mem_r[i] = r
        self.mem_ns[i] = ns
        self.mem_d[i] = d
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        max_idx = len(self)
        idxs = np.random.randint(0, max_idx, size=batch_size)
        batch = dict(
            s=self.mem_s[idxs],
            a=self.mem_a[idxs],
            r=self.mem_r[idxs],
            ns=self.mem_ns[idxs],
            d=self.mem_d[idxs],
        )
        return batch


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # 개선 4: 네트워크 용량 증가 (128 -> 256 hidden units, 2 layers -> 3 layers)
        # 더 복잡한 Q-함수를 학습할 수 있도록 모델 크기 확장
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        lr: float = 3e-4,  # 개선 5: 학습률 조정 (1e-3 -> 3e-4, 안정적 학습)
        gamma: float = 0.99,
        buffer_capacity: int = 100000,  # 개선 6: 버퍼 크기 증가 (더 다양한 경험 저장)
        batch_size: int = 128,  # 개선 7: 배치 크기 증가 (64 -> 128, 안정적 그래디언트)
        eps_start: float = 1.0,
        eps_end: float = 0.01,  # 개선 8: 최소 epsilon 감소 (0.05 -> 0.01, 더 많은 활용)
        eps_decay_steps: int = 20000,  # 개선 9: 더 긴 탐색 기간 (10000 -> 20000)
        target_update_interval: int = 500,  # 개선 10: 타겟 네트워크 업데이트 빈도 증가
        warmup_steps: int = 1000,  # 개선 11: 학습 시작 전 경험 수집
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.warmup_steps = warmup_steps

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim)

        # epsilon-greedy
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0

    def epsilon(self):
        # 선형 감쇠
        frac = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        # greedy
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax(dim=1).item())

    def push_transition(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, float(done))
        self.total_steps += 1

    def update(self):
        # 개선 12: 워밍업 단계에서는 학습하지 않고 경험만 수집
        # 충분한 경험이 쌓인 후 학습 시작하여 초기 불안정성 감소
        if len(self.buffer) < self.batch_size or self.total_steps < self.warmup_steps:
            return None

        batch = self.buffer.sample(self.batch_size)
        s = torch.tensor(batch["s"], dtype=torch.float32, device=self.device)
        a = torch.tensor(batch["a"], dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(batch["r"], dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(batch["ns"], dtype=torch.float32, device=self.device)
        d = torch.tensor(batch["d"], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_net(s).gather(1, a)

        # target: r + gamma * max_a' Q_target(ns, a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1.0 - d)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network 업데이트
        if self.total_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
