"""Experience replay buffer for DQN agents."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.device = device

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]

        state_batch = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
