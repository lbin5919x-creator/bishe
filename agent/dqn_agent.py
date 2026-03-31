"""自适应交通信号控制的深度Q网络智能体。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.settings import training as train_cfg
from models.q_network import QNetwork, QNetworkConfig
from utils.replay_buffer import ReplayBuffer, Transition


@dataclass(slots=True)
class AgentConfig:
    """智能体配置参数。"""
    state_dim: int                                    # 状态维度
    action_dim: int                                   # 动作维度
    device: torch.device                              # 计算设备
    double_dqn: bool = True                           # 是否使用Double DQN
    dueling: bool = True                              # 是否使用Dueling网络
    gamma: float = train_cfg.gamma                    # 折扣因子
    lr: float = train_cfg.learning_rate               # 学习率
    epsilon_start: float = train_cfg.epsilon_start    # 初始探索率
    epsilon_end: float = train_cfg.epsilon_end        # 最终探索率
    epsilon_decay: int = train_cfg.epsilon_decay      # 探索率衰减步数
    target_update: int = train_cfg.target_update      # 目标网络更新频率
    batch_size: int = train_cfg.batch_size            # 批次大小
    buffer_capacity: int = train_cfg.buffer_capacity  # 经验回放缓冲区容量


class DQNAgent:
    """
    DQN智能体。
    
    实现深度Q网络算法，包含：
    - 策略网络和目标网络
    - 经验回放
    - ε-贪婪探索策略
    - 可选的Double DQN和Dueling网络
    """
    
    def __init__(self, config: AgentConfig) -> None:
        """
        初始化DQN智能体。
        
        参数:
            config: 智能体配置
        """
        self.config = config
        self.device = config.device

        # 创建Q网络
        q_config = QNetworkConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            dueling=config.dueling,
        )
        self.policy_net = QNetwork(q_config).to(self.device)  # 策略网络
        self.target_net = QNetwork(q_config).to(self.device)  # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config.buffer_capacity, self.device)
        
        # 训练参数
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        # 探索参数
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        # 计数器
        self.steps_done = 0
        self.double_dqn = config.double_dqn
        self.target_update = config.target_update

    def select_action(self, state: np.ndarray, exploit: bool = False) -> int:
        """
        选择动作（ε-贪婪策略）。

        训练时探索率由 decay_epsilon() 按回合衰减；exploit=True 时为纯贪心（ε=0）。

        参数:
            state: 当前状态
            exploit: 是否评估模式（不探索）

        返回:
            选择的动作
        """
        self.steps_done += 1

        # 评估(exploit)时纯贪心；训练时由 decay_epsilon() 与 self.epsilon 控制探索
        eps_threshold = 0.0 if exploit else self.epsilon

        # ε-贪婪：以 eps 概率随机探索
        if np.random.rand() < eps_threshold:
            return np.random.randint(0, self.config.action_dim)

        # 利用：选择Q值最大的动作
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, transition: Transition) -> None:
        """
        存储经验到回放缓冲区。
        
        参数:
            transition: 经验转换（状态、动作、奖励、下一状态、完成标志）
        """
        self.replay_buffer.push(transition)

    def update(self) -> Optional[float]:
        """
        更新网络参数。
        
        返回:
            损失值（如果进行了更新），否则返回None
        """
        if not self.replay_buffer.ready(self.batch_size):
            return None

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 计算当前Q值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN：用策略网络选动作，用目标网络评估
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # 标准DQN
                next_q_values, _ = self.target_net(next_states).max(dim=1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失并更新
        loss_fn: nn.Module = nn.SmoothL1Loss()
        loss = loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=train_cfg.gradient_clip)
        self.optimizer.step()

        # 定期更新目标网络
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        """衰减探索率。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * np.exp(-1.0 / self.epsilon_decay))

    def save(self, path: str) -> None:
        """
        保存完整训练快照（支持断点续训）。

        参数:
            path: 保存路径
        """
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
        }, path)

    def load(self, path: str, weights_only: bool = False) -> None:
        """
        加载模型。

        参数:
            path: 模型路径
            weights_only: 仅加载网络权重（用于评估），忽略训练状态
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 兼容旧格式（仅含 state_dict）
        if "policy_net" not in checkpoint:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
            return

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if not weights_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.steps_done = checkpoint.get("steps_done", self.steps_done)
