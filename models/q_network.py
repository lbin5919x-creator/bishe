"""Q值近似的神经网络架构。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(slots=True)
class QNetworkConfig:
    """Q网络配置参数。"""
    state_dim: int                                    # 状态维度
    action_dim: int                                   # 动作维度
    hidden_dim: int = 128                             # 隐藏层维度
    hidden_layers: int = 2                            # 隐藏层数量
    dueling: bool = False                             # 是否使用Dueling架构
    activation: Callable[[], nn.Module] = nn.ReLU     # 激活函数


class QNetwork(nn.Module):
    """
    Q值神经网络。
    
    支持标准MLP和Dueling架构。
    Dueling架构将Q值分解为状态价值V(s)和优势函数A(s,a)。
    """
    
    def __init__(self, config: QNetworkConfig) -> None:
        """
        初始化Q网络。
        
        参数:
            config: 网络配置
        """
        super().__init__()
        self.config = config

        # 特征提取层
        layers = []
        input_dim = config.state_dim
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(config.activation())
            input_dim = config.hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

        if config.dueling:
            # Dueling架构：分离价值流和优势流
            self.value_head = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                config.activation(),
                nn.Linear(config.hidden_dim, 1),  # 输出状态价值V(s)
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                config.activation(),
                nn.Linear(config.hidden_dim, config.action_dim),  # 输出优势A(s,a)
            )
        else:
            # 标准架构：直接输出Q值
            self.q_head = nn.Linear(input_dim, config.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入状态张量
            
        返回:
            Q值张量
        """
        features = self.feature_extractor(x)
        
        if self.config.dueling:
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values
