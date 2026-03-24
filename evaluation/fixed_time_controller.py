"""传统定时交通信号控制器，用于基线对比。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config.settings import FIXED_TIME_PLANS, environment as env_cfg


@dataclass
class FixedTimeController:
    """
    定时信号控制器。
    
    使用预设的固定配时方案控制信号灯。
    """
    scenario: str
    plan: List[int] = field(default_factory=list)
    num_phases: int = 0
    cycle: int = 0

    def __post_init__(self) -> None:
        """初始化定时方案。"""
        if self.scenario not in FIXED_TIME_PLANS:
            raise ValueError(f"场景'{self.scenario}'没有定义定时方案")
        self.plan = FIXED_TIME_PLANS[self.scenario]
        self.num_phases = len(self.plan)
        self.cycle = sum(self.plan) + self.num_phases * (env_cfg.yellow + env_cfg.all_red)

    def select_action(self, time_step: int) -> int:
        """
        根据时间步选择动作。
        
        参数:
            time_step: 当前时间步
            
        返回:
            动作（0=保持，在相位边界时可能切换）
        """
        t = time_step % self.cycle
        elapsed = 0
        
        for phase_idx, green in enumerate(self.plan):
            phase_duration = green + env_cfg.yellow + env_cfg.all_red
            if t < elapsed + green:
                # 在绿灯期间，保持
                return 0
            elif t < elapsed + phase_duration:
                # 在黄灯/全红期间，已经在过渡
                return 0
            elapsed += phase_duration
        
        return 0

    def get_current_phase(self, time_step: int) -> int:
        """
        获取当前应该处于的相位。
        
        参数:
            time_step: 当前时间步
            
        返回:
            相位索引
        """
        t = time_step % self.cycle
        elapsed = 0
        
        for phase_idx, green in enumerate(self.plan):
            phase_duration = green + env_cfg.yellow + env_cfg.all_red
            if t < elapsed + phase_duration:
                return phase_idx
            elapsed += phase_duration
        
        return 0
