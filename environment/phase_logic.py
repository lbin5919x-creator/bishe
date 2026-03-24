"""信号相位管理和安全时序工具。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class PhaseTiming:
    """相位时序信息。"""
    phase_index: int    # 相位索引
    duration: int       # 持续时长


@dataclass(slots=True)
class PhaseController:
    """
    信号相位控制器。
    
    管理信号灯相位切换，确保满足最小绿灯、最大绿灯等安全约束。
    """
    min_green: int          # 最小绿灯时长（秒）
    max_green: int          # 最大绿灯时长（秒）
    yellow: int             # 黄灯时长（秒）
    all_red: int            # 全红时长（秒）
    phases: List[str]       # 相位状态列表

    current_phase: int = 0  # 当前相位索引
    elapsed: int = 0        # 当前相位已持续时间

    def reset(self) -> None:
        """重置相位控制器。"""
        self.current_phase = 0
        self.elapsed = 0

    def should_switch(self, action: int) -> bool:
        """
        判断是否应该切换到下一相位。
        
        参数:
            action: 动作（0=保持，1=切换）
            
        返回:
            是否应该切换
        """
        # 未达到最小绿灯时长，不允许切换
        if self.elapsed < self.min_green:
            return False
        # 智能体请求切换且已达到最小绿灯时长
        if action == 1 and self.elapsed >= self.min_green:
            return True
        # 达到最大绿灯时长，强制切换
        if self.elapsed >= self.max_green:
            return True
        return False

    def next_phase(self) -> PhaseTiming:
        """
        切换到下一相位。
        
        返回:
            PhaseTiming: 新相位的时序信息
        """
        self.current_phase = (self.current_phase + 1) % len(self.phases)
        self.elapsed = 0
        return PhaseTiming(self.current_phase, self.yellow + self.all_red)

    def keep_phase(self) -> None:
        """保持当前相位，增加已持续时间。"""
        self.elapsed += 1

    def start_phase(self, phase_index: int) -> None:
        """
        启动指定相位。
        
        参数:
            phase_index: 相位索引
        """
        self.current_phase = phase_index
        self.elapsed = 0
