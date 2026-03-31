"""传统定时交通信号控制器，用于基线对比。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config.settings import FIXED_TIME_PLANS, environment as env_cfg
from environment.phase_logic import PhaseController


def encode_joint_action(junction_actions: List[int]) -> int:
    """将各路口 0/1 动作编码为与 CascadedSumoEnvironment 一致的整数。"""
    action = 0
    for idx, a in enumerate(junction_actions):
        action += a * (2 ** idx)
    return action


@dataclass
class FixedTimeController:
    """
    定时信号控制器。

    使用预设的固定配时方案控制信号灯：在当前相位绿灯时长达到计划值
    （且满足最小绿灯约束）时请求切换，否则保持。
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

    def select_action(self, current_phase: int, elapsed_in_phase: int) -> int:
        """
        根据当前相位与相位内已持续绿灯时间选择动作。

        参数:
            current_phase: 当前相位索引
            elapsed_in_phase: 当前相位已持续秒数（与 PhaseController.elapsed 一致）

        返回:
            0=保持，1=切换至下一相位
        """
        if self.num_phases <= 0:
            return 0
        green = self.plan[current_phase % self.num_phases]
        if elapsed_in_phase >= env_cfg.max_green:
            return 1
        if elapsed_in_phase >= green and elapsed_in_phase >= env_cfg.min_green:
            return 1
        return 0

    def joint_action_cascaded(
        self,
        phase_ctrls: dict[str, PhaseController],
        tls_order: List[str],
    ) -> int:
        """
        多路口：按各路口相位状态生成联合动作编码。

        参数:
            phase_ctrls: tls_id -> PhaseController
            tls_order: 路口顺序（须与 CascadedSumoEnvironment.junction_ids 一致）
        """
        actions: List[int] = []
        for tls_id in tls_order:
            ctrl = phase_ctrls[tls_id]
            actions.append(self.select_action(ctrl.current_phase, ctrl.elapsed))
        return encode_joint_action(actions)

    def get_current_phase(self, time_step: int) -> int:
        """
        获取当前应该处于的相位（仅按周期时间推算，不依赖真实仿真状态）。

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
