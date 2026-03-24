"""基于DQN的交通信号控制SUMO环境封装。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import traci  # type: ignore

from config.settings import SCENARIO_DIR, environment as env_cfg
from .phase_logic import PhaseController


@dataclass(slots=True)
class StepResult:
    """环境步进结果。"""
    state: np.ndarray          # 状态向量
    reward: float              # 奖励值
    done: bool                 # 是否结束
    info: Dict[str, float]     # 附加信息


class SumoEnvironment:
    """SUMO与强化学习智能体之间的接口。"""

    def __init__(self, scenario: str, max_steps: int, use_gui: bool = False, seed: int = None) -> None:
        """
        初始化SUMO环境。
        
        参数:
            scenario: 场景名称
            max_steps: 每回合最大步数
            use_gui: 是否使用图形界面
            seed: SUMO随机种子（用于确保车流可复现）
        """
        self.scenario = scenario
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.seed = seed
        self.sumo_cfg = str(SCENARIO_DIR / scenario / "simulation.sumocfg")

        self.tls_id: Optional[str] = None           # 信号灯ID
        self.lanes: List[str] = []                  # 受控车道列表
        self.edges: List[str] = []                  # 边列表
        self.phases: List[str] = []                 # 相位状态列表
        self.phase_controller: Optional[PhaseController] = None

        self.prev_wait_priority: float = 0.0        # 上一步优先车辆等待时间
        self.prev_wait_normal: float = 0.0          # 上一步普通车辆等待时间
        self.time_step: int = 0                     # 当前时间步
        self.total_throughput: int = 0              # 累计通行量
        self._pending_arrivals: int = 0             # 相位过渡期间累计到达车辆

    # ------------------------------------------------------------------
    # 公共API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态。"""
        if traci.isLoaded():
            traci.close(False)
        self._init_sumo(self.seed)
        self._retrieve_network_info()

        self.time_step = 0
        self.total_throughput = 0
        self._pending_arrivals = 0
        if self.phase_controller:
            self.phase_controller.reset()

        waits = self._split_waiting_times()
        self.prev_wait_priority, self.prev_wait_normal = waits
        return self._get_state()

    def step(self, action: int) -> StepResult:
        """
        执行一步动作。

        参数:
            action: 动作（0=保持相位，1=切换相位）

        返回:
            StepResult: 包含状态、奖励、完成标志和信息
        """
        assert self.phase_controller is not None and self.tls_id is not None

        try:
            if self.phase_controller.should_switch(action):
                sim_steps = self._apply_phase_transition()
                self.time_step += sim_steps
            else:
                self.phase_controller.keep_phase()
                traci.simulationStep()
                self._pending_arrivals += traci.simulation.getArrivedNumber()
                self.time_step += 1
        except Exception as e:
            print(f"SUMO错误: {e}")
            print(f"当前时间步: {self.time_step}")
            print(f"仿真是否结束: {traci.simulation.getMinExpectedNumber() <= 0 if traci.isLoaded() else 'N/A'}")
            raise

        state = self._get_state()
        reward = self._calculate_reward()
        done = self._is_done()
        info = self._collect_metrics()
        return StepResult(state, reward, done, info)

    def close(self) -> None:
        """关闭SUMO仿真。"""
        if traci.isLoaded():
            traci.close(False)

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _init_sumo(self, seed: int = None) -> None:
        """初始化SUMO仿真。"""
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError("未设置SUMO_HOME环境变量。")
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_binary = os.path.join(sumo_home, "bin", binary)
        
        cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--waiting-time-memory", "1000",
        ]
        
        # 添加随机种子以确保可复现性
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        traci.start(cmd)

    def _retrieve_network_info(self) -> None:
        """从SUMO获取路网信息。"""
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("SUMO场景中未找到信号灯系统。")
        self.tls_id = tls_ids[0]
        # 过滤掉内部车道（以:开头的）
        all_lanes = list(traci.trafficlight.getControlledLanes(self.tls_id))
        self.lanes = [lane for lane in all_lanes if not lane.startswith(':')]
        # 去重
        self.lanes = list(dict.fromkeys(self.lanes))
        self.edges = sorted({traci.lane.getEdgeID(lane) for lane in self.lanes if not lane.startswith(':')})
        definition = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        self.phases = [phase.state for phase in definition.phases]
        self.phase_controller = PhaseController(
            min_green=env_cfg.min_green,
            max_green=env_cfg.max_green,
            yellow=env_cfg.yellow,
            all_red=env_cfg.all_red,
            phases=self.phases,
        )

    def _apply_phase_transition(self) -> int:
        """
        应用相位过渡：黄灯 → 全红 → 下一绿灯。

        模拟真实信号灯的安全时序约束（黄灯4s + 全红2s），
        确保相位切换过程中车辆有足够的清空时间。

        返回:
            消耗的仿真步数
        """
        assert self.phase_controller is not None and self.tls_id is not None

        current_phase_idx = self.phase_controller.current_phase
        current_state = self.phases[current_phase_idx]
        num_phases = len(self.phases)
        next_phase_idx = (current_phase_idx + 1) % num_phases

        # 1. 黄灯过渡：将绿灯信号替换为黄灯
        yellow_state = current_state.replace('G', 'y').replace('g', 'y')
        traci.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
        for _ in range(env_cfg.yellow):
            traci.simulationStep()
            self._pending_arrivals += traci.simulation.getArrivedNumber()

        # 2. 全红保护：所有方向红灯，确保路口清空
        all_red_state = 'r' * len(current_state)
        traci.trafficlight.setRedYellowGreenState(self.tls_id, all_red_state)
        for _ in range(env_cfg.all_red):
            traci.simulationStep()
            self._pending_arrivals += traci.simulation.getArrivedNumber()

        # 3. 切换到下一绿灯相位（用 setRedYellowGreenState 避免内联程序相位越界）
        traci.trafficlight.setRedYellowGreenState(self.tls_id, self.phases[next_phase_idx])
        traci.simulationStep()
        self._pending_arrivals += traci.simulation.getArrivedNumber()

        # 更新相位控制器状态
        self.phase_controller.current_phase = next_phase_idx
        self.phase_controller.elapsed = 0

        return env_cfg.yellow + env_cfg.all_red + 1

    def _get_state(self) -> np.ndarray:
        """获取当前状态向量。"""
        state_features: List[float] = []
        for lane in self.lanes:
            queue_length = traci.lane.getLastStepHaltingNumber(lane)  # 排队长度
            occupancy = traci.lane.getLastStepOccupancy(lane)         # 占用率
            state_features.extend([queue_length, occupancy])

        # 相位独热编码
        phase_one_hot = np.zeros(len(self.phases), dtype=np.float32)
        if self.phase_controller:
            phase_one_hot[self.phase_controller.current_phase] = 1.0
        state_features.extend(phase_one_hot.tolist())

        state = np.array(state_features, dtype=np.float32)
        return self._normalize_state(state)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """归一化状态值到[0, 1]范围。"""
        if not self.lanes:
            return state
        max_queue = 50.0       # 最大排队长度
        max_occupancy = 100.0  # 最大占用率
        for idx in range(0, len(self.lanes) * 2, 2):
            state[idx] = min(state[idx] / max_queue, 1.0)
            state[idx + 1] = min(state[idx + 1] / max_occupancy, 1.0)
        return state

    def _split_waiting_times(self) -> Tuple[float, float]:
        """分别计算优先车辆和普通车辆的等待时间。"""
        priority, normal = 0.0, 0.0
        for vehicle_id in traci.vehicle.getIDList():
            waiting = traci.vehicle.getWaitingTime(vehicle_id)
            veh_type = traci.vehicle.getTypeID(vehicle_id)
            if veh_type.lower().startswith("priority"):
                priority += waiting
            else:
                normal += waiting
        return priority, normal

    def _calculate_reward(self) -> float:
        """计算奖励值（等待时间减少量）。"""
        wait_priority, wait_normal = self._split_waiting_times()
        delta_priority = self.prev_wait_priority - wait_priority
        delta_normal = self.prev_wait_normal - wait_normal
        reward = env_cfg.reward_alpha * delta_priority + (1 - env_cfg.reward_alpha) * delta_normal
        self.prev_wait_priority, self.prev_wait_normal = wait_priority, wait_normal
        return reward

    def _collect_metrics(self) -> Dict[str, float]:
        """收集性能指标。"""
        total_wait = sum(traci.edge.getWaitingTime(edge) for edge in self.edges)
        avg_queue = float(np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])) if self.lanes else 0.0
        
        # 累计通行量（包含相位过渡期间的到达车辆）
        self.total_throughput += self._pending_arrivals
        self._pending_arrivals = 0
        
        return {
            "waiting_time": total_wait,    # 总等待时间
            "avg_queue": avg_queue,        # 平均排队长度
            "throughput": self.total_throughput,  # 累计通行量
            "time": self.time_step * env_cfg.time_step,  # 仿真时间
        }

    def _is_done(self) -> bool:
        """判断回合是否结束。"""
        if self.time_step >= self.max_steps:
            return True
        return traci.simulation.getMinExpectedNumber() <= 0
