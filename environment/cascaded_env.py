"""多路口级联环境，用于协同交通信号控制。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import traci

from config.settings import SCENARIO_DIR, environment as env_cfg, cascaded as cascaded_cfg
from .phase_logic import PhaseController


@dataclass(slots=True)
class CascadedStepResult:
    """多路口环境的步进结果。""" 
    state: np.ndarray          # 状态向量
    reward: float              # 总奖励
    done: bool                 # 是否结束
    info: Dict[str, float]     # 附加信息
    junction_rewards: Dict[str, float]  # 各路口单独奖励


@dataclass
class JunctionInfo:
    """单个路口的信息。"""
    tls_id: str                                    # 信号灯ID
    lanes: List[str] = field(default_factory=list)           # 受控车道列表
    incoming_edges: List[str] = field(default_factory=list)  # 进口边列表
    outgoing_edges: List[str] = field(default_factory=list)  # 出口边列表
    phases: List[str] = field(default_factory=list)          # 相位状态列表
    phase_controller: Optional[PhaseController] = None       # 相位控制器
    prev_wait_priority: float = 0.0                          # 上一步优先车辆等待时间
    prev_wait_normal: float = 0.0                            # 上一步普通车辆等待时间


class CascadedSumoEnvironment:
    """
    多路口级联环境，支持下游状态反馈。
    
    实现级联路口的协同控制，上游决策会影响下游排队状态。
    """

    def __init__(
        self,
        scenario: str = "cascaded_intersection",
        max_steps: int = 3600,
        use_gui: bool = False,
        coordination_weight: float = 0.3,
        seed: int | None = None,
    ) -> None:
        """
        初始化级联环境。
        
        参数:
            scenario: 场景名称
            max_steps: 每回合最大步数
            use_gui: 是否使用SUMO图形界面
            coordination_weight: 协同奖励权重
            seed: SUMO 随机种子，None 表示由 SUMO 默认随机
        """
        self.scenario = scenario
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.sumo_cfg = str(SCENARIO_DIR / scenario / "simulation.sumocfg")
        
        # 协同控制参数
        self.coordination_weight = coordination_weight  # 下游反馈权重
        self.seed = seed

        # 路口管理
        self.junctions: Dict[str, JunctionInfo] = {}
        self.junction_ids: List[str] = []
        self.link_edges: List[str] = []  # 连接路口的边（连接路）
        
        self.time_step: int = 0
        self.prev_total_wait: float = 0.0
        self.total_throughput: int = 0
        self._pending_arrivals: int = 0

    @property
    def num_junctions(self) -> int:
        """路口数量。"""
        return len(self.junction_ids)

    @property
    def state_dim(self) -> int:
        """状态维度：每路口状态 + 共享连接路状态。"""
        if not self.junctions:
            return 0
        sample_junction = next(iter(self.junctions.values()))
        per_junction_dim = len(sample_junction.lanes) * 2 + len(sample_junction.phases)
        link_dim = len(self.link_edges) * 2  # 连接路的排队和占用率
        return per_junction_dim * self.num_junctions + link_dim

    @property
    def action_dim(self) -> int:
        """动作空间：n个路口有2^n个动作（每个可保持/切换）。"""
        return 2 ** self.num_junctions

    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态。"""
        if traci.isLoaded():
            traci.close(False)
        self._init_sumo()
        self._retrieve_network_info()
        
        self.time_step = 0
        self.total_throughput = 0
        self._pending_arrivals = 0
        for junction in self.junctions.values():
            if junction.phase_controller:
                junction.phase_controller.reset()
            p, n = self._split_junction_waiting_times(junction)
            junction.prev_wait_priority = p
            junction.prev_wait_normal = n

        return self._get_state()

    def step(self, action: int) -> CascadedStepResult:
        """
        执行所有路口的动作。

        参数:
            action: 编码所有路口动作的整数
                   对于2个路口: 0=保持-保持, 1=切换-保持, 2=保持-切换, 3=切换-切换

        返回:
            CascadedStepResult: 包含状态、奖励、完成标志和信息的结果
        """
        # 解码每个路口的动作
        junction_actions = self._decode_action(action)

        # 找出需要切换和不需要切换的路口
        switching_junctions: List[JunctionInfo] = []
        for idx, jid in enumerate(self.junction_ids):
            junction = self.junctions[jid]
            if junction.phase_controller is None:
                continue

            if junction.phase_controller.should_switch(junction_actions[idx]):
                switching_junctions.append(junction)
            else:
                junction.phase_controller.keep_phase()

        # 如果有路口需要切换，执行黄灯→全红→绿灯过渡
        if switching_junctions:
            sim_steps = self._apply_synchronized_transition(switching_junctions)
            self.time_step += sim_steps
        else:
            traci.simulationStep()
            self._pending_arrivals += traci.simulation.getArrivedNumber()
            self.time_step += 1

        # 计算奖励
        junction_rewards = self._calculate_junction_rewards()
        coordination_bonus = self._calculate_coordination_bonus()
        total_reward = sum(junction_rewards.values()) + coordination_bonus

        state = self._get_state()
        done = self._is_done()
        info = self._collect_metrics()
        info["coordination_bonus"] = coordination_bonus

        return CascadedStepResult(state, total_reward, done, info, junction_rewards)

    def close(self) -> None:
        """关闭SUMO仿真。"""
        if traci.isLoaded():
            traci.close(False)

    # ------------------------------------------------------------------
    # 动作编码/解码
    # ------------------------------------------------------------------
    def _decode_action(self, action: int) -> List[int]:
        """将整数动作解码为每路口动作列表。"""
        actions = []
        for _ in range(self.num_junctions):
            actions.append(action % 2)
            action //= 2
        return actions

    def encode_action(self, junction_actions: List[int]) -> int:
        """将每路口动作列表编码为单个整数。"""
        action = 0
        for idx, a in enumerate(junction_actions):
            action += a * (2 ** idx)
        return action

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _init_sumo(self) -> None:
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
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])
        traci.start(cmd)

    def _retrieve_network_info(self) -> None:
        """获取所有信号灯的信息。"""
        tls_ids = list(traci.trafficlight.getIDList())
        if not tls_ids:
            raise RuntimeError("未找到信号灯系统。")
        
        self.junction_ids = sorted(tls_ids)  # 确保顺序一致
        self.junctions.clear()
        
        for tls_id in self.junction_ids:
            # 过滤掉内部车道（以:开头的）
            all_lanes = list(traci.trafficlight.getControlledLanes(tls_id))
            lanes = [lane for lane in all_lanes if not lane.startswith(':')]
            lanes = list(dict.fromkeys(lanes))  # 去重保持顺序
            incoming = sorted({traci.lane.getEdgeID(lane) for lane in lanes if not lane.startswith(':')})
            
            # 获取出口边
            outgoing = set()
            for lane in lanes:
                links = traci.lane.getLinks(lane)
                for link in links:
                    out_lane = link[0]
                    outgoing.add(traci.lane.getEdgeID(out_lane))
            
            definition = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            phases = [phase.state for phase in definition.phases]
            
            phase_controller = PhaseController(
                min_green=env_cfg.min_green,
                max_green=env_cfg.max_green,
                yellow=env_cfg.yellow,
                all_red=env_cfg.all_red,
                phases=phases,
            )
            
            self.junctions[tls_id] = JunctionInfo(
                tls_id=tls_id,
                lanes=lanes,
                incoming_edges=incoming,
                outgoing_edges=sorted(outgoing),
                phases=phases,
                phase_controller=phase_controller,
            )
        
        # 识别连接边（连接两个路口的道路）
        self._identify_link_edges()

    def _identify_link_edges(self) -> None:
        """找出连接两个路口的边（共享道路）。"""
        all_incoming = set()
        all_outgoing = set()
        
        for junction in self.junctions.values():
            all_incoming.update(junction.incoming_edges)
            all_outgoing.update(junction.outgoing_edges)
        
        # 连接边是既作为一个路口的进口又作为另一个路口出口的边
        self.link_edges = sorted(all_incoming & all_outgoing)

    def _apply_synchronized_transition(self, junctions: List[JunctionInfo]) -> int:
        """
        对多个路口同步执行黄灯→全红→绿灯过渡。

        所有需要切换的路口共享同一个过渡时间窗口，确保仿真步数一致。

        参数:
            junctions: 需要切换相位的路口列表

        返回:
            消耗的仿真步数
        """
        # 记录每个路口的当前和下一相位
        transitions = []
        for junction in junctions:
            current_idx = junction.phase_controller.current_phase
            current_state = junction.phases[current_idx]
            next_idx = (current_idx + 1) % len(junction.phases)
            transitions.append((junction, current_state, next_idx))

        # 1. 黄灯过渡：将绿灯信号替换为黄灯
        for junction, current_state, _ in transitions:
            yellow_state = current_state.replace('G', 'y').replace('g', 'y')
            traci.trafficlight.setRedYellowGreenState(junction.tls_id, yellow_state)
        for _ in range(env_cfg.yellow):
            traci.simulationStep()
            self._pending_arrivals += traci.simulation.getArrivedNumber()

        # 2. 全红保护：所有方向红灯，确保路口清空
        for junction, current_state, _ in transitions:
            all_red_state = 'r' * len(current_state)
            traci.trafficlight.setRedYellowGreenState(junction.tls_id, all_red_state)
        for _ in range(env_cfg.all_red):
            traci.simulationStep()
            self._pending_arrivals += traci.simulation.getArrivedNumber()

        # 3. 切换到下一绿灯相位（用 setRedYellowGreenState 避免内联程序相位越界）
        for junction, _, next_idx in transitions:
            traci.trafficlight.setRedYellowGreenState(
                junction.tls_id, junction.phases[next_idx])
            junction.phase_controller.current_phase = next_idx
            junction.phase_controller.elapsed = 0
        traci.simulationStep()
        self._pending_arrivals += traci.simulation.getArrivedNumber()

        return env_cfg.yellow + env_cfg.all_red + 1

    def _apply_phase_transition(self, junction: JunctionInfo) -> None:
        """应用单个路口的相位过渡（向后兼容）。"""
        self._apply_synchronized_transition([junction])

    def _get_state(self) -> np.ndarray:
        """
        获取所有路口的组合状态。
        
        状态包括：
        - 每路口：排队长度、占用率、当前相位（独热编码）
        - 连接路：排队和占用率（用于协同感知）
        """
        state_features: List[float] = []
        
        # 每路口状态
        for jid in self.junction_ids:
            junction = self.junctions[jid]
            
            for lane in junction.lanes:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                occupancy = traci.lane.getLastStepOccupancy(lane)
                state_features.extend([queue, occupancy])
            
            # 相位独热编码
            phase_one_hot = np.zeros(len(junction.phases), dtype=np.float32)
            if junction.phase_controller:
                phase_one_hot[junction.phase_controller.current_phase] = 1.0
            state_features.extend(phase_one_hot.tolist())
        
        # 连接路状态（下游反馈）
        for edge in self.link_edges:
            lanes = traci.edge.getLaneNumber(edge)
            total_queue = sum(
                traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                for i in range(lanes)
            )
            avg_occupancy = np.mean([
                traci.lane.getLastStepOccupancy(f"{edge}_{i}")
                for i in range(lanes)
            ])
            state_features.extend([total_queue, avg_occupancy])
        
        state = np.array(state_features, dtype=np.float32)
        return self._normalize_state(state)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """将状态值归一化到[0, 1]范围。"""
        max_queue = 50.0
        max_occupancy = 100.0
        
        idx = 0
        for jid in self.junction_ids:
            junction = self.junctions[jid]
            num_lanes = len(junction.lanes)
            
            for i in range(num_lanes):
                state[idx] = min(state[idx] / max_queue, 1.0)
                state[idx + 1] = min(state[idx + 1] / max_occupancy, 1.0)
                idx += 2
            
            idx += len(junction.phases)  # 跳过相位独热编码
        
        # 归一化连接路特征
        for _ in self.link_edges:
            state[idx] = min(state[idx] / (max_queue * 2), 1.0)
            state[idx + 1] = min(state[idx + 1] / max_occupancy, 1.0)
            idx += 2
        
        return state

    def _get_junction_waiting_time(self, tls_id: str) -> float:
        """获取某路口车辆的总等待时间。"""
        junction = self.junctions[tls_id]
        total_wait = 0.0
        for edge in junction.incoming_edges:
            total_wait += traci.edge.getWaitingTime(edge)
        return total_wait

    def _split_junction_waiting_times(self, junction: JunctionInfo) -> Tuple[float, float]:
        """分别计算某路口优先车辆和普通车辆的等待时间（与单路口环境一致）。"""
        priority, normal = 0.0, 0.0
        incoming_edges_set = set(junction.incoming_edges)
        for vehicle_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(vehicle_id)
            if road_id not in incoming_edges_set:
                continue
            waiting = traci.vehicle.getWaitingTime(vehicle_id)
            veh_type = traci.vehicle.getTypeID(vehicle_id)
            if veh_type.lower().startswith("priority"):
                priority += waiting
            else:
                normal += waiting
        return priority, normal

    def _get_downstream_queue(self, junction: JunctionInfo) -> float:
        """获取某路口下游连接路段的排队长度（用于逐路口下游反馈）。"""
        queue = 0.0
        for edge in self.link_edges:
            # 连接路是该路口的出口边 → 它通向下游路口
            if edge in junction.outgoing_edges:
                lanes = traci.edge.getLaneNumber(edge)
                queue += sum(
                    traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                    for i in range(lanes)
                )
        return queue

    def _calculate_junction_rewards(self) -> Dict[str, float]:
        """
        计算每个路口的奖励。

        与单路口环境一致，区分优先/普通车辆加权奖励；
        同时将该路口下游连接路段的排队作为惩罚项纳入，
        实现开题报告中"下游状态反馈"的协同机制。
        """
        alpha = env_cfg.reward_alpha
        rewards = {}

        for jid in self.junction_ids:
            junction = self.junctions[jid]
            cur_priority, cur_normal = self._split_junction_waiting_times(junction)

            # 等待时间改善量（与单路口奖励公式一致）
            delta_priority = junction.prev_wait_priority - cur_priority
            delta_normal = junction.prev_wait_normal - cur_normal
            local_reward = alpha * delta_priority + (1 - alpha) * delta_normal

            # 下游连接路排队惩罚（逐路口级别）
            downstream_queue = self._get_downstream_queue(junction)
            downstream_penalty = -(cascaded_cfg.link_queue_penalty
                                   * downstream_queue
                                   / cascaded_cfg.max_link_queue)

            rewards[jid] = local_reward + self.coordination_weight * downstream_penalty

            # 更新历史等待时间
            junction.prev_wait_priority = cur_priority
            junction.prev_wait_normal = cur_normal

        return rewards

    def _calculate_coordination_bonus(self) -> float:
        """
        全局协同奖励：惩罚所有连接路段的总排队积压。

        使用 CascadedConfig 中的参数，归一化后施加惩罚。
        """
        if not self.link_edges:
            return 0.0

        total_link_queue = 0.0
        for edge in self.link_edges:
            lanes = traci.edge.getLaneNumber(edge)
            total_link_queue += sum(
                traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                for i in range(lanes)
            )

        # 归一化后加权惩罚
        normalized = total_link_queue / (cascaded_cfg.max_link_queue * len(self.link_edges))
        return -cascaded_cfg.link_queue_penalty * self.coordination_weight * normalized

    def _collect_metrics(self) -> Dict[str, float]:
        """收集性能指标。"""
        total_wait = 0.0
        total_queue = 0.0
        num_lanes = 0
        
        for junction in self.junctions.values():
            for edge in junction.incoming_edges:
                total_wait += traci.edge.getWaitingTime(edge)
            for lane in junction.lanes:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                num_lanes += 1
        
        # 连接路指标
        link_queue = 0.0
        for edge in self.link_edges:
            lanes = traci.edge.getLaneNumber(edge)
            link_queue += sum(
                traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                for i in range(lanes)
            )
        
        # 累计通行量（包含相位过渡期间的到达车辆）
        self.total_throughput += self._pending_arrivals
        self._pending_arrivals = 0
        
        return {
            "waiting_time": total_wait,      # 总等待时间
            "avg_queue": total_queue / max(num_lanes, 1),  # 平均排队长度
            "link_queue": link_queue,        # 连接路排队
            "throughput": self.total_throughput,  # 累计通行量
            "time": self.time_step * env_cfg.time_step,  # 仿真时间
        }

    def _is_done(self) -> bool:
        """判断回合是否结束。"""
        if self.time_step >= self.max_steps:
            return True
        return traci.simulation.getMinExpectedNumber() <= 0
