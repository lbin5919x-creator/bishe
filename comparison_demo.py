"""固定配时 vs DQN 逐一对比演示。

先运行固定配时 SUMO GUI 演示，关闭后再运行 DQN 演示，
两次使用相同随机种子确保车流一致，结束后生成对比图表。

用法:
    python comparison_demo.py --scenario t_intersection --duration 600
    python comparison_demo.py --scenario x_intersection --duration 600
    python comparison_demo.py --scenario cascaded_intersection --duration 600
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import traci

from agent.dqn_agent import DQNAgent, AgentConfig
from config.settings import (
    FIXED_TIME_PLANS,
    MODEL_DIR,
    SCENARIO_DIR,
    environment as env_cfg,
)
from environment.phase_logic import PhaseController
from evaluation.fixed_time_controller import FixedTimeController


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class NetworkInfo:
    """从 SUMO 获取的路网结构信息。"""
    tls_ids: List[str]
    lanes: Dict[str, List[str]]
    edges: Dict[str, List[str]]
    phases: Dict[str, List[str]]
    link_edges: List[str] = field(default_factory=list)


@dataclass
class StepMetrics:
    """单步指标。"""
    step: int
    waiting_time: float
    avg_queue: float
    throughput: int
    action: int


# ======================================================================
# TraCI 工具函数
# ======================================================================

def _sumo_binary() -> str:
    """获取 sumo-gui 可执行文件路径。"""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("未设置 SUMO_HOME 环境变量。")
    return os.path.join(sumo_home, "bin", "sumo-gui")


def start_sumo(scenario: str, seed: int) -> None:
    """启动单个 SUMO GUI 实例。"""
    sumo_cfg = str(SCENARIO_DIR / scenario / "simulation.sumocfg")
    binary = _sumo_binary()
    cmd = [
        binary, "-c", sumo_cfg,
        "--start",
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--waiting-time-memory", "1000",
        "--seed", str(seed),
    ]
    traci.start(cmd)


def get_network_info() -> NetworkInfo:
    """从当前 SUMO 实例获取路网信息。"""
    tls_ids = sorted(traci.trafficlight.getIDList())
    if not tls_ids:
        raise RuntimeError("未找到信号灯系统。")

    lanes_map: Dict[str, List[str]] = {}
    edges_map: Dict[str, List[str]] = {}
    phases_map: Dict[str, List[str]] = {}

    all_incoming: set[str] = set()
    all_outgoing: set[str] = set()

    for tls_id in tls_ids:
        raw_lanes = list(traci.trafficlight.getControlledLanes(tls_id))
        lanes = list(dict.fromkeys(l for l in raw_lanes if not l.startswith(':')))
        incoming = sorted({traci.lane.getEdgeID(l) for l in lanes})
        outgoing: set[str] = set()
        for lane in lanes:
            for link in traci.lane.getLinks(lane):
                outgoing.add(traci.lane.getEdgeID(link[0]))

        definition = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        phases = [p.state for p in definition.phases]

        lanes_map[tls_id] = lanes
        edges_map[tls_id] = incoming
        phases_map[tls_id] = phases

        all_incoming.update(incoming)
        all_outgoing.update(outgoing)

    link_edges = sorted(all_incoming & all_outgoing)

    return NetworkInfo(
        tls_ids=tls_ids,
        lanes=lanes_map,
        edges=edges_map,
        phases=phases_map,
        link_edges=link_edges,
    )


# ======================================================================
# 状态构造
# ======================================================================

def _get_single_state(net: NetworkInfo, phase_ctrls: Dict[str, PhaseController]) -> np.ndarray:
    """单路口状态向量。"""
    tls_id = net.tls_ids[0]
    features: List[float] = []
    for lane in net.lanes[tls_id]:
        features.append(traci.lane.getLastStepHaltingNumber(lane))
        features.append(traci.lane.getLastStepOccupancy(lane))
    phase_one_hot = np.zeros(len(net.phases[tls_id]), dtype=np.float32)
    phase_one_hot[phase_ctrls[tls_id].current_phase] = 1.0
    features.extend(phase_one_hot.tolist())

    state = np.array(features, dtype=np.float32)
    for idx in range(0, len(net.lanes[tls_id]) * 2, 2):
        state[idx] = min(state[idx] / 50.0, 1.0)
        state[idx + 1] = min(state[idx + 1] / 100.0, 1.0)
    return state


def _get_cascaded_state(net: NetworkInfo, phase_ctrls: Dict[str, PhaseController]) -> np.ndarray:
    """级联多路口联合状态向量。"""
    features: List[float] = []
    for tls_id in net.tls_ids:
        for lane in net.lanes[tls_id]:
            features.append(traci.lane.getLastStepHaltingNumber(lane))
            features.append(traci.lane.getLastStepOccupancy(lane))
        phase_one_hot = np.zeros(len(net.phases[tls_id]), dtype=np.float32)
        phase_one_hot[phase_ctrls[tls_id].current_phase] = 1.0
        features.extend(phase_one_hot.tolist())

    for edge in net.link_edges:
        nlanes = traci.edge.getLaneNumber(edge)
        total_q = sum(traci.lane.getLastStepHaltingNumber(f"{edge}_{i}") for i in range(nlanes))
        avg_occ = np.mean([traci.lane.getLastStepOccupancy(f"{edge}_{i}") for i in range(nlanes)])
        features.extend([total_q, avg_occ])

    state = np.array(features, dtype=np.float32)
    idx = 0
    for tls_id in net.tls_ids:
        for _ in net.lanes[tls_id]:
            state[idx] = min(state[idx] / 50.0, 1.0)
            state[idx + 1] = min(state[idx + 1] / 100.0, 1.0)
            idx += 2
        idx += len(net.phases[tls_id])
    for _ in net.link_edges:
        state[idx] = min(state[idx] / 100.0, 1.0)
        state[idx + 1] = min(state[idx + 1] / 100.0, 1.0)
        idx += 2
    return state


def get_state(net: NetworkInfo, phase_ctrls: Dict[str, PhaseController]) -> np.ndarray:
    """根据路口数量自动选择状态构造方式。"""
    if len(net.tls_ids) == 1:
        return _get_single_state(net, phase_ctrls)
    return _get_cascaded_state(net, phase_ctrls)


# ======================================================================
# 仿真步进封装（确保到达车辆计数不丢失）
# ======================================================================

_pending_arrivals: int = 0


def _sim_step() -> None:
    """执行一步仿真并累计到达车辆数。"""
    global _pending_arrivals
    traci.simulationStep()
    _pending_arrivals += traci.simulation.getArrivedNumber()


def _flush_arrivals() -> int:
    """读取并清零累计到达车辆数。"""
    global _pending_arrivals
    count = _pending_arrivals
    _pending_arrivals = 0
    return count


# ======================================================================
# 动作执行（含黄灯 + 全红过渡）
# ======================================================================

def apply_phase_transition(tls_id: str, phases: List[str], ctrl: PhaseController) -> int:
    """执行黄灯->全红->绿灯过渡，返回消耗的仿真步数。"""
    current_idx = ctrl.current_phase
    current_state = phases[current_idx]
    next_idx = (current_idx + 1) % len(phases)

    yellow_state = current_state.replace('G', 'y').replace('g', 'y')
    traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
    for _ in range(env_cfg.yellow):
        _sim_step()

    all_red = 'r' * len(current_state)
    traci.trafficlight.setRedYellowGreenState(tls_id, all_red)
    for _ in range(env_cfg.all_red):
        _sim_step()

    traci.trafficlight.setRedYellowGreenState(tls_id, phases[next_idx])
    _sim_step()

    ctrl.current_phase = next_idx
    ctrl.elapsed = 0
    return env_cfg.yellow + env_cfg.all_red + 1


def apply_action_single(
    action: int,
    net: NetworkInfo,
    phase_ctrls: Dict[str, PhaseController],
) -> int:
    """单路口：执行动作，返回消耗的仿真步数。"""
    tls_id = net.tls_ids[0]
    ctrl = phase_ctrls[tls_id]
    if ctrl.should_switch(action):
        return apply_phase_transition(tls_id, net.phases[tls_id], ctrl)
    ctrl.keep_phase()
    _sim_step()
    return 1


def apply_action_cascaded(
    action: int,
    net: NetworkInfo,
    phase_ctrls: Dict[str, PhaseController],
) -> int:
    """级联：解码联合动作并执行，返回消耗的仿真步数。"""
    n = len(net.tls_ids)
    actions = []
    a = action
    for _ in range(n):
        actions.append(a % 2)
        a //= 2

    switching = []
    for idx, tls_id in enumerate(net.tls_ids):
        ctrl = phase_ctrls[tls_id]
        if ctrl.should_switch(actions[idx]):
            switching.append(tls_id)
        else:
            ctrl.keep_phase()

    if switching:
        transitions = []
        for tls_id in switching:
            cur_idx = phase_ctrls[tls_id].current_phase
            cur_state = net.phases[tls_id][cur_idx]
            nxt_idx = (cur_idx + 1) % len(net.phases[tls_id])
            transitions.append((tls_id, cur_state, nxt_idx))

        for tls_id, cur_state, _ in transitions:
            traci.trafficlight.setRedYellowGreenState(
                tls_id, cur_state.replace('G', 'y').replace('g', 'y'))
        for _ in range(env_cfg.yellow):
            _sim_step()

        for tls_id, cur_state, _ in transitions:
            traci.trafficlight.setRedYellowGreenState(tls_id, 'r' * len(cur_state))
        for _ in range(env_cfg.all_red):
            _sim_step()

        for tls_id, _, nxt_idx in transitions:
            traci.trafficlight.setRedYellowGreenState(
                tls_id, net.phases[tls_id][nxt_idx])
            phase_ctrls[tls_id].current_phase = nxt_idx
            phase_ctrls[tls_id].elapsed = 0
        _sim_step()

        return env_cfg.yellow + env_cfg.all_red + 1

    _sim_step()
    return 1


def apply_action(
    action: int,
    net: NetworkInfo,
    phase_ctrls: Dict[str, PhaseController],
) -> int:
    """根据路口数量自动选择执行方式。"""
    if len(net.tls_ids) == 1:
        return apply_action_single(action, net, phase_ctrls)
    return apply_action_cascaded(action, net, phase_ctrls)


# ======================================================================
# 指标采集 & GUI 文字
# ======================================================================

def collect_metrics(net: NetworkInfo, step: int, action: int) -> StepMetrics:
    """采集当前步的交通指标。"""
    total_wait = 0.0
    total_queue = 0.0
    num_lanes = 0
    for tls_id in net.tls_ids:
        for edge in net.edges[tls_id]:
            total_wait += traci.edge.getWaitingTime(edge)
        for lane in net.lanes[tls_id]:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
            num_lanes += 1
    throughput = _flush_arrivals()
    return StepMetrics(
        step=step,
        waiting_time=total_wait,
        avg_queue=total_queue / max(num_lanes, 1),
        throughput=throughput,
        action=action,
    )


_poi_ids: List[str] = []


def _ensure_poi(poi_id: str, x: float, y: float, text: str) -> None:
    """创建或更新一个 POI 文本标签。"""
    if poi_id not in _poi_ids:
        try:
            traci.poi.add(poi_id, x=x, y=y,
                          color=(0, 0, 0, 0), poiType=text, layer=255)
            _poi_ids.append(poi_id)
        except traci.TraCIException:
            pass
    else:
        try:
            traci.poi.setType(poi_id, text)
        except traci.TraCIException:
            pass


def update_gui_fixed(step: int, m: StepMetrics, cum_throughput: int,
                     base_x: float = 120.0, base_y: float = 180.0) -> None:
    """固定配时运行时：在 SUMO GUI 显示当前指标（2行）。"""
    _ensure_poi("line1", base_x, base_y,
                f"[ Fixed-Time ]  t={step}s")
    _ensure_poi("line2", base_x, base_y - 20,
                f"Wait={m.waiting_time:.0f}s | Queue={m.avg_queue:.1f} | Cars={cum_throughput}")


def update_gui_dqn(
    step: int,
    m_dqn: StepMetrics,
    cum_tp_dqn: int,
    baseline_fixed: Dict[int, StepMetrics] | None,
    cum_tp_fixed_map: Dict[int, int] | None,
    base_x: float = 120.0,
    base_y: float = 200.0,
) -> None:
    """DQN 运行时：在 SUMO GUI 显示 DQN 指标 + 固定配时基线 + 改善率（5行）。"""
    action_text = "SWITCH" if m_dqn.action == 1 else "KEEP"

    # 第1行：标题
    _ensure_poi("line1", base_x, base_y,
                f"[ DQN vs Fixed-Time ]  t={step}s  Action={action_text}")

    # 第2行：DQN 当前指标
    _ensure_poi("line2", base_x, base_y - 15,
                f"DQN   >> Wait={m_dqn.waiting_time:.0f}s | "
                f"Queue={m_dqn.avg_queue:.1f} | Cars={cum_tp_dqn}")

    # 第3-5行：如果有固定配时基线，显示对比
    if baseline_fixed is not None and cum_tp_fixed_map is not None:
        # 找最接近当前时间步的基线数据
        closest_step = min(baseline_fixed.keys(), key=lambda s: abs(s - step), default=None)
        if closest_step is not None:
            m_f = baseline_fixed[closest_step]
            cum_f = cum_tp_fixed_map[closest_step]

            _ensure_poi("line3", base_x, base_y - 30,
                        f"Fixed >> Wait={m_f.waiting_time:.0f}s | "
                        f"Queue={m_f.avg_queue:.1f} | Cars={cum_f}")

            # 改善率
            wait_imp = ((m_f.waiting_time - m_dqn.waiting_time)
                        / m_f.waiting_time * 100) if m_f.waiting_time > 0 else 0
            queue_imp = ((m_f.avg_queue - m_dqn.avg_queue)
                         / m_f.avg_queue * 100) if m_f.avg_queue > 0 else 0
            cars_imp = ((cum_tp_dqn - cum_f) / cum_f * 100) if cum_f > 0 else 0

            wait_arrow = "v" if wait_imp > 0 else "^"
            queue_arrow = "v" if queue_imp > 0 else "^"
            cars_arrow = "^" if cars_imp > 0 else "v"

            _ensure_poi("line4", base_x, base_y - 45,
                        f"Improve >> Wait {wait_imp:+.1f}%{wait_arrow} | "
                        f"Queue {queue_imp:+.1f}%{queue_arrow} | "
                        f"Cars {cars_imp:+.1f}%{cars_arrow}")
        else:
            _ensure_poi("line3", base_x, base_y - 30, "Fixed >> (no baseline data)")
            _ensure_poi("line4", base_x, base_y - 45, "")


# ======================================================================
# 加载 DQN 智能体
# ======================================================================

def load_dqn_agent(state_dim: int, action_dim: int, scenario: str) -> DQNAgent:
    """加载训练好的 DQN 模型。"""
    device = torch.device("cpu")
    config = AgentConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = DQNAgent(config)

    if scenario == "cascaded_intersection":
        model_path = MODEL_DIR / "cascaded_best.pt"
    else:
        model_path = MODEL_DIR / f"{scenario}_best.pt"

    if model_path.exists():
        try:
            agent.load(str(model_path), weights_only=True)
            print(f"  已加载模型: {model_path.name}")
        except Exception as e:
            print(f"  模型加载失败: {e}，使用未训练模型")
    else:
        print(f"  未找到 {model_path.name}，使用未训练模型")

    return agent


# ======================================================================
# PhaseController 创建
# ======================================================================

def make_phase_controllers(net: NetworkInfo) -> Dict[str, PhaseController]:
    """为每个路口创建 PhaseController。"""
    ctrls = {}
    for tls_id in net.tls_ids:
        ctrls[tls_id] = PhaseController(
            min_green=env_cfg.min_green,
            max_green=env_cfg.max_green,
            yellow=env_cfg.yellow,
            all_red=env_cfg.all_red,
            phases=net.phases[tls_id],
        )
    return ctrls


# ======================================================================
# 单次仿真运行
# ======================================================================

def run_fixed_time(
    scenario: str,
    duration: int,
    seed: int,
) -> List[StepMetrics]:
    """运行固定配时演示，返回逐步指标。"""
    global _pending_arrivals
    _pending_arrivals = 0
    _poi_ids.clear()

    print(f"\n{'='*60}")
    print(f"  【固定配时】 场景: {scenario} | 时长: {duration}s")
    print(f"{'='*60}")
    print("  特点: 固定周期切换，无法适应车流变化")
    print("  观察: 车少方向绿灯浪费，车多方向排队严重\n")

    input("  按 Enter 键启动固定配时演示...")

    start_sumo(scenario, seed)
    net = get_network_info()
    phase_ctrls = make_phase_controllers(net)
    controller = FixedTimeController(scenario)

    metrics: List[StepMetrics] = []
    cum_throughput = 0
    sim_time = 0

    print("  仿真运行中...\n")

    while sim_time < duration:
        if traci.simulation.getMinExpectedNumber() <= 0:
            break

        action = controller.select_action(sim_time)
        steps_consumed = apply_action(action, net, phase_ctrls)
        sim_time += steps_consumed

        m = collect_metrics(net, sim_time, action)
        cum_throughput += m.throughput
        m.throughput = cum_throughput
        metrics.append(m)

        update_gui_fixed(sim_time, m, cum_throughput)

        if sim_time % 50 < steps_consumed:
            print(f"    t={sim_time:>4d}s | "
                  f"等待={m.waiting_time:>6.0f}s | "
                  f"排队={m.avg_queue:.1f} | "
                  f"通行量={cum_throughput}")

    traci.close()
    _poi_ids.clear()

    if metrics:
        avg_wait = np.mean([m.waiting_time for m in metrics])
        print(f"\n  固定配时结束: 平均等待={avg_wait:.1f}s, 总通行量={cum_throughput}")
    print(f"{'='*60}\n")

    return metrics


def run_dqn(
    scenario: str,
    duration: int,
    seed: int,
    baseline_fixed: List[StepMetrics] | None = None,
) -> List[StepMetrics]:
    """运行 DQN 演示，返回逐步指标。

    参数:
        baseline_fixed: 固定配时的逐步指标（用于在 GUI 中对比显示）。
    """
    global _pending_arrivals
    _pending_arrivals = 0
    _poi_ids.clear()

    # 构建基线查找表（按仿真时间索引）
    fixed_by_step: Dict[int, StepMetrics] | None = None
    fixed_cum_tp: Dict[int, int] | None = None
    if baseline_fixed:
        fixed_by_step = {m.step: m for m in baseline_fixed}
        fixed_cum_tp = {m.step: m.throughput for m in baseline_fixed}

    print(f"\n{'='*60}")
    print(f"  【DQN自适应控制】 场景: {scenario} | 时长: {duration}s")
    print(f"{'='*60}")
    print("  特点: 根据实时车流动态调整信号灯")
    print("  观察: 车少时快速切换，车多时延长绿灯")
    if baseline_fixed:
        print("  GUI中将同时显示固定配时基线数据用于对比")
    print()

    input("  按 Enter 键启动 DQN 演示...")

    start_sumo(scenario, seed)
    net = get_network_info()
    is_cascaded = len(net.tls_ids) > 1
    action_dim = 2 ** len(net.tls_ids)
    phase_ctrls = make_phase_controllers(net)

    # 构造初始状态以获取维度，再加载模型
    state = get_state(net, phase_ctrls)
    state_dim = state.shape[0]
    agent = load_dqn_agent(state_dim, action_dim, scenario)

    print(f"  路口: {net.tls_ids} ({'级联' if is_cascaded else '单路口'})")
    print(f"  状态维度: {state_dim} | 动作空间: {action_dim}")

    metrics: List[StepMetrics] = []
    cum_throughput = 0
    sim_time = 0
    phase_durations: List[int] = []
    current_phase_start = 0

    print("  仿真运行中...\n")

    while sim_time < duration:
        if traci.simulation.getMinExpectedNumber() <= 0:
            break

        state = get_state(net, phase_ctrls)
        action = agent.select_action(state, exploit=True)
        steps_consumed = apply_action(action, net, phase_ctrls)
        sim_time += steps_consumed

        # 记录相位切换间隔
        if action == 1:
            dur = sim_time - current_phase_start
            if dur > 0:
                phase_durations.append(dur)
            current_phase_start = sim_time

        m = collect_metrics(net, sim_time, action)
        cum_throughput += m.throughput
        m.throughput = cum_throughput
        metrics.append(m)

        # 在 GUI 中显示 DQN 指标 + 固定配时基线对比
        update_gui_dqn(sim_time, m, cum_throughput, fixed_by_step, fixed_cum_tp)

        if sim_time % 50 < steps_consumed:
            act_str = "切换" if action == 1 else "保持"
            print(f"    t={sim_time:>4d}s | "
                  f"动作={act_str} | "
                  f"等待={m.waiting_time:>6.0f}s | "
                  f"排队={m.avg_queue:.1f} | "
                  f"通行量={cum_throughput}")

    traci.close()
    _poi_ids.clear()

    if metrics:
        avg_wait = np.mean([m.waiting_time for m in metrics])
        print(f"\n  DQN结束: 平均等待={avg_wait:.1f}s, 总通行量={cum_throughput}")
    if phase_durations:
        print(f"  相位时长: 平均={np.mean(phase_durations):.1f}s, "
              f"范围={min(phase_durations)}-{max(phase_durations)}s, "
              f"标准差={np.std(phase_durations):.1f}s (体现自适应特性)")
    print(f"{'='*60}\n")

    return metrics


# ======================================================================
# 对比图表生成
# ======================================================================

def plot_comparison(
    metrics_fixed: List[StepMetrics],
    metrics_dqn: List[StepMetrics],
    scenario: str,
    output_dir: Path,
) -> None:
    """生成时间序列对比图 + 汇总柱状图。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    df_f = pd.DataFrame([vars(m) for m in metrics_fixed])
    df_d = pd.DataFrame([vars(m) for m in metrics_dqn])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'固定配时 vs DQN 对比 — {scenario}', fontsize=14, fontweight='bold')

    # 1. 等待时间时间序列
    ax = axes[0, 0]
    ax.plot(df_f['step'], df_f['waiting_time'], label='固定配时', alpha=0.7, color='red')
    ax.plot(df_d['step'], df_d['waiting_time'], label='DQN', alpha=0.7, color='green')
    ax.set_xlabel('仿真时间 (s)')
    ax.set_ylabel('等待时间 (s)')
    ax.set_title('实时等待时间')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 排队长度时间序列
    ax = axes[0, 1]
    ax.plot(df_f['step'], df_f['avg_queue'], label='固定配时', alpha=0.7, color='red')
    ax.plot(df_d['step'], df_d['avg_queue'], label='DQN', alpha=0.7, color='green')
    ax.set_xlabel('仿真时间 (s)')
    ax.set_ylabel('平均排队长度')
    ax.set_title('实时排队长度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 累计通行量
    ax = axes[1, 0]
    ax.plot(df_f['step'], df_f['throughput'], label='固定配时', alpha=0.7, color='red')
    ax.plot(df_d['step'], df_d['throughput'], label='DQN', alpha=0.7, color='green')
    ax.set_xlabel('仿真时间 (s)')
    ax.set_ylabel('累计通行量')
    ax.set_title('累计通行量')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 汇总柱状图
    ax = axes[1, 1]
    final_f = metrics_fixed[-1] if metrics_fixed else StepMetrics(0, 0, 0, 0, 0)
    final_d = metrics_dqn[-1] if metrics_dqn else StepMetrics(0, 0, 0, 0, 0)

    avg_wait_f = np.mean([m.waiting_time for m in metrics_fixed]) if metrics_fixed else 0
    avg_wait_d = np.mean([m.waiting_time for m in metrics_dqn]) if metrics_dqn else 0
    avg_q_f = np.mean([m.avg_queue for m in metrics_fixed]) if metrics_fixed else 0
    avg_q_d = np.mean([m.avg_queue for m in metrics_dqn]) if metrics_dqn else 0

    labels = ['平均等待时间', '平均排队', '总通行量']
    vals_f = [avg_wait_f, avg_q_f, final_f.throughput]
    vals_d = [avg_wait_d, avg_q_d, final_d.throughput]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, vals_f, w, label='固定配时', color='salmon', alpha=0.8)
    bars2 = ax.bar(x + w/2, vals_d, w, label='DQN', color='mediumseagreen', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('最终指标对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar_group in (bars1, bars2):
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'comparison_{scenario}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  对比图表已保存: {path}")
    plt.show()


def print_summary(
    metrics_fixed: List[StepMetrics],
    metrics_dqn: List[StepMetrics],
    scenario: str,
) -> None:
    """打印对比摘要。"""
    if not metrics_fixed or not metrics_dqn:
        return

    avg_wait_f = np.mean([m.waiting_time for m in metrics_fixed])
    avg_wait_d = np.mean([m.waiting_time for m in metrics_dqn])
    total_cars_f = metrics_fixed[-1].throughput
    total_cars_d = metrics_dqn[-1].throughput
    avg_q_f = np.mean([m.avg_queue for m in metrics_fixed])
    avg_q_d = np.mean([m.avg_queue for m in metrics_dqn])

    wait_improve = ((avg_wait_f - avg_wait_d) / avg_wait_f * 100) if avg_wait_f > 0 else 0
    cars_improve = ((total_cars_d - total_cars_f) / total_cars_f * 100) if total_cars_f > 0 else 0

    print(f"\n{'='*60}")
    print(f"  对比结果: {scenario}")
    print(f"{'='*60}")
    print(f"  {'指标':>12s} | {'固定配时':>10s} | {'DQN':>10s} | {'改善':>8s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'平均等待':>12s} | {avg_wait_f:>10.1f} | {avg_wait_d:>10.1f} | {wait_improve:>7.1f}%")
    print(f"  {'平均排队':>12s} | {avg_q_f:>10.2f} | {avg_q_d:>10.2f} |")
    print(f"  {'总通行量':>12s} | {total_cars_f:>10d} | {total_cars_d:>10d} | {cars_improve:>7.1f}%")
    print(f"{'='*60}")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定配时 vs DQN 逐一对比演示")
    parser.add_argument("--scenario", type=str, default="t_intersection",
                        choices=["t_intersection", "x_intersection", "cascaded_intersection"],
                        help="场景名称")
    parser.add_argument("--duration", type=int, default=600, help="仿真时长（秒）")
    parser.add_argument("--seed", type=int, default=12345, help="随机种子（确保车流一致）")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["fixed", "dqn", "both"],
                        help="演示模式: fixed=仅固定配时, dqn=仅DQN, both=先后对比")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  固定配时 vs DQN 逐一对比演示")
    print(f"{'='*60}")
    print(f"  场景: {args.scenario}")
    print(f"  时长: {args.duration}s")
    print(f"  种子: {args.seed} (确保两次车流一致)")
    print(f"  模式: {args.mode}")
    print(f"{'='*60}")

    metrics_fixed: List[StepMetrics] = []
    metrics_dqn: List[StepMetrics] = []

    # 第一轮：固定配时
    if args.mode in ("fixed", "both"):
        metrics_fixed = run_fixed_time(args.scenario, args.duration, args.seed)

    # 第二轮：DQN（传入固定配时基线用于 GUI 对比显示）
    if args.mode in ("dqn", "both"):
        metrics_dqn = run_dqn(args.scenario, args.duration, args.seed,
                              baseline_fixed=metrics_fixed if metrics_fixed else None)

    # 对比总结
    if args.mode == "both" and metrics_fixed and metrics_dqn:
        print_summary(metrics_fixed, metrics_dqn, args.scenario)

        output_dir = Path("outputs") / "comparison_demo"
        plot_comparison(metrics_fixed, metrics_dqn, args.scenario, output_dir)


if __name__ == "__main__":
    main()
