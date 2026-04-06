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

# 添加numpy安全全局变量以支持PyTorch 2.6加载旧模型
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
except:
    pass

from agent.dqn_agent import DQNAgent, AgentConfig
from config.settings import (
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
    departed: int
    unfinished: int
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


def start_sumo(scenario: str, seed: int, delay: int = 100) -> None:
    """启动单个 SUMO GUI 实例。
    
    参数:
        scenario: 场景名称
        seed: 随机种子
        delay: GUI延迟（毫秒），默认100ms。设为0则最快速度运行。
    """
    sumo_cfg = str(SCENARIO_DIR / scenario / "simulation.sumocfg")
    binary = _sumo_binary()
    cmd = [
        binary, "-c", sumo_cfg,
        "--start",
        "--delay", str(delay),
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
_pending_departed: int = 0


def _sim_step() -> None:
    """执行一步仿真并累计到达/发车车辆数。"""
    global _pending_arrivals, _pending_departed
    traci.simulationStep()
    _pending_arrivals += traci.simulation.getArrivedNumber()
    _pending_departed += traci.simulation.getDepartedNumber()


def _flush_arrivals() -> int:
    """读取并清零累计到达车辆数。"""
    global _pending_arrivals
    count = _pending_arrivals
    _pending_arrivals = 0
    return count


def _flush_departed() -> int:
    """读取并清零累计发车车辆数。"""
    global _pending_departed
    count = _pending_departed
    _pending_departed = 0
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

def collect_metrics(
    net: NetworkInfo,
    step: int,
    action: int,
    cum_arrived: int,
    cum_departed: int,
) -> StepMetrics:
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

    unfinished = max(cum_departed - cum_arrived, 0)
    return StepMetrics(
        step=step,
        waiting_time=total_wait,
        avg_queue=total_queue / max(num_lanes, 1),
        throughput=cum_arrived,
        departed=cum_departed,
        unfinished=unfinished,
        action=action,
    )


_poi_ids: List[str] = []
_polygon_ids: List[str] = []


def _ensure_poi(poi_id: str, x: float, y: float, text: str, color: Tuple[int, int, int, int] = (0, 0, 0, 255)) -> None:
    """创建或更新一个 POI 文本标签。
    
    参数:
        color: RGBA颜色，默认黑色 (0, 0, 0, 255)
    """
    if poi_id not in _poi_ids:
        try:
            traci.poi.add(poi_id, x=x, y=y,
                          color=color, poiType=text, layer=255)
            _poi_ids.append(poi_id)
        except traci.TraCIException:
            pass
    else:
        try:
            traci.poi.setType(poi_id, text)
            traci.poi.setColor(poi_id, color)
        except traci.TraCIException:
            pass


def _add_text_box(box_id: str, x: float, y: float, width: float, height: float,
                  text: str, bg_color: Tuple[int, int, int, int] = (255, 255, 255, 230)) -> None:
    """创建一个带背景的文本框。"""
    # 创建背景矩形
    if box_id not in _polygon_ids:
        try:
            shape = [
                (x, y),
                (x + width, y),
                (x + width, y - height),
                (x, y - height)
            ]
            traci.polygon.add(box_id, shape, bg_color, fill=True, layer=200)
            _polygon_ids.append(box_id)
        except traci.TraCIException:
            pass
    
    # 在矩形中心添加文字
    text_x = x + width / 2
    text_y = y - height / 2
    _ensure_poi(f"{box_id}_text", text_x, text_y, text, (0, 0, 0, 255))


def update_gui_fixed(step: int, m: StepMetrics,
                     base_x: float = 120.0, base_y: float = 180.0) -> None:
    """固定配时运行时：在 SUMO GUI 显示当前指标（2行）。"""
    _ensure_poi("line1", base_x, base_y,
                f"[ Fixed-Time ]  t={step}s")
    _ensure_poi("line2", base_x, base_y - 20,
                f"Wait={m.waiting_time:.0f}s | Queue={m.avg_queue:.1f} | Arr={m.throughput}")
    _ensure_poi("line3", base_x, base_y - 40,
                f"Dep={m.departed} | Unfinished={m.unfinished}")


def update_gui_dqn(
    step: int,
    m_dqn: StepMetrics,
    baseline_fixed: Dict[int, StepMetrics] | None,
    base_x: float = 120.0,
    base_y: float = 200.0,
) -> None:
    """DQN 运行时：在 SUMO GUI 显示 DQN 指标 + 固定配时基线。"""
    action_text = "SWITCH" if m_dqn.action == 1 else "KEEP"

    # 第1行：标题
    _ensure_poi("line1", base_x, base_y,
                f"[ DQN vs Fixed-Time ]  t={step}s  Action={action_text}")

    # 第2行：DQN 当前指标
    _ensure_poi("line2", base_x, base_y - 15,
                f"DQN   >> Wait={m_dqn.waiting_time:.0f}s | "
                f"Queue={m_dqn.avg_queue:.1f} | Arr={m_dqn.throughput}")
    _ensure_poi("line3", base_x, base_y - 30,
                f"DQN   >> Dep={m_dqn.departed} | Unfinished={m_dqn.unfinished}")

    # 第4行：如果有固定配时基线，显示对比
    if baseline_fixed is not None:
        closest_step = min(baseline_fixed.keys(), key=lambda s: abs(s - step), default=None)
        if closest_step is not None:
            m_f = baseline_fixed[closest_step]
            _ensure_poi("line4", base_x, base_y - 45,
                        f"Fixed >> Wait={m_f.waiting_time:.0f}s | Queue={m_f.avg_queue:.1f} | Arr={m_f.throughput}")
            _ensure_poi("line5", base_x, base_y - 60,
                        f"Fixed >> Dep={m_f.departed} | Unfinished={m_f.unfinished}")
        else:
            _ensure_poi("line4", base_x, base_y - 45, "Fixed >> (no baseline data)")


def show_final_results_on_map(net: NetworkInfo, metrics: List[StepMetrics], title: str, lane_stats: Dict[str, List[float]]) -> None:
    """在仿真结束后，将最终结果直接打在 SUMO 地图上（英文显示，大字体）。"""
    if not metrics:
        return

    # 计算全局最终指标
    avg_wait = np.mean([m.waiting_time for m in metrics])
    avg_q = np.mean([m.avg_queue for m in metrics])
    total_cars = metrics[-1].throughput
    total_departed = metrics[-1].departed
    unfinished = metrics[-1].unfinished

    # 动态计算面板位置
    boundary = traci.simulation.getNetBoundary()
    xmin, ymin = boundary[0]
    xmax, ymax = boundary[1]

    # 面板尺寸
    box_width = 50
    box_height = 58

    # 面板放在左上象限边缘（不遮挡中心）
    box_x = xmin + 100
    box_y = ymax - 40

    # 白色背景 + 黑色文字
    if "固定" in title or "FIXED" in title.upper():
        title_text = "FIXED"
    else:
        title_text = "DQN"
    
    bg_color = (255, 255, 255, 250)  # 白色背景
    text_color = (0, 0, 0, 255)  # 黑色文字
    
    # 创建白色背景框
    _add_text_box("result_box", box_x, box_y, box_width, box_height, "", bg_color)
    
    # 面板文字坐标
    text_x = box_x + 15
    text_y = box_y - 15
    line_height = 9.0

    # 显示结果（原版6行）
    _ensure_poi("final_title", text_x, text_y, f"== {title_text} ==", text_color)
    _ensure_poi("final_wait", text_x, text_y - line_height, f"Wait: {avg_wait:.0f}s", text_color)
    _ensure_poi("final_queue", text_x, text_y - line_height*2, f"Queue: {avg_q:.1f}", text_color)
    _ensure_poi("final_cars", text_x, text_y - line_height*3, f"Arrived: {total_cars}", text_color)
    _ensure_poi("final_dep", text_x, text_y - line_height*4, f"Departed: {total_departed}", text_color)
    _ensure_poi("final_unf", text_x, text_y - line_height*5, f"Unfinished: {unfinished}", text_color)
    
    # 注释掉车道数据显示，避免画面杂乱
    # 如果需要显示车道数据，可以取消下面的注释
    # for tls_id in net.tls_ids:
    #     for lane in net.lanes[tls_id]:
    #         avg_lane_q = np.mean(lane_stats[lane]) if lane in lane_stats else 0.0
    #         shape = traci.lane.getShape(lane)
    #         if len(shape) >= 2:
    #             p1, p2 = shape[-2], shape[-1]
    #             dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    #             dist = math.hypot(dx, dy)
    #             offset = min(15.0, dist)
    #             x = p2[0] + (dx / dist) * offset
    #             y = p2[1] + (dy / dist) * offset
    #             text = f"Avg Q: {avg_lane_q:.1f}"
    #             _ensure_poi(f"final_lane_{lane}", x, y, text)


def clear_all_pois():
    """清理运行过程中产生的动态文本，让最终截图更干净"""
    for poi_id in list(_poi_ids):
        try:
            traci.poi.remove(poi_id)
        except traci.TraCIException:
            pass
    _poi_ids.clear()
    
    for poly_id in list(_polygon_ids):
        try:
            traci.polygon.remove(poly_id)
        except traci.TraCIException:
            pass
    _polygon_ids.clear()


# ======================================================================
# 加载 DQN 智能体
# ======================================================================

def load_dqn_agent(
    state_dim: int,
    action_dim: int,
    scenario: str,
    strict_model: bool = False,
) -> DQNAgent:
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
            # 直接使用 weights_only=False 加载旧模型（兼容性最好）
            agent.load(str(model_path), weights_only=False)
            print(f"  已加载模型: {model_path.name}")
        except Exception as e:
            if strict_model:
                raise RuntimeError(f"严格模式下模型加载失败: {model_path} | {e}") from e
            print(f"  模型加载失败: {e}，使用未训练模型（仅调试用途）")
    else:
        if strict_model:
            raise FileNotFoundError(f"严格模式下未找到模型: {model_path}")
        print(f"  未找到 {model_path.name}，使用未训练模型（仅调试用途）")

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
    delay: int = 100,
) -> List[StepMetrics]:
    """运行固定配时演示，返回逐步指标。
    
    参数:
        delay: SUMO GUI延迟（毫秒）
    """
    global _pending_arrivals, _pending_departed
    _pending_arrivals = 0
    _pending_departed = 0
    _poi_ids.clear()

    print(f"\n{'='*60}")
    print(f"  【固定配时】 场景: {scenario} | 时长: {duration}s")
    print(f"{'='*60}")
    print("  特点: 固定周期切换，无法适应车流变化")
    print("  观察: 车少方向绿灯浪费，车多方向排队严重\n")

    input("  按 Enter 键启动固定配时演示...")

    start_sumo(scenario, seed, delay)
    net = get_network_info()
    phase_ctrls = make_phase_controllers(net)
    controller = FixedTimeController(scenario)

    metrics: List[StepMetrics] = []
    cum_throughput = 0
    cum_departed = 0
    sim_time = 0

    print("  仿真运行中...\n")

    # 【新增】用于记录每条车道的排队历史
    lane_stats = {lane: [] for tls_id in net.tls_ids for lane in net.lanes[tls_id]}

    while sim_time < duration:
        if traci.simulation.getMinExpectedNumber() <= 0:
            break

        if len(net.tls_ids) == 1:
            tls_id = net.tls_ids[0]
            ctrl = phase_ctrls[tls_id]
            action = controller.select_action(ctrl.current_phase, ctrl.elapsed)
        else:
            action = controller.joint_action_cascaded(phase_ctrls, list(net.tls_ids))
        steps_consumed = apply_action(action, net, phase_ctrls)
        sim_time += steps_consumed

        cum_throughput += _flush_arrivals()
        cum_departed += _flush_departed()
        m = collect_metrics(net, sim_time, action, cum_throughput, cum_departed)
        metrics.append(m)

        # 【新增】记录当前步每条车道的排队长度
        for tls_id in net.tls_ids:
            for lane in net.lanes[tls_id]:
                lane_stats[lane].append(traci.lane.getLastStepHaltingNumber(lane))

        # update_gui_fixed(sim_time, m)  # 已禁用实时显示，避免遮挡截图

        if sim_time % 50 < steps_consumed:
            print(f"    t={sim_time:>4d}s | "
                  f"等待={m.waiting_time:>6.0f}s | "
                  f"排队={m.avg_queue:.1f} | "
                  f"到达={m.throughput} | 发车={m.departed} | 未完成={m.unfinished}")

    # ================= 【新增/修改结尾部分】 =================
    clear_all_pois()  # 清理动态文字
    show_final_results_on_map(net, metrics, "固定配时", lane_stats)  # 渲染最终结果

    print("\n  [固定配时] 仿真已完成！请在 SUMO 窗口中查看结果并截图。")
    input("  截图完成后，请按 Enter 键关闭当前窗口继续...")

    traci.close()
    _poi_ids.clear()
    # =========================================================

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
    delay: int = 100,
    strict_model: bool = False,
) -> List[StepMetrics]:
    """运行 DQN 演示，返回逐步指标。

    参数:
        baseline_fixed: 固定配时的逐步指标（用于在 GUI 中对比显示）。
        delay: SUMO GUI延迟（毫秒）
    """
    global _pending_arrivals, _pending_departed
    _pending_arrivals = 0
    _pending_departed = 0
    _poi_ids.clear()

    # 构建基线查找表（按仿真时间索引）
    fixed_by_step: Dict[int, StepMetrics] | None = None
    if baseline_fixed:
        fixed_by_step = {m.step: m for m in baseline_fixed}

    print(f"\n{'='*60}")
    print(f"  【DQN自适应控制】 场景: {scenario} | 时长: {duration}s")
    print(f"{'='*60}")
    print("  特点: 根据实时车流动态调整信号灯")
    print("  观察: 车少时快速切换，车多时延长绿灯")
    if baseline_fixed:
        print("  GUI中将同时显示固定配时基线数据用于对比")
    print()

    input("  按 Enter 键启动 DQN 演示...")

    start_sumo(scenario, seed, delay)
    net = get_network_info()
    is_cascaded = len(net.tls_ids) > 1
    action_dim = 2 ** len(net.tls_ids)
    phase_ctrls = make_phase_controllers(net)

    # 构造初始状态以获取维度，再加载模型
    state = get_state(net, phase_ctrls)
    state_dim = state.shape[0]
    agent = load_dqn_agent(state_dim, action_dim, scenario, strict_model=strict_model)

    print(f"  路口: {net.tls_ids} ({'级联' if is_cascaded else '单路口'})")
    print(f"  状态维度: {state_dim} | 动作空间: {action_dim}")

    metrics: List[StepMetrics] = []
    cum_throughput = 0
    cum_departed = 0
    sim_time = 0
    phase_durations: List[int] = []
    current_phase_start = 0

    print("  仿真运行中...\n")

    # 【新增】记录每条车道的排队历史
    lane_stats = {lane: [] for tls_id in net.tls_ids for lane in net.lanes[tls_id]}

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

        cum_throughput += _flush_arrivals()
        cum_departed += _flush_departed()
        m = collect_metrics(net, sim_time, action, cum_throughput, cum_departed)
        metrics.append(m)

        # 【新增】记录当前步每条车道的排队长度
        for tls_id in net.tls_ids:
            for lane in net.lanes[tls_id]:
                lane_stats[lane].append(traci.lane.getLastStepHaltingNumber(lane))

        # 在 GUI 中显示 DQN 指标 + 固定配时基线对比
        # update_gui_dqn(sim_time, m, fixed_by_step)  # 已禁用实时显示，避免遮挡截图

        if sim_time % 50 < steps_consumed:
            act_str = "切换" if action == 1 else "保持"
            print(f"    t={sim_time:>4d}s | "
                  f"动作={act_str} | "
                  f"等待={m.waiting_time:>6.0f}s | "
                  f"排队={m.avg_queue:.1f} | "
                  f"到达={m.throughput} | 发车={m.departed} | 未完成={m.unfinished}")

    # ================= 【新增/修改结尾部分】 =================
    clear_all_pois()
    show_final_results_on_map(net, metrics, "DQN 智能控制", lane_stats)

    print("\n  [DQN 智能控制] 仿真已完成！请在 SUMO 窗口中查看结果并截图。")
    input("  截图完成后，请按 Enter 键关闭当前窗口继续...")

    traci.close()
    _poi_ids.clear()
    # =========================================================

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
    """生成时间序列对比图 + 汇总柱状图（彩色显示，紧凑布局）。"""
    # 设置紧凑参数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['lines.linewidth'] = 1.8

    df_f = pd.DataFrame([vars(m) for m in metrics_fixed])
    df_d = pd.DataFrame([vars(m) for m in metrics_dqn])

    # 紧凑尺寸
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    color_fixed = '#1f77b4'   # 蓝色
    color_dqn = '#ff7f0e'     # 橙色

    # 1. 等待时间时间序列
    ax = axes[0, 0]
    ax.plot(df_f['step'], df_f['waiting_time'], label='固定配时',
            linewidth=2.2, color=color_fixed, linestyle='-', marker='o',
            markevery=max(1, len(df_f)//8), markersize=5,
            markerfacecolor='white', markeredgewidth=1.2,
            markeredgecolor=color_fixed)
    ax.plot(df_d['step'], df_d['waiting_time'], label='DQN',
            linewidth=2.0, color=color_dqn, linestyle='--', marker='s',
            markevery=max(1, len(df_d)//8), markersize=4,
            markerfacecolor=color_dqn, markeredgecolor=color_dqn)
    ax.set_xlabel('仿真时间 (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('等待时间 (s)', fontsize=9, fontweight='bold')
    ax.set_title('实时等待时间', fontsize=10, fontweight='bold', pad=5)
    ax.legend(loc='best', frameon=True, shadow=False, fontsize=8,
              edgecolor='black', fancybox=False)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='gray')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # 2. 排队长度时间序列
    ax = axes[0, 1]
    ax.plot(df_f['step'], df_f['avg_queue'], label='固定配时',
            linewidth=2.2, color=color_fixed, linestyle='-', marker='o',
            markevery=max(1, len(df_f)//8), markersize=5,
            markerfacecolor='white', markeredgewidth=1.2,
            markeredgecolor=color_fixed)
    ax.plot(df_d['step'], df_d['avg_queue'], label='DQN',
            linewidth=2.0, color=color_dqn, linestyle='--', marker='s',
            markevery=max(1, len(df_d)//8), markersize=4,
            markerfacecolor=color_dqn, markeredgecolor=color_dqn)
    ax.set_xlabel('仿真时间 (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('平均排队长度', fontsize=9, fontweight='bold')
    ax.set_title('实时排队长度', fontsize=10, fontweight='bold', pad=5)
    ax.legend(loc='best', frameon=True, shadow=False, fontsize=8,
              edgecolor='black', fancybox=False)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='gray')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # 3. 累计通行量
    ax = axes[1, 0]
    ax.plot(df_f['step'], df_f['throughput'], label='固定配时',
            linewidth=2.2, color=color_fixed, linestyle='-', marker='o',
            markevery=max(1, len(df_f)//8), markersize=5,
            markerfacecolor='white', markeredgewidth=1.2,
            markeredgecolor=color_fixed)
    ax.plot(df_d['step'], df_d['throughput'], label='DQN',
            linewidth=2.0, color=color_dqn, linestyle='--', marker='s',
            markevery=max(1, len(df_d)//8), markersize=4,
            markerfacecolor=color_dqn, markeredgecolor=color_dqn)
    ax.set_xlabel('仿真时间 (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('累计通行量', fontsize=9, fontweight='bold')
    ax.set_title('累计通行量', fontsize=10, fontweight='bold', pad=5)
    ax.legend(loc='best', frameon=True, shadow=False, fontsize=8,
              edgecolor='black', fancybox=False)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='gray')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # 4. 汇总柱状图
    ax = axes[1, 1]
    final_f = metrics_fixed[-1] if metrics_fixed else StepMetrics(0, 0, 0, 0, 0, 0, 0)
    final_d = metrics_dqn[-1] if metrics_dqn else StepMetrics(0, 0, 0, 0, 0, 0, 0)

    avg_wait_f = np.mean([m.waiting_time for m in metrics_fixed]) if metrics_fixed else 0
    avg_wait_d = np.mean([m.waiting_time for m in metrics_dqn]) if metrics_dqn else 0
    avg_q_f = np.mean([m.avg_queue for m in metrics_fixed]) if metrics_fixed else 0
    avg_q_d = np.mean([m.avg_queue for m in metrics_dqn]) if metrics_dqn else 0

    labels = ['平均等待', '平均排队', '总到达量', '未完成车']
    vals_f = [avg_wait_f, avg_q_f, final_f.throughput, final_f.unfinished]
    vals_d = [avg_wait_d, avg_q_d, final_d.throughput, final_d.unfinished]

    x = np.arange(len(labels))
    w = 0.32
    bars1 = ax.bar(x - w/2, vals_f, w, label='固定配时',
                   color='#6baed6', edgecolor='#2b6cb0', linewidth=1.5)
    bars2 = ax.bar(x + w/2, vals_d, w, label='DQN',
                   color='#fdae6b', edgecolor='#c05621', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, fontweight='bold')
    ax.set_title('最终指标对比', fontsize=10, fontweight='bold', pad=5)
    ax.legend(loc='best', frameon=True, shadow=False, fontsize=8,
              edgecolor='black', fancybox=False)
    ax.grid(True, alpha=0.25, axis='y', linestyle=':', linewidth=0.8, color='gray')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 数值标注
    for bar_group in (bars1, bars2):
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='black')

    plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存多种格式（适合论文使用）
    path_png = output_dir / f'comparison_{scenario}.png'
    path_pdf = output_dir / f'comparison_{scenario}.pdf'
    path_svg = output_dir / f'comparison_{scenario}.svg'
    path_eps = output_dir / f'comparison_{scenario}.eps'
    
    plt.savefig(path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(path_pdf, bbox_inches='tight', facecolor='white')
    plt.savefig(path_svg, bbox_inches='tight', facecolor='white')
    plt.savefig(path_eps, bbox_inches='tight', facecolor='white')
    
    print(f"\n  对比图表已保存（多种格式）:")
    print(f"    PNG (300 DPI): {path_png}")
    print(f"    PDF (矢量图): {path_pdf}")
    print(f"    SVG (矢量图): {path_svg}")
    print(f"    EPS (矢量图): {path_eps}")
    print(f"  提示: Word 可直接插入 PDF/SVG/EPS，或使用 Inkscape 转换为 EMF")
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
    departed_f = metrics_fixed[-1].departed
    departed_d = metrics_dqn[-1].departed
    unfinished_f = metrics_fixed[-1].unfinished
    unfinished_d = metrics_dqn[-1].unfinished

    print(f"  {'指标':>12s} | {'固定配时':>10s} | {'DQN':>10s} | {'改善':>8s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'平均等待':>12s} | {avg_wait_f:>10.1f} | {avg_wait_d:>10.1f} | {wait_improve:>7.1f}%")
    print(f"  {'平均排队':>12s} | {avg_q_f:>10.2f} | {avg_q_d:>10.2f} |")
    print(f"  {'总到达量':>12s} | {total_cars_f:>10d} | {total_cars_d:>10d} | {cars_improve:>7.1f}%")
    print(f"  {'总发车量':>12s} | {departed_f:>10d} | {departed_d:>10d} |")
    print(f"  {'未完成车':>12s} | {unfinished_f:>10d} | {unfinished_d:>10d} |")
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
    parser.add_argument("--delay", type=int, default=100,
                        help="SUMO GUI延迟（毫秒），默认100ms。设为0则最快速度运行")
    parser.add_argument("--strict-model", action="store_true",
                        help="严格模型加载：DQN模型缺失/损坏即报错退出")
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
        metrics_fixed = run_fixed_time(args.scenario, args.duration, args.seed, args.delay)

    # 第二轮：DQN（传入固定配时基线用于 GUI 对比显示）
    if args.mode in ("dqn", "both"):
        metrics_dqn = run_dqn(args.scenario, args.duration, args.seed,
                              baseline_fixed=metrics_fixed if metrics_fixed else None,
                              delay=args.delay,
                              strict_model=args.strict_model)

    # 对比总结
    if args.mode == "both" and metrics_fixed and metrics_dqn:
        print_summary(metrics_fixed, metrics_dqn, args.scenario)

        output_dir = Path("outputs") / "comparison_demo"
        plot_comparison(metrics_fixed, metrics_dqn, args.scenario, output_dir)


if __name__ == "__main__":
    main()
