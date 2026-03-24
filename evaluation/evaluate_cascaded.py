"""级联多路口控制器评估脚本。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch

from agent.dqn_agent import DQNAgent, AgentConfig
from config.settings import MODEL_DIR, FIXED_TIME_PLANS, evaluation as eval_cfg, environment as env_cfg, get_device
from environment.cascaded_env import CascadedSumoEnvironment
from utils import MetricsRecorder

ControllerType = Literal["dqn", "fixed_time", "independent"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估级联交通控制器")
    parser.add_argument("--scenario", type=str, default="cascaded_intersection",
                        help="场景名称")
    parser.add_argument("--controller", type=str, default="dqn",
                        choices=["dqn", "fixed_time", "independent"],
                        help="控制器类型")
    parser.add_argument("--episodes", type=int, default=eval_cfg.episodes,
                        help="评估回合数")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "cascaded_best.pt"),
                        help="模型路径")
    parser.add_argument("--device", type=str, default=get_device(),
                        help="计算设备")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="每回合最大步数")
    parser.add_argument("--gui", action="store_true",
                        help="使用SUMO图形界面")
    return parser.parse_args()


class CascadedFixedTimeController:
    """多路口定时控制器。"""

    def __init__(self, scenario: str, num_junctions: int) -> None:
        """
        初始化定时控制器。
        
        参数:
            scenario: 场景名称
            num_junctions: 路口数量
        """
        if scenario not in FIXED_TIME_PLANS:
            raise ValueError(f"场景'{scenario}'没有定义定时方案")
        self.plan = FIXED_TIME_PLANS[scenario]
        self.num_phases = len(self.plan)
        self.num_junctions = num_junctions
        self.cycle = sum(self.plan) + self.num_phases * (env_cfg.yellow + env_cfg.all_red)

    def select_action(self, time_step: int) -> int:
        """
        选择动作（所有路口同步）。
        
        参数:
            time_step: 当前时间步
            
        返回:
            动作编码（0=保持）
        """
        t = time_step % self.cycle
        elapsed = 0
        current_phase = 0

        for phase_idx, green in enumerate(self.plan):
            phase_duration = green + env_cfg.yellow + env_cfg.all_red
            if t < elapsed + green:
                current_phase = phase_idx
                break
            elapsed += phase_duration

        # 简化处理：大部分时间返回0（保持）
        return 0


def evaluate_dqn(
    env: CascadedSumoEnvironment,
    agent: DQNAgent,
    episodes: int,
    max_steps: int,
    recorder: MetricsRecorder,
) -> dict:
    """
    评估DQN控制器。
    
    参数:
        env: 级联环境
        agent: DQN智能体
        episodes: 评估回合数
        max_steps: 每回合最大步数
        recorder: 指标记录器
        
    返回:
        平均指标字典
    """
    total_metrics = {
        "waiting_time": 0.0,
        "throughput": 0,
        "avg_queue": 0.0,
        "link_queue": 0.0,
        "reward": 0.0,
    }

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state, exploit=True)
            result = env.step(action)
            episode_reward += result.reward
            state = result.state

            if result.done:
                break

        # 记录最终指标
        metrics = {
            "episode": ep,
            "reward": episode_reward,
            **result.info,
        }
        recorder.log(metrics)

        total_metrics["waiting_time"] += result.info.get("waiting_time", 0)
        total_metrics["throughput"] += result.info.get("throughput", 0)
        total_metrics["avg_queue"] += result.info.get("avg_queue", 0)
        total_metrics["link_queue"] += result.info.get("link_queue", 0)
        total_metrics["reward"] += episode_reward

        print(f"回合 {ep + 1}/{episodes} | 奖励: {episode_reward:.2f} | "
              f"等待时间: {result.info.get('waiting_time', 0):.1f} | "
              f"通行量: {result.info.get('throughput', 0)}")

    # 计算平均值
    for key in total_metrics:
        total_metrics[key] /= episodes

    return total_metrics


def evaluate_fixed_time(
    env: CascadedSumoEnvironment,
    controller: CascadedFixedTimeController,
    episodes: int,
    max_steps: int,
    recorder: MetricsRecorder,
) -> dict:
    """
    评估定时控制器。
    
    参数:
        env: 级联环境
        controller: 定时控制器
        episodes: 评估回合数
        max_steps: 每回合最大步数
        recorder: 指标记录器
        
    返回:
        平均指标字典
    """
    total_metrics = {
        "waiting_time": 0.0,
        "throughput": 0,
        "avg_queue": 0.0,
        "link_queue": 0.0,
        "reward": 0.0,
    }

    for ep in range(episodes):
        env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            action = controller.select_action(step)
            result = env.step(action)
            episode_reward += result.reward

            if result.done:
                break

        metrics = {
            "episode": ep,
            "reward": episode_reward,
            **result.info,
        }
        recorder.log(metrics)

        total_metrics["waiting_time"] += result.info.get("waiting_time", 0)
        total_metrics["throughput"] += result.info.get("throughput", 0)
        total_metrics["avg_queue"] += result.info.get("avg_queue", 0)
        total_metrics["link_queue"] += result.info.get("link_queue", 0)
        total_metrics["reward"] += episode_reward

        print(f"回合 {ep + 1}/{episodes} | 奖励: {episode_reward:.2f} | "
              f"等待时间: {result.info.get('waiting_time', 0):.1f} | "
              f"通行量: {result.info.get('throughput', 0)}")

    for key in total_metrics:
        total_metrics[key] /= episodes

    return total_metrics


def main() -> None:
    """主评估函数。"""
    args = parse_args()
    device = torch.device(args.device)

    env = CascadedSumoEnvironment(
        scenario=args.scenario,
        max_steps=args.max_steps,
        use_gui=args.gui,
    )

    # 初始化以获取维度
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.action_dim

    print(f"评估 {args.controller} 控制器，场景: {args.scenario}")
    print(f"  - 路口: {env.junction_ids}")
    print(f"  - 状态维度: {state_dim}, 动作维度: {action_dim}")

    recorder = MetricsRecorder(Path("outputs") / args.scenario / args.controller)

    if args.controller == "dqn":
        agent_config = AgentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        agent = DQNAgent(agent_config)
        agent.load(args.model)
        metrics = evaluate_dqn(env, agent, args.episodes, args.max_steps, recorder)
    else:
        controller = CascadedFixedTimeController(args.scenario, env.num_junctions)
        metrics = evaluate_fixed_time(env, controller, args.episodes, args.max_steps, recorder)

    recorder.flush(f"{args.controller}_cascaded_metrics.csv")
    env.close()

    print("\n" + "=" * 50)
    print("评估结果（平均值）:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
