"""DQN智能体与定时控制基线的对比评估脚本。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch

from agent import DQNAgent, AgentConfig
from config.settings import MODEL_DIR, evaluation as eval_cfg, get_device
from environment import SumoEnvironment
from evaluation.fixed_time_controller import FixedTimeController
from utils import MetricsRecorder

ControllerType = Literal["dqn", "fixed_time"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估交通信号控制器")
    parser.add_argument("--scenario", type=str, default="t_intersection", 
                        choices=["t_intersection", "x_intersection"],
                        help="场景名称")
    parser.add_argument("--controller", type=str, default="dqn", 
                        choices=["dqn", "fixed_time"],
                        help="控制器类型")
    parser.add_argument("--episodes", type=int, default=eval_cfg.episodes,
                        help="评估回合数")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（如果不指定，自动查找最新模型）")
    parser.add_argument("--device", type=str, default=get_device(),
                        help="计算设备")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="每回合最大步数")
    parser.add_argument("--gui", action="store_true",
                        help="使用SUMO图形界面")
    return parser.parse_args()


def evaluate_dqn(
    env: SumoEnvironment, 
    agent: DQNAgent, 
    episodes: int, 
    max_steps: int, 
    recorder: MetricsRecorder
) -> dict:
    """
    评估DQN控制器。
    
    参数:
        env: SUMO环境
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

        # 记录指标
        metrics = {
            "episode": ep,
            "reward": episode_reward,
            **result.info,
        }
        recorder.log(metrics)

        total_metrics["waiting_time"] += result.info.get("waiting_time", 0)
        total_metrics["throughput"] += result.info.get("throughput", 0)
        total_metrics["avg_queue"] += result.info.get("avg_queue", 0)
        total_metrics["reward"] += episode_reward

        print(f"回合 {ep + 1}/{episodes} | 奖励: {episode_reward:.2f} | "
              f"等待时间: {result.info.get('waiting_time', 0):.1f} | "
              f"通行量: {result.info.get('throughput', 0)}")

    # 计算平均值
    for key in total_metrics:
        total_metrics[key] /= episodes

    return total_metrics


def evaluate_fixed_time(
    env: SumoEnvironment, 
    controller: FixedTimeController, 
    episodes: int, 
    max_steps: int, 
    recorder: MetricsRecorder
) -> dict:
    """
    评估定时控制器。
    
    参数:
        env: SUMO环境
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

    # 创建环境
    env = SumoEnvironment(args.scenario, max_steps=args.max_steps, use_gui=args.gui if hasattr(args, 'gui') else False)
    
    # 初始化以获取维度
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = 2  # 二元动作

    print(f"评估 {args.controller} 控制器，场景: {args.scenario}")
    print(f"  - 状态维度: {state_dim}, 动作维度: {action_dim}")

    recorder = MetricsRecorder(Path("outputs") / args.scenario / args.controller)

    if args.controller == "dqn":
        # 如果没有指定模型，按场景名查找最佳模型
        if args.model is None:
            best_path = MODEL_DIR / f"{args.scenario}_best.pt"
            if best_path.exists():
                args.model = str(best_path)
            else:
                # 回退：查找场景特定的检查点
                checkpoints = list(MODEL_DIR.glob(f"{args.scenario}_ep*.pt"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('ep')[1]))
                    args.model = str(latest_checkpoint)
                else:
                    args.model = str(best_path)  # 即使不存在也给出路径，后续会报错提示
            print(f"  - 使用模型: {args.model}")
        
        # 创建智能体并加载模型
        agent_config = AgentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        agent = DQNAgent(agent_config)
        
        # 尝试加载模型，如果失败则警告
        try:
            agent.load(args.model)
        except RuntimeError as e:
            print(f"  ⚠️  警告: 无法加载模型 {args.model}")
            print(f"  错误: {e}")
            print(f"  将使用未训练的模型进行评估")
        
        metrics = evaluate_dqn(env, agent, args.episodes, args.max_steps, recorder)
    else:
        controller = FixedTimeController(args.scenario)
        metrics = evaluate_fixed_time(env, controller, args.episodes, args.max_steps, recorder)

    recorder.flush(f"{args.controller}_metrics.csv")
    env.close()

    print("\n" + "=" * 50)
    print("评估结果（平均值）:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
