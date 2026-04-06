"""级联多路口控制器评估脚本。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch

from agent.dqn_agent import DQNAgent, AgentConfig
from config.settings import MODEL_DIR, evaluation as eval_cfg, get_device
from environment.cascaded_env import CascadedStepResult, CascadedSumoEnvironment
from environment.phase_logic import PhaseController
from evaluation.fixed_time_controller import FixedTimeController
from utils import MetricsRecorder

ControllerType = Literal["dqn", "fixed_time"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估级联交通控制器")
    parser.add_argument("--scenario", type=str, default="cascaded_intersection",
                        help="场景名称")
    parser.add_argument("--controller", type=str, default="dqn",
                        choices=["dqn", "fixed_time"],
                        help="控制器类型")
    parser.add_argument("--episodes", type=int, default=eval_cfg.episodes,
                        help="评估回合数")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "cascaded_best.pt"),
                        help="模型路径")
    parser.add_argument("--device", type=str, default=get_device(),
                        help="计算设备")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="每回合最大步数（须 >= 1）")
    parser.add_argument("--strict-model", action="store_true",
                        help="严格模型加载：加载失败即报错退出（推荐正式对比）")
    parser.add_argument("--gui", action="store_true",
                        help="使用SUMO图形界面")
    parser.add_argument("--seed", type=int, default=42,
                        help="SUMO 仿真随机种子（默认42，确保可复现与公平对比）")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps 须为 >= 1 的整数")
    return args


class CascadedFixedTimeController:
    """多路口定时控制器（各路口按计划绿灯时长独立决策，联合编码）。"""

    def __init__(self, scenario: str, num_junctions: int) -> None:
        """
        初始化定时控制器。

        参数:
            scenario: 场景名称
            num_junctions: 路口数量（用于校验）
        """
        self._inner = FixedTimeController(scenario)
        self.num_junctions = num_junctions

    def select_action(self, env: CascadedSumoEnvironment) -> int:
        """根据各路口 PhaseController 状态生成联合动作。"""
        if env.num_junctions != self.num_junctions:
            raise ValueError("路口数量与控制器初始化不一致")
        ctrls: dict[str, PhaseController] = {}
        for jid in env.junction_ids:
            pc = env.junctions[jid].phase_controller
            if pc is None:
                raise RuntimeError(f"路口 {jid} 未初始化相位控制器")
            ctrls[jid] = pc
        return self._inner.joint_action_cascaded(ctrls, env.junction_ids)


def evaluate_dqn(
    env: CascadedSumoEnvironment,
    agent: DQNAgent,
    episodes: int,
    max_steps: int,
    recorder: MetricsRecorder,
    seed_start: int,
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
        "departed": 0,
        "unfinished": 0,
        "reward": 0.0,
    }

    for ep in range(episodes):
        env.seed = seed_start + ep
        state = env.reset()
        episode_reward = 0.0
        result: CascadedStepResult | None = None

        for step in range(max_steps):
            action = agent.select_action(state, exploit=True)
            result = env.step(action)
            episode_reward += result.reward
            state = result.state

            if result.done:
                break

        if result is None:
            raise RuntimeError("max_steps 为 0 或未执行任何仿真步，无法记录指标")

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
        total_metrics["departed"] += result.info.get("departed", 0)
        total_metrics["unfinished"] += result.info.get("unfinished", 0)
        total_metrics["reward"] += episode_reward

        print(f"回合 {ep + 1}/{episodes} | seed={seed_start + ep} | 奖励: {episode_reward:.2f} | "
              f"等待时间: {result.info.get('waiting_time', 0):.1f} | "
              f"到达: {result.info.get('throughput', 0)} | "
              f"发车: {result.info.get('departed', 0)} | "
              f"未完成: {result.info.get('unfinished', 0)}")

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
    seed_start: int,
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
        "departed": 0,
        "unfinished": 0,
        "reward": 0.0,
    }

    for ep in range(episodes):
        env.seed = seed_start + ep
        env.reset()
        episode_reward = 0.0
        result: CascadedStepResult | None = None

        for step in range(max_steps):
            action = controller.select_action(env)
            result = env.step(action)
            episode_reward += result.reward

            if result.done:
                break

        if result is None:
            raise RuntimeError("max_steps 为 0 或未执行任何仿真步，无法记录指标")

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
        total_metrics["departed"] += result.info.get("departed", 0)
        total_metrics["unfinished"] += result.info.get("unfinished", 0)
        total_metrics["reward"] += episode_reward

        print(f"回合 {ep + 1}/{episodes} | seed={seed_start + ep} | 奖励: {episode_reward:.2f} | "
              f"等待时间: {result.info.get('waiting_time', 0):.1f} | "
              f"到达: {result.info.get('throughput', 0)} | "
              f"发车: {result.info.get('departed', 0)} | "
              f"未完成: {result.info.get('unfinished', 0)}")

    for key in total_metrics:
        total_metrics[key] /= episodes

    return total_metrics


def main() -> None:
    """主评估函数。"""
    args = parse_args()
    device = torch.device(args.device)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("你指定了 --device cuda，但当前PyTorch未检测到可用CUDA设备。")

    env = CascadedSumoEnvironment(
        scenario=args.scenario,
        max_steps=args.max_steps,
        use_gui=args.gui,
        seed=args.seed,
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
        try:
            agent.load(args.model)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            print(f"  模型加载失败: {args.model}")
            print(f"  错误: {e}")
            if args.strict_model:
                env.close()
                raise RuntimeError("严格模式下模型加载失败，评估已中止。") from e
            print("  ⚠️ 将使用未训练模型继续评估（仅用于调试，不建议用于论文对比）")
        metrics = evaluate_dqn(
            env, agent, args.episodes, args.max_steps, recorder, seed_start=args.seed
        )
    else:
        controller = CascadedFixedTimeController(args.scenario, env.num_junctions)
        metrics = evaluate_fixed_time(
            env, controller, args.episodes, args.max_steps, recorder, seed_start=args.seed
        )

    recorder.flush(f"{args.controller}_cascaded_metrics.csv")
    env.close()

    print("\n" + "=" * 50)
    print("评估结果（平均值）:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
