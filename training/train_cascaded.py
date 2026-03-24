"""级联多路口DQN控制器训练脚本。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from agent.dqn_agent import DQNAgent, AgentConfig
from config.settings import MODEL_DIR, training as train_cfg, get_device
from environment.cascaded_env import CascadedSumoEnvironment
from utils import MetricsRecorder, Transition, set_global_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练级联DQN交通控制器")
    parser.add_argument("--scenario", type=str, default="cascaded_intersection",
                        help="场景名称")
    parser.add_argument("--episodes", type=int, default=300,
                        help="训练回合数")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="每回合最大步数")
    parser.add_argument("--double-dqn", action="store_true", default=True,
                        help="启用Double DQN")
    parser.add_argument("--dueling", action="store_true", default=True,
                        help="使用Dueling网络架构")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="评估频率")
    parser.add_argument("--device", type=str, default=get_device(),
                        help="计算设备")
    parser.add_argument("--save-dir", type=str, default=MODEL_DIR.as_posix(),
                        help="模型保存目录")
    parser.add_argument("--coordination-weight", type=float, default=0.3,
                        help="协同奖励权重")
    parser.add_argument("--gui", action="store_true",
                        help="使用SUMO图形界面")
    return parser.parse_args()


def main() -> None:
    """主训练函数。"""
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device)

    # 创建级联环境
    env = CascadedSumoEnvironment(
        scenario=args.scenario,
        max_steps=args.max_steps,
        use_gui=args.gui,
        coordination_weight=args.coordination_weight,
    )

    # 初始化环境以获取维度
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.action_dim

    print(f"级联环境信息:")
    print(f"  - 路口数量: {env.num_junctions}")
    print(f"  - 路口ID: {env.junction_ids}")
    print(f"  - 状态维度: {state_dim}")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 连接边: {env.link_edges}")

    # 创建智能体
    agent_config = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        double_dqn=args.double_dqn,
        dueling=args.dueling,
    )
    agent = DQNAgent(agent_config)

    # 指标记录
    metrics_dir = Path("outputs") / args.scenario / "dqn_cascaded"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    recorder = MetricsRecorder(metrics_dir)

    best_reward = float("-inf")

    print(f"\n开始训练，共{args.episodes}回合...")

    pbar = tqdm(range(args.episodes), desc="训练进度", unit="回合")
    for episode in pbar:
        state = env.reset()
        episode_reward = 0.0
        episode_coordination_bonus = 0.0
        episode_losses: list[float] = []
        junction_rewards = {jid: 0.0 for jid in env.junction_ids}

        for step in range(args.max_steps):
            action = agent.select_action(state)
            result = env.step(action)

            episode_reward += result.reward
            episode_coordination_bonus += result.info.get("coordination_bonus", 0.0)

            for jid, jr in result.junction_rewards.items():
                junction_rewards[jid] += jr

            transition = Transition(state, action, result.reward, result.state, result.done)
            agent.store_transition(transition)
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = result.state
            if result.done:
                break

        agent.decay_epsilon()

        # 回合平均损失
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None

        # 更新进度条
        pbar.set_postfix({
            "奖励": f"{episode_reward:.1f}",
            "协同": f"{episode_coordination_bonus:.1f}",
            "ε": f"{agent.epsilon:.3f}"
        })

        # 记录指标
        metrics = {
            "episode": episode,
            "reward": episode_reward,
            "coordination_bonus": episode_coordination_bonus,
            "waiting_time": result.info.get("waiting_time", 0),
            "avg_queue": result.info.get("avg_queue", 0),
            "link_queue": result.info.get("link_queue", 0),
            "throughput": result.info.get("throughput", 0),
        }
        for jid, jr in junction_rewards.items():
            metrics[f"reward_{jid}"] = jr

        if avg_loss is not None:
            metrics["loss"] = avg_loss

        recorder.log(metrics)

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = Path(args.save_dir) / "cascaded_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(save_path))

        # 定期检查点
        if (episode + 1) % args.eval_every == 0:
            save_path = Path(args.save_dir) / f"cascaded_ep{episode + 1}.pt"
            agent.save(str(save_path))

    recorder.flush("cascaded_training_metrics.csv")
    env.close()

    print(f"\n训练完成！最佳奖励: {best_reward:.2f}")
    print(f"模型保存至 {args.save_dir}")


if __name__ == "__main__":
    main()
