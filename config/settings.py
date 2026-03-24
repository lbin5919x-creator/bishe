"""基于DQN的交通信号控制全局配置参数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# 项目路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
SCENARIO_DIR = BASE_DIR / "scenarios"
MODEL_DIR = BASE_DIR / "models" / "checkpoints"


@dataclass(slots=True)
class TrainingConfig:
    """训练配置参数。"""
    scenario: str = "t_intersection"      # 场景名称
    episodes: int = 200                   # 训练回合数
    max_steps: int = 3600                 # 每回合最大步数（秒）
    warmup_episodes: int = 5              # 预热回合数
    eval_every: int = 20                  # 评估频率
    batch_size: int = 64                  # 批次大小
    gamma: float = 0.99                   # 折扣因子
    learning_rate: float = 1e-3           # 学习率
    epsilon_start: float = 1.0            # 初始探索率
    epsilon_end: float = 0.05             # 最终探索率
    epsilon_decay: int = 200              # 探索率衰减步数
    buffer_capacity: int = 20000          # 经验回放缓冲区容量
    target_update: int = 10               # 目标网络更新频率
    double_dqn: bool = True               # 是否使用Double DQN
    dueling: bool = True                  # 是否使用Dueling网络
    gradient_clip: float = 5.0            # 梯度裁剪阈值


@dataclass(slots=True)
class EnvironmentConfig:
    """环境配置参数。"""
    min_green: int = 10                   # 最小绿灯时长（秒）
    max_green: int = 60                   # 最大绿灯时长（秒）
    yellow: int = 4                       # 黄灯时长（秒）
    all_red: int = 2                      # 全红时长（秒）
    time_step: float = 1.0                # 仿真时间步长（秒）
    reward_alpha: float = 0.7             # 优先车辆奖励权重


@dataclass(slots=True)
class EvaluationConfig:
    """评估配置参数。"""
    episodes: int = 20                    # 评估回合数
    log_interval: int = 100               # 日志记录间隔


# 创建配置实例
training = TrainingConfig()
environment = EnvironmentConfig()
evaluation = EvaluationConfig()

# 定时控制方案（每相位绿灯时长，秒）
FIXED_TIME_PLANS = {
    "t_intersection": [40, 40],
    "x_intersection": [30, 30, 30, 30],
    "cascaded_intersection": [30, 30, 30, 30],  # 每路口
}


@dataclass(slots=True)
class CascadedConfig:
    """级联多路口控制配置。"""
    coordination_weight: float = 0.3      # 下游反馈奖励权重
    link_queue_penalty: float = 0.5       # 连接路排队惩罚系数
    max_link_queue: float = 30.0          # 连接路排队归一化因子


cascaded = CascadedConfig()


def get_device(prefer_gpu: bool = True) -> str:
    """
    获取计算设备。
    
    参数:
        prefer_gpu: 是否优先使用GPU
        
    返回:
        设备名称 ("cuda" 或 "cpu")
    """
    if prefer_gpu:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"
