# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于DQN（深度Q网络）的区域交通信号协同优化系统，构建在SUMO微观交通仿真平台之上。本项目为毕业设计，旨在突破传统固定配时方法的局限性，利用深度强化学习实现信号灯的动态自适应调控。

核心研究目标：
- 针对不同路口拓扑结构（十字路口、丁字路口）学习最优控制策略
- 通过多路口级联场景，验证基于下游状态反馈的区域协同控制机制
- 以平均等待时间、通行率等为指标，量化DQN相对于固定配时的性能优势

## 环境要求

- **SUMO** (≥ 1.18) 已安装且设置了 `SUMO_HOME` 环境变量
- **Python 3.10+**（项目虚拟环境使用 3.13）
- 依赖安装：`pip install -r requirements.txt`

## 常用命令

所有命令从项目根目录以模块方式运行：

```bash
# 单路口训练（丁字路口 / 十字路口）
python -m training.train_dqn --scenario t_intersection --episodes 200 --double-dqn --dueling
python -m training.train_dqn --scenario x_intersection --episodes 200 --double-dqn --dueling

# 级联多路口训练
python -m training.train_cascaded --scenario cascaded_intersection --episodes 300

# 评估（DQN 或 固定配时基线）
python -m evaluation.evaluate --scenario t_intersection --controller dqn --episodes 20
python -m evaluation.evaluate --scenario t_intersection --controller fixed_time --episodes 20
python -m evaluation.evaluate_cascaded --scenario cascaded_intersection --controller dqn --episodes 20

# 绘制训练曲线
python -m utils.plot_metrics --csv outputs/t_intersection/dqn/training_metrics.csv

# 一键运行所有实验（训练+评估+报告）
python train_all.py
```

训练可选参数：`--double-dqn`、`--dueling`、`--eval-every 20`、`--gui`（SUMO可视化）、`--coordination-weight 0.3`（仅级联场景）。

无测试套件、无构建步骤、无 linter 配置。

## 系统架构

```
config/settings.py            全局配置中心（dataclass 超参数 + 路径常量）
        |
   +---------+-----------+
   |         |           |
 agent/   environment/  models/
   |         |           |
   |    sumo_env.py      q_network.py    (QNetwork: 标准MLP / Dueling架构)
   |    cascaded_env.py
   |    phase_logic.py
   |         |
   |    [SUMO via TraCI]
   |
 dqn_agent.py  (DQNAgent: 策略网络+目标网络, 经验回放, ε-贪婪)
   |
training/                  evaluation/
  train_dqn.py               evaluate.py
  train_cascaded.py          evaluate_cascaded.py
                             fixed_time_controller.py
```

### MDP 模型设计

- **状态空间（State）**：每条受控车道的 `[排队长度, 车道占用率]` + 当前信号相位的 one-hot 编码，归一化至 [0,1]。
- **动作空间（Action）**：二元动作——保持当前相位(0) / 切换到下一相位(1)。`PhaseController` 负责强制执行安全时序约束（最小绿灯10s、最大绿灯60s、黄灯4s、全红2s）。
- **奖励函数（Reward）**：基于等待时间改善量的加权奖励：`r = α × Δ优先车辆等待时间 + (1-α) × Δ普通车辆等待时间`，其中 α=0.7，引导智能体最小化车辆延误。

### 关键设计决策

- **Double DQN + Dueling 网络**：默认启用，提供更稳定准确的 Q 值估计。
- **经验回放**：固定容量20000的回放池，随机采样打破数据相关性。
- **目标网络**：每10个训练步软更新，解决训练不稳定问题。
- **梯度裁剪**：阈值5.0，配合Huber损失（SmoothL1Loss）。
- **级联协同控制**：联合动作空间 `2^n`（n个路口），单一集中式智能体。计算上游路口奖励时，将下游路口的实时拥堵状态（连接路段排队长度）作为惩罚项纳入，打破各路口孤立控制的壁垒，实现区域联动。

### 模块职责

- **`config/settings.py`**：所有超参数以 dataclass 实例管理（`training`、`environment`、`evaluation`、`cascaded`），路径常量（`BASE_DIR`、`OUTPUT_DIR`、`SCENARIO_DIR`、`MODEL_DIR`），设备选择，固定配时方案。
- **`environment/sumo_env.py`**（`SumoEnvironment`）：单路口 SUMO 环境封装，通过 TraCI 接口交互。`reset()` 启动仿真；`step(action)` 返回 `StepResult(state, reward, done, info)`。
- **`environment/cascaded_env.py`**（`CascadedSumoEnvironment`）：多路口级联环境，自动发现路口间的连接路段，拼接各路口特征 + 连接路段特征为联合状态。
- **`environment/phase_logic.py`**（`PhaseController`）：相位安全约束状态机，强制最小/最大绿灯、黄灯、全红时序。
- **`agent/dqn_agent.py`**（`DQNAgent`）：完整DQN实现，含经验回放、ε-贪婪探索（指数衰减）、目标网络更新、可选 Double DQN。
- **`models/q_network.py`**（`QNetwork`）：双隐藏层（128单元）MLP，可选 Dueling 分支（价值流 + 优势流）。
- **`utils/replay_buffer.py`**：固定容量 deque 存储 `Transition` 命名元组，批量采样转为 PyTorch 张量。
- **`utils/metrics.py`**（`MetricsRecorder`）：累积每回合指标，通过 pandas 写入 CSV。

### 仿真场景

三种 SUMO 场景位于 `scenarios/`，每个包含 `network.net.xml`、`routes.rou.xml`、`simulation.sumocfg`（3600s仿真、1s步长、禁用瞬移）：

| 场景 | 目录 | 研究目的 |
|------|------|----------|
| 十字路口 | `x_intersection/`（4相位） | 验证算法基础性能 |
| 丁字路口 | `t_intersection/`（2相位） | 测试非对称拓扑下的适应性 |
| 级联双路口 | `cascaded_intersection/` | 验证多路口协同控制策略的有效性 |

### 输出结果

- 实验指标（CSV）和图表（PNG）保存至 `outputs/{scenario}/{controller}/`
- 模型检查点保存至 `models/checkpoints/`
- 对比报告保存至 `outputs/comparison_report/`

## 代码规范

- 所有注释、文档字符串、UI 文本均为**中文**
- 全局使用类型注解（`from __future__ import annotations`）
- 非可安装包——必须从项目根目录以模块方式运行
- 使用 Python dataclasses 且 `slots=True`
