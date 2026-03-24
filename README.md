# 基于DQN的区域交通信号协同优化系统

基于深度Q网络（DQN）强化学习的自适应交通信号控制系统，构建在SUMO微观交通仿真平台之上。针对丁字路口、十字路口和级联多路口三种场景，通过与固定配时基线的对比实验，验证DQN在降低等待时间、提升通行效率方面的优势。

## 功能特性

- **深度强化学习控制**：Double DQN + Dueling 网络架构，经验回放，目标网络软更新
- **SUMO仿真集成**：通过TraCI接口实时获取交通数据并控制信号灯
- **安全相位约束**：最小/最大绿灯时长、黄灯过渡、全红保护的完整状态机
- **多路口协同**：级联场景中基于下游拥堵状态反馈的区域联动控制
- **可视化对比演示**：SUMO GUI中实时展示DQN与固定配时的性能差异

## 项目结构

```
dqn_traffic/
├── .gitignore
├── README.md
├── CLAUDE.md
├── requirements.txt
├── comparison_demo.py              # 固定配时 vs DQN 可视化对比演示
│
├── agent/
│   └── dqn_agent.py                # DQN智能体（策略网络+目标网络+经验回放）
├── config/
│   └── settings.py                 # 全局超参数与路径常量
├── environment/
│   ├── sumo_env.py                 # 单路口SUMO环境封装
│   ├── cascaded_env.py             # 多路口级联环境
│   └── phase_logic.py              # 相位安全约束状态机
├── evaluation/
│   ├── evaluate.py                 # 单路口评估
│   ├── evaluate_cascaded.py        # 级联场景评估
│   └── fixed_time_controller.py    # 固定配时基线控制器
├── models/
│   ├── q_network.py                # Q网络（MLP / Dueling架构）
│   └── checkpoints/                # 训练模型存放目录
├── training/
│   ├── train_dqn.py                # 单路口训练脚本
│   └── train_cascaded.py           # 级联场景训练脚本
├── scenarios/
│   ├── t_intersection/             # 丁字路口（3方向，2相位组）
│   ├── x_intersection/             # 十字路口（4方向，4相位组）
│   └── cascaded_intersection/      # 级联双路口（十字+丁字）
└── utils/
    ├── replay_buffer.py            # 经验回放缓冲区
    ├── metrics.py                  # 指标记录器
    ├── plot_metrics.py             # 训练曲线绘制
    └── seed.py                     # 随机种子工具
```

## 环境要求

- **SUMO** ≥ 1.18，且设置了 `SUMO_HOME` 环境变量
- **Python** 3.10+

```bash
pip install -r requirements.txt
```

## 训练

所有命令从项目根目录以模块方式运行：

```bash
# 丁字路口（模型保存为 t_intersection_best.pt）
python -m training.train_dqn --scenario t_intersection --episodes 200 --double-dqn --dueling

# 十字路口（模型保存为 x_intersection_best.pt）
python -m training.train_dqn --scenario x_intersection --episodes 200 --double-dqn --dueling

# 级联双路口（模型保存为 cascaded_best.pt）
python -m training.train_cascaded --scenario cascaded_intersection --episodes 300
```

可选参数：
- `--double-dqn` 启用Double DQN（默认开启）
- `--dueling` 使用Dueling网络架构（默认开启）
- `--eval-every 20` 定期保存检查点
- `--gui` 使用SUMO图形界面观察训练过程
- `--coordination-weight 0.3` 协同奖励权重（仅级联场景）

## 评估

```bash
# 单路口：DQN vs 固定配时
python -m evaluation.evaluate --scenario t_intersection --controller dqn --episodes 20
python -m evaluation.evaluate --scenario t_intersection --controller fixed_time --episodes 20

# 级联场景
python -m evaluation.evaluate_cascaded --scenario cascaded_intersection --controller dqn --episodes 20
python -m evaluation.evaluate_cascaded --scenario cascaded_intersection --controller fixed_time --episodes 20
```

评估指标保存至 `outputs/{scenario}/{controller}/`。

## 对比演示

在SUMO GUI中逐一运行固定配时和DQN，实时显示性能对比数据：

```bash
python comparison_demo.py --scenario t_intersection --duration 600
python comparison_demo.py --scenario x_intersection --duration 600
python comparison_demo.py --scenario cascaded_intersection --duration 600
```

支持 `--mode fixed`（仅固定配时）、`--mode dqn`（仅DQN）、`--mode both`（先后对比，默认）。

## 绘制训练曲线

```bash
python -m utils.plot_metrics --csv outputs/t_intersection/dqn/training_metrics.csv
```

## MDP模型设计

| 要素 | 设计 |
|------|------|
| **状态** | 每条受控车道的 [排队长度, 占用率] + 当前相位 one-hot，归一化至 [0,1] |
| **动作** | 二元：保持当前相位(0) / 切换到下一相位(1) |
| **奖励** | `r = 0.7 × Δ优先车辆等待时间 + 0.3 × Δ普通车辆等待时间` |
| **级联扩展** | 联合动作空间 2^n，下游拥堵状态作为惩罚项纳入上游奖励 |
