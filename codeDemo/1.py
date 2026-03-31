import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, config: QNetworkConfig) -> None:
        super().__init__()
        # 动态构建特征提取层，默认采用双层隐藏层结构且每层包含一百二十八个神经元
        layers = []
        input_dim = config.state_dim
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            # 引入线性整流激活函数以增强网络对复杂交通流特征的非线性映射能力
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

        # 动作优势分离架构，剥离环境固有状态价值流与具体动作优势流
        if config.dueling:
            self.value_head = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                # 该分支最终汇总输出单一的环境状态综合评估价值
                nn.Linear(config.hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                # 该分支最终汇总输出动作空间各个动作的相对优势数值
                nn.Linear(config.hidden_dim, config.action_dim),
            )
        else:
            # 兼容保留标准深度网络架构，直接输出动作空间维度的最终评估值
            self.q_head = nn.Linear(input_dim, config.action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        if hasattr(self, 'value_head'):
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            # 根据分离评估理论公式，通过均值去中心化计算重构出聚合后的最终动作参考价值
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return self.q_head(features)