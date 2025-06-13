# abra_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ABRAAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化 RL Agent 的策略网络。
        :param state_dim: 状态特征向量的维度。
        :param action_dim: 动作空间的维度 (在我们的简化版中是2)。
        :param hidden_dim: 神经网络隐藏层的大小。
        """
        super(ABRAAgent, self).__init__()
        # 一个简单的多层感知机 (MLP) 作为策略网络
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # 输出每个动作的概率
        )

    def forward(self, state_features: torch.Tensor) -> Categorical:
        """
        根据输入的状态特征，输出动作的概率分布。
        :param state_features: 从 ABRAEnv._get_state() 得到的特征化后的状态，转换成PyTorch张量。
        :return: 一个Categorical分布对象，可以从中采样动作。
        """
        action_probs = self.network(state_features)
        return Categorical(action_probs)

    def select_action(self, state_features: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        根据当前策略选择一个动作。
        :param state_features: 当前状态的特征向量。
        :param deterministic: 是否选择确定性动作（概率最高的）而不是采样。评估时通常设为True。
        :return: (选择的动作索引, 该动作的对数概率)
        """
        action_distribution = self.forward(state_features)
        if deterministic:
            action = torch.argmax(action_distribution.probs).item()
        else:
            action = action_distribution.sample().item()
        
        log_prob = action_distribution.log_prob(torch.tensor(action).to(state_features.device))
        return action, log_prob

    # 如果使用Actor-Critic类型的RL算法，还需要一个Value Network来评估状态的价值
    # class ABRAValueNetwork(nn.Module):
    #     def __init__(self, state_dim: int, hidden_dim: int = 128):
    #         super(ABRAValueNetwork, self).__init__()
    #         self.network = nn.Sequential(
    #             nn.Linear(state_dim, hidden_dim),
    #             nn.ReLU(),
    #             nn.Linear(hidden_dim, hidden_dim),
    #             nn.ReLU(),
    #             nn.Linear(hidden_dim, 1) # 输出一个标量，代表状态的价值
    #         )
    #     def forward(self, state_features: torch.Tensor) -> torch.Tensor:
    #         return self.network(state_features)