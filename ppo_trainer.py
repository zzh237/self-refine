# ppo_trainer.py (简化版PPO核心逻辑示例)
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

class PPOTrainer:
    def __init__(self, policy_net: ABRAAgent, value_net, # 如果是Actor-Critic PPO，则需要value_net
                 lr_policy=3e-4, lr_value=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2,
                 entropy_coeff=0.01):
        self.policy_net = policy_net
        # self.value_net = value_net # 如果有价值网络
        self.optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr_policy)
        # self.optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value) # 如果有价值网络

        self.gamma = gamma # 折扣因子，未来奖励的重要性
        self.K_epochs = K_epochs # 每次策略更新时，用同一批数据训练几轮
        self.eps_clip = eps_clip # PPO的裁剪参数，限制策略更新幅度
        self.entropy_coeff = entropy_coeff # 熵正则化系数，鼓励探索

        self.mse_loss = torch.nn.MSELoss() # 用于价值网络损失 (如果使用)

    def update(self, replay_buffer: List[Dict]): # replay_buffer 存储的是完整的轨迹
        """
        使用从经验回放缓冲区中采样的轨迹来更新策略网络。
        :param replay_buffer: 包含多条轨迹的列表，每条轨迹是一个字典，
                              例如: {'states': [s0, s1, ...], 'actions': [a0, a1, ...],
                                     'log_probs_old': [log_p0, log_p1, ...], 'rewards': [r0, r1, ..., R_final]}
                                     注意：这里的rewards应该是每一步的reward，对于ABRA，大部分是0，最后一步是R_final
        """
        # 1. 从轨迹计算回报 (Returns) 和优势 (Advantages)
        #    这部分是PPO的核心，需要仔细实现，这里只是一个概念性的表示
        all_rewards = []
        all_old_log_probs = []
        all_states_for_policy = [] # 收集所有状态特征张量
        all_actions_for_policy = [] # 收集所有动作张量

        for trajectory in replay_buffer:
            rewards = []
            discounted_reward = 0
            # 从后往前计算每个时间步的折扣回报 G_t
            for r_step in reversed(trajectory['rewards_at_each_step']): # 假设轨迹中存的是每一步的即时奖励
                discounted_reward = r_step + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward) # 插入到列表头部
            
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            # 标准化回报 (可选，但通常有益)
            # rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)

            all_rewards.extend(rewards_tensor)
            all_old_log_probs.extend(torch.tensor(trajectory['log_probs_old'], dtype=torch.float32))
            all_states_for_policy.extend(trajectory['states_features_tensor']) # 假设状态特征已是张量
            all_actions_for_policy.extend(torch.tensor(trajectory['actions_taken'], dtype=torch.int64))

        # 转换成张量
        all_rewards = torch.stack(all_rewards)
        all_old_log_probs = torch.stack(all_old_log_probs).detach() # detach() 因为这是旧策略的，不需要梯度
        all_states_for_policy = torch.stack(all_states_for_policy)
        all_actions_for_policy = torch.stack(all_actions_for_policy)

        # 2. 多轮优化 (K_epochs)
        for _ in range(self.K_epochs):
            # 从收集的数据中随机采样小批次进行训练
            # (更复杂的实现会用 BatchSampler(SubsetRandomSampler(...)))
            # 这里为了简化，假设我们一次性处理所有收集到的数据（如果数据量不大）
            
            # a. 计算当前策略下，这些动作的对数概率和状态的价值（如果用Actor-Critic）
            action_dists = self.policy_net(all_states_for_policy) # [batch_size, action_dim]
            current_log_probs = action_dists.log_prob(all_actions_for_policy) # [batch_size]
            entropy = action_dists.entropy().mean() # 熵，用于鼓励探索

            # state_values = self.value_net(all_states_for_policy).squeeze() # [batch_size] (如果使用价值网络)
            # advantages = all_rewards - state_values.detach() # 计算优势函数 (简化版)
            # 对于 REINFORCE 类型的 PPO，优势就是回报本身 (或者回报减去基线)
            advantages = all_rewards # 简单起见，直接用回报作为优势 (或经过基线处理的回报)


            # b. 计算 PPO 裁剪损失 (Clipped Surrogate Objective)
            ratios = torch.exp(current_log_probs - all_old_log_probs) # pi_theta(a|s) / pi_theta_old(a|s)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean() # PPO 策略损失

            # c. (可选) 计算价值损失 (如果使用Actor-Critic)
            # value_loss = self.mse_loss(state_values, all_rewards)

            # d. 总损失 (策略损失 - 熵奖励 + 价值损失)
            # loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy # Actor-Critic PPO
            loss = policy_loss - self.entropy_coeff * entropy # REINFORCE-style PPO

            # e. 梯度反向传播和优化
            self.optimizer_policy.zero_grad()
            # if self.optimizer_value: self.optimizer_value.zero_grad() # 如果有价值网络
            
            loss.backward()
            
            self.optimizer_policy.step()
            # if self.optimizer_value: self.optimizer_value.step() # 如果有价值网络

        # （可选）清空replay_buffer，或者只清空用于本次更新的部分
        # replay_buffer.clear()