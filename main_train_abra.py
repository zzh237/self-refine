# main_train_abra.py
import torch
from collections import deque # 用于经验回放缓冲区
import numpy as np

import config # 导入配置
from utils import set_seed, load_dataset, featurize_state_for_agent, MockGSMInit, MockGSMFeedback, mock_evaluate_solution_and_cost
from abra_env import ABRAEnv
from abra_agent import ABRAAgent
from ppo_trainer import PPOTrainer # 我们手写的PPO，或者用RL库的

def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # 1. 加载数据集
    train_problems = load_dataset(config.TRAIN_DATA_PATH)
    # val_problems = load_dataset(config.VAL_DATA_PATH) # 用于评估

    # 2. 初始化 LLM 组件 (这里使用Mock，你需要替换成真实的)
    m_gen_instance = MockGSMInit(**config.M_GEN_PARAMS)
    m_edit_feedback_instance = MockGSMFeedback(**config.M_EDIT_FEEDBACK_PARAMS)
    # 评估函数可以直接使用 mock_evaluate_solution_and_cost
    m_eval_function = mock_evaluate_solution_and_cost

    # 3. 初始化 RL Agent
    # 确保 config.STATE_DIM_PLACEHOLDER 与 featurize_state_for_agent 的输出维度匹配
    agent = ABRAAgent(state_dim=config.STATE_DIM_PLACEHOLDER, 
                      action_dim=config.ACTION_DIM, 
                      hidden_dim=config.AGENT_HIDDEN_DIM).to(device)

    # 4. 初始化 PPO 训练器 (如果使用手写的)
    # 如果用RL库，这里会是库的PPO模型初始化
    ppo_trainer = PPOTrainer(policy_net=agent, value_net=None, # 简化版PPO不需要value_net
                             lr_policy=config.LR_POLICY, 
                             gamma=config.GAMMA, 
                             K_epochs=config.K_EPOCHS, 
                             eps_clip=config.EPS_CLIP,
                             entropy_coeff=config.ENTROPY_COEFF)

    # 5. 初始化经验回放缓冲区
    # replay_buffer = deque(maxlen=config.REPLAY_BUFFER_CAPACITY)
    # 对于PPO，通常是在每次策略更新前收集一批新的数据，而不是用一个持续的大的replay buffer
    # 所以，我们会在每个 update_frequency 周期内收集数据
    collected_trajectories_for_update = []


    print("开始训练 ABRA Agent...")
    for episode_num in range(1, config.TOTAL_TRAINING_EPISODES + 1):
        # 从训练集中随机选择一个问题
        problem_data = random.choice(train_problems)
        
        # 初始化当前问题的环境
        env = ABRAEnv(problem_data=problem_data,
                      m_gen=m_gen_instance,
                      m_edit_feedback=m_edit_feedback_instance,
                      m_eval_func=m_eval_function,
                      total_budget=config.TOTAL_BUDGET_PER_PROBLEM,
                      max_iterations=config.MAX_ITERATIONS_PER_PROBLEM)

        current_state_dict = env.reset()
        current_state_features = featurize_state_for_agent(current_state_dict, device)
        
        # --- 为当前 episode 收集轨迹 ---
        episode_states = []
        episode_actions = []
        episode_log_probs_old = []
        episode_rewards_at_each_step = [] # 存储每一步的即时奖励
        episode_total_reward = 0

        for t in range(config.MAX_ITERATIONS_PER_PROBLEM): # 或者直到 done
            action, log_prob_old = agent.select_action(current_state_features, deterministic=False)
            
            next_state_dict, reward, done, info = env.step(action)
            
            # 存储经验
            episode_states.append(current_state_features) # 存储特征化后的状态
            episode_actions.append(action)
            episode_log_probs_old.append(log_prob_old.item()) # 存储标量值
            episode_rewards_at_each_step.append(reward) # 存储这一步的实际奖励
            
            episode_total_reward += reward # 累计的是最终的稀疏奖励

            current_state_dict = next_state_dict
            current_state_features = featurize_state_for_agent(current_state_dict, device)

            if done:
                break
        
        # 将完成的轨迹存储起来，用于后续的PPO更新
        collected_trajectories_for_update.append({
            'states_features_tensor': torch.cat(episode_states, dim=0), # 将列表中的张量堆叠起来
            'actions_taken': episode_actions,
            'log_probs_old': episode_log_probs_old,
            'rewards_at_each_step': episode_rewards_at_each_step, # 传递每一步的奖励
            'final_reward_for_episode': episode_total_reward # 这个主要是用于日志记录
        })

        # --- 策略更新 ---
        if episode_num % config.POLICY_UPDATE_FREQUENCY == 0 and len(collected_trajectories_for_update) >= config.BATCH_SIZE_RL:
            print(f"Episode {episode_num}: 正在更新策略...")
            # 从 collected_trajectories_for_update 中采样 BATCH_SIZE_RL 条轨迹进行更新
            # （更严谨的PPO通常会用完所有收集到的数据进行更新，然后清空）
            # 这里简化为：如果收集的轨迹足够一个batch，就用这些轨迹更新
            
            # 实际的PPO通常会用完当前收集周期内的所有数据
            # 这里假设我们用所有收集到的轨迹，如果数量超过BATCH_SIZE_RL
            if len(collected_trajectories_for_update) >= config.BATCH_SIZE_RL : # 确保至少有一个RL Batch的数据
                # 如果要严格按 BATCH_SIZE_RL, 需要从 collected_trajectories_for_update 中采样
                # batch_to_update = random.sample(collected_trajectories_for_update, config.BATCH_SIZE_RL)
                # ppo_trainer.update(batch_to_update)

                # 更常见的做法是用完所有收集到的数据
                ppo_trainer.update(collected_trajectories_for_update)
                collected_trajectories_for_update.clear() # 更新后清空，为下一个周期收集数据
            
            print("策略更新完成。")

        # --- 日志记录 ---
        if episode_num % config.LOG_FREQUENCY == 0:
            # 计算最近N个episode的平均最终奖励
            avg_final_reward = np.mean([traj['final_reward_for_episode'] for traj in collected_trajectories_for_update[-config.LOG_FREQUENCY:] if collected_trajectories_for_update]) if collected_trajectories_for_update else 0
            print(f"Episode {episode_num}/{config.TOTAL_TRAINING_EPISODES} | "
                  f"最近{config.LOG_FREQUENCY}轮平均最终奖励: {avg_final_reward:.3f} | "
                  f"问题ID示例: {problem_data.get('id', 'N/A')}")
            if 'final_solution' in info and info['final_solution']:
                 print(f"  -> 最终解 (示例): {info['final_solution'][:100]}...")
                 print(f"  -> 状态: {info.get('status', 'N/A')}")


        # --- 定期评估 (可选) ---
        if episode_num % config.EVAL_FREQUENCY == 0:
            print(f"\nEpisode {episode_num}: 正在评估模型...")
            # 这里需要一个评估函数，它使用当前agent（deterministic=True）在验证集上运行
            # 并报告性能指标（例如，准确率，平均预算使用）
            # evaluate_agent_performance(agent, val_problems, config, device)
            print("评估完成 (占位符)。\n")
            
            # 保存模型
            torch.save(agent.state_dict(), f"{config.MODEL_SAVE_PATH}.ep{episode_num}")
            print(f"模型已保存到 {config.MODEL_SAVE_PATH}.ep{episode_num}")


    print("训练完成!")
    torch.save(agent.state_dict(), config.MODEL_SAVE_PATH)
    print(f"最终模型已保存到 {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()