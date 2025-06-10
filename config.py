# config.py

# --- 模型相关配置 ---
# 假设你有一个函数或类来加载你的LLM模型，这里用占位符
# 你需要根据你的实际情况来初始化这些模型
# 例如:
# from src.gsm.task_init import GSMInit
# from src.gsm.feedback import GSMFeedback
# GENERATION_MODEL_ENGINE = "code-davinci-002" # 或其他
# FEEDBACK_MODEL_ENGINE = "code-davinci-002" # 或其他
# PROMPT_INIT_PATH = "data/prompt/gsm/init.txt"
# PROMPT_FEEDBACK_PATH = "data/prompt/gsm/feedback.txt"

# 占位符 - 你需要替换成实际的模型加载/初始化逻辑
M_GEN_PARAMS = {"engine": "your_engine_name", "prompt_examples": "path/to/init_prompt.txt", "temperature": 0.0}
M_EDIT_FEEDBACK_PARAMS = {"engine": "your_engine_name", "prompt_examples": "path/to/feedback_prompt.txt", "temperature": 0.7}
# M_EVAL_PARAMS = {} # 如果有独立的评估模型

# --- ABRA 环境配置 ---
TOTAL_BUDGET_PER_PROBLEM = 1000.0 # 示例：最大token数
MAX_ITERATIONS_PER_PROBLEM = 5     # 示例：ABRA算法中的 i_max

# --- RL Agent 配置 ---
STATE_DIM_PLACEHOLDER = 6 # 状态特征的维度，需要根据 _get_state() 的实际输出确定
ACTION_DIM = 2            # 0: ContinueRefinement, 1: TerminateAndSelect
AGENT_HIDDEN_DIM = 64

# --- PPO 训练配置 (如果手写PPO，或者给RL库的参数) ---
LR_POLICY = 3e-4
LR_VALUE = 1e-3         # 如果使用Actor-Critic
GAMMA = 0.99            # 折扣因子
K_EPOCHS = 4            # 每次更新时，用同一批数据训练的轮数
EPS_CLIP = 0.2          # PPO裁剪参数
ENTROPY_COEFF = 0.01    # 熵正则化系数
BATCH_SIZE_RL = 32      # 从经验回放缓冲区采样的小批量大小 (对应ABRA算法的 B_RL)
REPLAY_BUFFER_CAPACITY = 1000 # 经验回放缓冲区的最大容量

# --- 训练循环配置 (ABRA Algorithm 1 的参数) ---
TOTAL_TRAINING_EPISODES = 10000 # 总训练轮次 (N_episodes_total)
POLICY_UPDATE_FREQUENCY = 100   # 每多少轮更新一次策略 (K_update)
EVAL_FREQUENCY = 500          # 每多少轮评估一次模型 (N_eval_frequency)
LOG_FREQUENCY = 50            # 每多少轮打印一次日志

# --- 数据集路径 ---
TRAIN_DATA_PATH = "path/to/your/train_data.jsonl" # 训练数据集
VAL_DATA_PATH = "path/to/your/val_data.jsonl"     # 验证数据集

# --- 输出路径 ---
MODEL_SAVE_PATH = "saved_models/abra_agent.pth"
LOG_DIR = "logs/"

# --- 其他 ---
DEVICE = "cuda" # 或者 "cpu"
SEED = 42