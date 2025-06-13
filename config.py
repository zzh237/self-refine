# config.py

# --- 常量定义模型基础路径 ---
# 这样做的好处是，如果你的模型都放在一个共同的父目录下，修改起来方便
MODEL_CACHE_BASE_PATH = "/data/ebay-lvs-a100/notebooks/zzhang12/model_cache/" # <--- 你的模型缓存根目录

# --- 模型相关配置 ---
# 你需要根据你实际想用的模型，填写它们在 MODEL_CACHE_BASE_PATH 下的子目录名
# 例如，如果你的 Mistral-7B-Instruct-v0.3 模型在 
# /data/ebay-lvs-a100/notebooks/zzhang12/model_cache/mistralai--Mistral-7B-Instruct-v0.3/
# 那么 engine 的值就应该是这个完整路径。

# 示例：假设你想用 Mistral-7B-Instruct-v0.3 作为初始生成模型
M_GEN_ENGINE_NAME = "mistralai--Mistral-7B-Instruct-v0.3" # 或者你下载的其他模型的文件夹名
M_GEN_PROMPT_PATH = "data/prompt/gsm/init.txt" # 这个路径是相对于项目根目录的

# 示例：假设你想用 Qwen2.5-7B-Instruct 作为反馈和修正模型
M_EDIT_FEEDBACK_ENGINE_NAME = "Qwen--Qwen2.5-7B-Instruct" # 或者你下载的其他模型的文件夹名
M_EDIT_FEEDBACK_PROMPT_PATH = "data/prompt/gsm/feedback.txt" # 相对于项目根目录

M_GEN_PARAMS = {
    "engine": MODEL_CACHE_BASE_PATH + M_GEN_ENGINE_NAME, # <--- 拼接成完整路径
    "prompt_examples": M_GEN_PROMPT_PATH, 
    "temperature": 0.0,
    # ===> 为本地HF模型添加必要的加载参数 <===
    "trust_remote_code": True, # 大部分开源模型需要这个
    # "model_load_kwargs": {"revision": "main"} # 如果需要指定分支等
}

M_EDIT_FEEDBACK_PARAMS = {
    "engine": MODEL_CACHE_BASE_PATH + M_EDIT_FEEDBACK_ENGINE_NAME, # <--- 拼接成完整路径
    "prompt_examples": M_EDIT_FEEDBACK_PROMPT_PATH, 
    "temperature": 0.7,
    "max_tokens": 600,
    # ===> 为本地HF模型添加必要的加载参数 <===
    "trust_remote_code": True,
}

# --- ABRA 环境配置 ---
TOTAL_BUDGET_PER_PROBLEM = 1000.0 
MAX_ITERATIONS_PER_PROBLEM = 5     

# --- RL Agent 配置 ---
STATE_DIM = 6 # 保持与 featurize_state_for_agent 的输出一致
ACTION_DIM = 2            
AGENT_HIDDEN_DIM = 64

# --- PPO 训练配置 ---
# ... (PPO参数保持之前的建议) ...
LR_POLICY = 3e-4
LR_VALUE = 1e-3        
GAMMA = 0.99           
K_EPOCHS = 4           
EPS_CLIP = 0.2         
ENTROPY_COEFF = 0.01   
BATCH_SIZE_RL = 32     
REPLAY_BUFFER_CAPACITY = 1000 

# --- 训练循环配置 ---
TOTAL_TRAINING_EPISODES = 10000 
POLICY_UPDATE_FREQUENCY = 100   
EVAL_FREQUENCY = 500          
LOG_FREQUENCY = 50            

# --- 数据集路径 ---
# ===> 这些路径也是相对于项目根目录的，请确保正确 <===
TRAIN_DATA_PATH = "data/tasks/gsm/your_training_data_subset.jsonl" 
VAL_DATA_PATH = "data/tasks/gsm/your_validation_data_subset.jsonl"     

# --- 输出路径 ---
# ===> 这些路径也是相对于项目根目录的 <===
MODEL_SAVE_PATH = "saved_models/abra_agent.pth"
LOG_DIR = "logs/"
# 确保 saved_models 和 logs 文件夹存在，或者代码有创建它们的逻辑

# --- 其他 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 自动检测
SEED = 42
BUDGET_PENALTY_FACTOR = 0.1 # 奖励函数中的预算惩罚因子
EVAL_EXEC_TIMEOUT_SECONDS = 3 # 评估代码执行时的超时秒数