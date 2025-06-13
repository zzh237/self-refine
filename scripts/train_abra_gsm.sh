#!/bin/bash

# 获取项目根目录 (假设此脚本在 "scripts" 文件夹下, "scripts" 在项目根目录)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# 设置PYTHONPATH，确保src目录下的模块能被找到
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" 

# (可选) 设置模型缓存目录的环境变量，然后在config.py中读取
# export MY_MODEL_CACHE_DIR="/data/ebay-lvs-a100/notebooks/zzhang12/model_cache/"

# 进入项目根目录执行 (或者直接用绝对路径调用python脚本)
cd "$PROJECT_ROOT"

# 定义输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/abra_train_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

python main_train_abra.py \
    --config_path config.py \ # 如果你的config加载方式改变了
    --model_save_path "${OUTPUT_DIR}/abra_agent_final.pth" \
    --log_dir "${OUTPUT_DIR}/logs/" \
    # ... 其他你希望通过命令行覆盖的config参数 ...
    # 例如：
    # --train_data_path "data/tasks/gsm/my_train.jsonl" \
    # --total_training_episodes 5000 

echo "训练完成，结果保存在 ${OUTPUT_DIR}"