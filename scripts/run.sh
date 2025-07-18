#!/bin/bash
# scripts/run_all.sh

# Exit immediately if a command exits with a non-zero status.
set -e



# echo ""
# echo ">>> 开始环境侦察..."
# echo "------------------------------------"

# # 1. 打印当前激活的 Conda 环境名称
# # 如果没有激活任何Conda环境，这可能会打印 "None" 或为空
# echo "当前 Conda 环境 (CONDA_DEFAULT_ENV): $CONDA_DEFAULT_ENV"

# # 2. 打印正在使用的 python 解释器的绝对路径
# # `which python3` 会告诉你，当系统执行 `python3` 命令时，它用的是哪个文件
# echo "使用的 Python 解释器路径 (which python3): $(which python3)"

# echo "正在检查 Python 运行时链接的 GLIBCXX 版本..."
# python3 -c 'import os; print(os.popen("strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX").read())'

# # 3. 打印 Conda 环境列表，星号(*)会标记出当前激活的环境
# echo "所有可用的 Conda 环境列表 (conda info --envs):"
# conda info --envs
# echo "------------------------------------"

# echo "PYTHONPATH已更新为: $PYTHONPATH" # 调试时可以看看这个路径对不对
# echo "--- LD_LIBRARY_PATH ---"
# echo $LD_LIBRARY_PATH




# echo ">>> 环境侦察结束。"
# echo ""
# ====



export PROXY_USER="zzhang12"
export PROXY_PASS="1011ZZaibm871011"

# 获取项目根目录 (假设此脚本在 "scripts" 文件夹下, "scripts" 在项目根目录)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."    # /data/ebay-slc-h200/notebooks/zzhang12/backups/self-refine/ 

# # 设置PYTHONPATH，确保src目录下的模块能被找到
# export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" 

# (可选) 设置模型缓存目录的环境变量，然后在config.py中读取
# export MY_MODEL_CACHE_DIR="/data/ebay-lvs-a100/notebooks/zzhang12/model_cache/"

# 进入项目根目录执行 (或者直接用绝对路径调用python脚本)
cd "$PROJECT_ROOT"


# # Define project root relative to the script location
# PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_SCRIPT="src/experiments/run_experiment.py"
# export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "项目根目录已设置为: $PROJECT_ROOT"



MODEL_CACHE_PATH="/data/ebay-slc-h200/notebooks/zzhang12/model_cache/meta-llama--Llama-3.3-70B-Instruct"
MODEL_NAME="Llama-3.3-70B-Instruct"




echo "====================================================="
echo "  ABRA PRELIMINARY EXPERIMENT LAUNCHER"
echo "  Project Root: $PROJECT_ROOT"
echo "====================================================="

# Define the full experimental matrix
# declare -a DATASETS=("google_math" "gsm8k" "hotpotqa" "musique" "2wikimultihopqa")
# declare -a STRATEGIES=("parallel" "sequential" "parallel-rrm")
# declare -a BUDGETS=(1024 2048 4096 8192) 1024 2048 
# 3. Total Compute Budgets (total tokens allowed per question)
#    We use your suggestion to test two different levels.
# declare -a COMPUTE_BUDGETS=(4096 8192)
# 4. Number of Samples/Steps (how to allocate the budget)
# declare -a N_SAMPLES=(4 8 16 32)


DEBUG_MODE_ENABLED="false"
declare -a DATASETS=("hotpotqa")
declare -a STRATEGIES=("parallel-rrm")
declare -a BUDGETS=(4096 8192)
declare -a N_SAMPLES=(4 8 16 32)

for dataset in "${DATASETS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      for n_samples in "${N_SAMPLES[@]}"; do

        # --- LOGIC FOR SETTING RUN ARGUMENTS ---
        EXTRA_ARGS=""
        if [ "$DEBUG_MODE_ENABLED" == "true" ]; then
          # If debug is on, use the debug flags
          echo "    >> RUNNING IN DEBUG MODE"
          EXTRA_ARGS="--debug_mode --debug_sample_size 10"
        else
          # If debug is off, use the full dataset limits
          echo "    >> RUNNING IN FULL MODE"
          if [ "$dataset" == "gsm8k" ]; then
            EXTRA_ARGS="--limit 500"
          elif [ "$dataset" == "math_500" ]; then
            EXTRA_ARGS="--limit 500"
          elif [ "$dataset" == "hotpotqa" ]; then
            EXTRA_ARGS="--limit 500"
          elif [ "$dataset" == "musique" ]; then
            EXTRA_ARGS="--limit 500"
          elif [ "$dataset" == "2wikimultihopqa" ]; then
            EXTRA_ARGS="--limit 500"
          fi
        fi

        # Create a unique name for the run for logging and results
        RUN_NAME="${dataset}_${strategy}_b${budget}_n${n_samples}"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        
        # Define output and log directories
        OUTPUT_DIR="$PROJECT_ROOT/results/$RUN_NAME/$TIMESTAMP"
        LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

        echo "--> Launching run: $RUN_NAME with args: ${EXTRA_ARGS}"
        echo "    Logs will be saved to: $LOG_FILE"
        echo "    Results will be saved to: $OUTPUT_DIR"
        
        # --- Simplified Execution Command ---
        # We no longer need 'env' or the proxy variables here.
        # nohup python3 "$PROJECT_ROOT/$PYTHON_SCRIPT" \
        #     --dataset "$dataset" \
        #     --strategy "$strategy" \
        #     --compute_budget "$budget" \
        #     --n_samples "$n_samples" \
        #     --model_cache_path "$MODEL_CACHE_PATH" \
        #     --model_name "$MODEL_NAME" \
        #     --base_path "$PROJECT_ROOT" \
        #     --output_dir "$OUTPUT_DIR" \
        #     $EXTRA_ARGS \
        #     > "$LOG_FILE" 2>&1 &
        # --- THE FIX IS HERE ---
        # We REMOVED 'nohup' and the trailing '&' to run commands one by one.
        python3 "$PROJECT_ROOT/$PYTHON_SCRIPT" \
            --dataset "$dataset" \
            --strategy "$strategy" \
            --compute_budget "$budget" \
            --n_samples "$n_samples" \
            --model_cache_path "$MODEL_CACHE_PATH" \
            --model_name "$MODEL_NAME" \
            --base_path "$PROJECT_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            $EXTRA_ARGS \
            > "$LOG_FILE" 2>&1
        
        echo "--- Finished Run: $RUN_NAME ---"
        # echo "-----------------------------------------------------"
        # PID=$!
        # echo "    Process started with PID: $PID. Monitor with: tail -f $LOG_FILE"
        echo "-----------------------------------------------------"
        # Optional: add a small delay to avoid overwhelming the system
        # sleep 2
      done
    done
  done
done

echo "All experiments have been launched in the background."
echo "Monitor progress via the log files in the '$PROJECT_ROOT/logs/' directory."