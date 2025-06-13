#!/bin/bash

# ==============================================================================
#  超精简版 run_self_refine_test.sh
#  直接在此脚本中指定要使用的本地模型路径，并运行 src/gsm/run.py test
# ==============================================================================

# --- 在这里配置你要用于测试的本地模型路径 ---
# 请将下面的路径替换成你实际的模型绝对路径
# 例如: "/data/ebay-lvs-a100/notebooks/zzhang12/model_cache/mistralai--Mistral-7B-Instruct-v0.3"
# 如果你希望每次测试都用不同的模型，你仍然需要修改这个脚本。
# 或者，你可以取消注释下面的行，并从第一个命令行参数获取模型路径：
# MY_LOCAL_MODEL_PATH="${1:-/path/to/your/default_model_if_no_arg}" 
# if [ -z "$1" ]; then
#     echo "提示: 未提供模型路径作为第一个参数，将使用脚本内定义的默认路径。"
#     echo "用法示例: bash scripts/run_self_refine_test.sh /path/to/your/model"
# fi

MY_LOCAL_MODEL_PATH="/data/ebay-lvs-a100/notebooks/zzhang12/model_cache/mistralai--Mistral-7B-Instruct-v0.3" # <--- 请修改这里！

# --- 检查模型路径是否存在 (可选但推荐) ---
if [ ! -d "$MY_LOCAL_MODEL_PATH" ]; then
    echo "错误: 在脚本中指定的模型路径 '$MY_LOCAL_MODEL_PATH' 不存在或不是一个目录。"
    echo "请编辑此脚本，并设置正确的 MY_LOCAL_MODEL_PATH。"
    exit 1
fi

# --- 设置项目路径 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# --- 设置PYTHONPATH ---
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# --- 进入项目根目录 ---
cd "$PROJECT_ROOT" || { echo "错误: 无法进入项目根目录 $PROJECT_ROOT"; exit 1; }

# --- 执行命令 ---
echo "项目根目录: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "将使用模型: $MY_LOCAL_MODEL_PATH"
echo "正在执行: python src/gsm/run.py test --engine_path \"$MY_LOCAL_MODEL_PATH\""
echo "-----------------------------------------------------"

python src/gsm/run.py test --engine_path "$MY_LOCAL_MODEL_PATH"

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "Python脚本成功执行完毕。"
else
    echo "错误: Python脚本执行失败。"
fi
echo "-----------------------------------------------------"
echo "测试脚本执行结束。"