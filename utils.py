# utils.py
import json
import torch
import numpy as np
import random
from typing import List, Dict, Any

def set_seed(seed: int):
    """设置随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset(file_path: str) -> List[Dict]:
    """
    从jsonl文件加载数据集。
    每行是一个JSON对象，期望包含 'input' 和 'target_answer' (或其他你需要的字段)。
    """
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"警告: 无法解析行: {line.strip()}")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到 {file_path}")
        # 为了能跑起来，返回一个虚拟数据
        print("警告: 使用虚拟数据集")
        dataset = [
            {"input": "Natalia sold clips to 48 of her friends at school. Later that day she sold 52 more. How many clips did she sell in total?", "target_answer": 100.0, "id": "q1"},
            {"input": "A train travels at 60 km/h for 2.5 hours. How far does it travel?", "target_answer": 150.0, "id": "q2"}
        ] * 10 # 复制一些数据
    return dataset


def featurize_state_for_agent(state_dict: Dict[str, Any], device: str = "cpu") -> torch.Tensor:
    """
    将 ABRAEnv._get_state() 返回的字典状态转换为一个PyTorch张量。
    顺序需要与 ABRAAgent 初始化时的 state_dim 对应。
    """
    # 确保这里的顺序和维度与 config.STATE_DIM_PLACEHOLDER 和 ABRAAgent 的输入一致
    # "num_solutions", "max_score", "query_difficulty", "budget_ratio", "iteration_ratio", "is_refining_active"
    features = [
        float(state_dict.get("num_solutions", 0)),
        float(state_dict.get("max_score", 0.0)),
        float(state_dict.get("query_difficulty", 0.0)),
        float(state_dict.get("budget_ratio", 0.0)),
        float(state_dict.get("iteration_ratio", 0.0)),
        float(state_dict.get("is_refining_active", 0.0))
    ]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # 添加batch维度

# 你的 retry_parse_fail_prone_cmd 装饰器
# (从你提供的代码中复制过来，或者如果它在你的 src.utils 中，确保可以导入)
def retry_parse_fail_prone_cmd(func):
    def wrapper(*args, **kwargs):
        # 这是一个占位符，你需要实现你的重试逻辑
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"函数 {func.__name__} 执行失败: {e}, 将尝试重试 (这里是占位符，未实现重试)")
            # raise e # 或者根据你的逻辑进行重试
            # 为了能跑通，我们这里先简单地重新抛出异常，或者返回一个表示失败的值
            if func.__name__ == "_execute_one_refinement_step": # 特殊处理
                return None, f"Error in func {func.__name__}: {e}", 0
            return None # 对于其他函数
    return wrapper

# --- 占位符/模拟 LLM 调用 ---
# 为了让代码能直接跑起来，我们创建非常简单的模拟LLM类
class MockGSMInit:
    def __init__(self, engine, prompt_examples, temperature):
        print(f"MockGSMInit: 使用引擎 {engine}, 提示 {prompt_examples}, 温度 {temperature}")
    
    @retry_parse_fail_prone_cmd
    def __call__(self, solution: str, **kwargs) -> str: # solution 参数应为 question
        question = solution # 假设输入的是问题
        print(f"MockGSMInit: 收到问题 '{question[:30]}...'")
        # 模拟生成一个非常简单的初始答案
        if "Natalia" in question:
            return "Natalia sold 48 + 52 = 90 clips. The answer is $\boxed{90}$." # 故意给个错的初始解
        elif "train" in question:
            return "The train traveled 60 * 2 = 120 km. The answer is $\boxed{120}$."
        return f"Initial solution for '{question[:20]}...' The answer is $\boxed{0}$."

class MockGSMFeedback:
    def __init__(self, engine, prompt_examples, temperature):
        print(f"MockGSMFeedback: 使用引擎 {engine}, 提示 {prompt_examples}, 温度 {temperature}")
        self.attempt_count = 0

    @retry_parse_fail_prone_cmd
    def __call__(self, solution: str, **kwargs) -> Dict[str, str]:
        print(f"MockGSMFeedback: 收到待评估/修正的方案 '{solution[:50]}...'")
        self.attempt_count +=1
        feedback = f"Feedback attempt {self.attempt_count}: "
        fixed_solution = solution # 默认不修改

        if "90 clips" in solution and "Natalia" in solution:
            feedback += "The sum 48+52 is not 90. Please recheck."
            fixed_solution = "Natalia sold 48 + 52 = 100 clips. The answer is $\boxed{100}$."
        elif "120 km" in solution and "train" in solution:
            feedback += "The calculation 60*2 is incorrect for 2.5 hours."
            fixed_solution = "The train traveled 60 * 2.5 = 150 km. The answer is $\boxed{150}$."
        elif self.attempt_count >= 2: # 模拟经过几轮修正后认为正确
             feedback += "It looks correct now."
        else:
            feedback += "Please check your calculations carefully."
            # 简单模拟修正，比如在答案后面加一个修正标记
            fixed_solution += " (revised)"
            
        return {"feedback": feedback, "solution": fixed_solution}

def mock_evaluate_solution_and_cost(solution_text: str, ground_truth_answer: Any) -> Tuple[float, float]:
    """模拟评估解决方案的正确性并估算成本。"""
    is_correct = False
    cost = len(solution_text.split()) if solution_text else 0

    # 简化答案提取和比较逻辑
    try:
        # 尝试从 \boxed{} 中提取答案
        start_idx = solution_text.rfind("$\boxed{")
        if start_idx != -1:
            end_idx = solution_text.rfind("}$", start_idx)
            if end_idx != -1:
                extracted_ans_str = solution_text[start_idx + len("$\boxed{"):end_idx].strip()
                # 假设答案是浮点数
                extracted_ans = float(extracted_ans_str)
                if ground_truth_answer is not None and abs(extracted_ans - float(ground_truth_answer)) < 1e-3:
                    is_correct = True
    except ValueError: # float转换失败
        pass
    except Exception: # 其他提取错误
        pass
        
    reward = 1.0 if is_correct else 0.0
    # print(f"MockEval: Sol='{solution_text[:30]}...', GT='{ground_truth_answer}', Extracted='{extracted_ans_str if 'extracted_ans_str' in locals() else 'N/A'}', Correct={is_correct}, Reward={reward}")
    return reward, float(cost)