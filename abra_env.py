# abra_env.py

import random
from typing import List, Dict, Tuple, Any
# 假设您的 gsm 模块和 utils 模块可以被导入
from src.gsm.task_init import GSMInit
from src.gsm.feedback import GSMFeedback
from src.utils import retry_parse_fail_prone_cmd # 您提供的代码中使用了这个

# 假设我们有一个函数来评估最终答案的正确性并计算成本
def evaluate_solution_and_cost(solution_text: str, ground_truth_answer: Any) -> Tuple[float, float]:
    """
    评估解决方案的正确性并估算成本。
    返回: (奖励, 成本)
    奖励：例如，正确为1.0，错误为0.0 或 -1.0
    成本：例如，消耗的token数量或API调用次数
    """
    # 这是一个占位符，您需要根据实际情况实现
    # 例如，从 solution_text 中提取答案，与 ground_truth_answer 比较
    # 并估算生成 solution_text 的成本
    is_correct = False # 假设默认是错误的
    cost = len(solution_text.split()) # 简单地用词数作为成本示例

    # 实际的答案提取和比较逻辑会更复杂
    # 比如:
    # try:
    #     extracted_ans = float(solution_text.split("The answer is ")[-1].replace("$.", "").replace("$", "").replace(",", ""))
    #     if abs(extracted_ans - float(ground_truth_answer)) < 1e-3:
    #         is_correct = True
    # except:
    #     pass # 解析失败或类型不匹配

    reward = 1.0 if is_correct else 0.0
    return reward, float(cost)

class ABRAEnv:
    def __init__(self, problem_data: Dict, m_gen: GSMInit, m_edit_feedback: GSMFeedback, m_eval, total_budget: float, max_iterations: int):
        """
        初始化ABRA环境。
        :param problem_data: 当前要解决的问题，应包含 'input' (问题描述) 和 'target_answer' (标准答案)
        :param m_gen: 用于初始生成的LLM封装 (task_init)
        :param m_edit_feedback: 用于生成反馈和修正的LLM封装 (task_feedback)
        :param m_eval: 用于评估解决方案的LLM或函数 (这里暂时简化，直接用evaluate_solution_and_cost)
        :param total_budget: 总计算预算 (例如，最大token数)
        :param max_iterations: 单个问题解决的最大迭代次数 (对应ABRA算法中的 i)
        """
        self.problem_input = problem_data["input"]
        self.ground_truth_answer = problem_data.get("target_answer", None) # 有些数据集可能没有直接的目标答案用于训练期奖励

        self.m_gen = m_gen
        self.m_edit_feedback = m_edit_feedback
        self.m_eval = m_eval # 注意：在您的代码中，m_eval 的角色被 task_feedback 和最终答案检查分担了

        self.total_budget = total_budget
        self.max_iterations = max_iterations

        # 状态相关变量
        self.current_iteration = 0
        self.remaining_budget = total_budget
        self.solution_set_S = [] # 存储所有尝试过的解决方案路径及其评估 (P_text, score, feedback_text)
        self.current_best_solution = None
        self.current_best_solution_score = -float('inf') # 或其他初始劣质分数

        # 动作空间 (简化版)
        # 0: ContinueRefinement (如果已有解，则修正；如果没有，则初始生成)
        # 1: TerminateAndSelect
        self.action_space_n = 2

        self.current_solution_being_refined = None # 当前正在被修正的解（来自 iterative_gsm 的 solution）
        self.log_for_current_problem = [] # 记录 iterative_gsm 的日志


    def _get_state(self) -> Dict[str, Any]:
        """
        构建并返回当前状态给 RL agent。
        这里需要将 S, D_Q, A_ability, B_rem 等信息特征化。
        这是一个简化的示例，实际需要更复杂的特征工程。
        """
        # 简单示例：
        # LLM能力/当前解的质量：最好解的分数，S中解的数量，平均反馈长度等
        num_solutions = len(self.solution_set_S)
        max_score_in_S = self.current_best_solution_score if self.current_best_solution else 0.0
        
        # 查询难度：可以基于问题长度，或一个简单的分类器，这里简化
        query_difficulty_feature = len(self.problem_input.split()) / 100.0 # 简单示例

        # 剩余预算比例
        budget_ratio = self.remaining_budget / self.total_budget if self.total_budget > 0 else 0

        # 当前迭代次数比例
        iteration_ratio = self.current_iteration / self.max_iterations if self.max_iterations > 0 else 0
        
        # 当前是否有解正在被修正
        is_refining = 1.0 if self.current_solution_being_refined else 0.0

        state = {
            "num_solutions": num_solutions,
            "max_score": max_score_in_S,
            "query_difficulty": query_difficulty_feature,
            "budget_ratio": budget_ratio,
            "iteration_ratio": iteration_ratio,
            "is_refining_active": is_refining,
            # 可以加入更多特征，比如最近几次反馈的情感、修正的幅度等
        }
        # 实际中，这些特征需要被转换成一个固定长度的向量
        return state

    def reset(self) -> Dict[str, Any]:
        """
        重置环境到一个新的开始状态（通常是新问题的开始）。
        """
        self.current_iteration = 0
        self.remaining_budget = self.total_budget
        self.solution_set_S = []
        self.current_best_solution = None
        self.current_best_solution_score = -float('inf')
        self.current_solution_being_refined = None
        self.log_for_current_problem = []
        return self._get_state()

    @retry_parse_fail_prone_cmd # 沿用您代码中的装饰器
    def _execute_one_refinement_step(self):
        """
        执行一步迭代修正，模拟 iterative_gsm 中的一轮循环。
        """
        cost_this_step = 0
        feedback_text = ""
        newly_generated_solution = ""

        if not self.current_solution_being_refined: # 如果还没有初始解
            # 调用 task_init 生成初始解
            initial_solution_text = self.m_gen(solution=self.problem_input) # solution 参数可能需要调整为 question
            cost_this_step += len(initial_solution_text.split()) # 估算成本
            self.current_solution_being_refined = initial_solution_text
            newly_generated_solution = initial_solution_text
            # 将初始解加入S，并评估
            # (在SCoRe中，第一轮的评估是后续进行的，这里为了简化，可以先不评估或给个默认值)
            # 我们这里假设初始解也需要一次评估和反馈才能进入下一轮修正
            # 或者，我们可以认为初始生成后，下一步动作是获取反馈
            self.log_for_current_problem.append({
                "attempt": self.current_iteration, # 或者用一个内部的修正步骤计数器
                "solution_curr": "N/A (Initial Generation)", # 标记这是初始生成
                "solution_fixed": initial_solution_text,
                "feedback": "Initial generation, awaiting feedback."
            })
        
        # 获取反馈并尝试修正 (对应 iterative_gsm 中的核心循环)
        # 注意：您的 iterative_gsm 是一次性跑完的，这里我们要把它拆成一步步
        # 我们假设每一步 refine 都包含 "获取反馈" 和 "根据反馈生成新解"
        if not self.current_solution_being_refined: # 如果上面初始生成失败或没有
            # 这种情况不应该发生，如果发生了，则认为无法继续
            return None, "Error: No solution to refine.", 0 

        fb_and_maybe_soln = self.m_edit_feedback(solution=self.current_solution_being_refined)
        
        feedback_text = fb_and_maybe_soln["feedback"]
        cost_this_step += len(feedback_text.split()) # 反馈生成的成本

        if "it is correct" in feedback_text.lower():
            # 如果反馈说已经正确了，那么下一轮动作应该是 Terminate
            # 但这里我们还是先记录下这个“正确的”解
            newly_generated_solution = self.current_solution_being_refined # 保持不变
            # 可以在这里给一个小的“已解决”信号给RL agent的状态
        else:
            newly_generated_solution = fb_and_maybe_soln["solution"]
            cost_this_step += len(newly_generated_solution.split()) # 修正生成的成本
        
        # 更新日志和当前待修正的解
        self.log_for_current_problem.append({
            "attempt": self.current_iteration,
            "solution_curr": self.current_solution_being_refined,
            "solution_fixed": newly_generated_solution,
            "feedback": feedback_text
        })
        self.current_solution_being_refined = newly_generated_solution
        
        return newly_generated_solution, feedback_text, cost_this_step


    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        执行一个动作，返回 (next_state, reward, done, info)
        """
        if self.remaining_budget <= 0 or self.current_iteration >= self.max_iterations:
            # 如果预算耗尽或达到最大迭代次数，强制终止
            action = 1 # TerminateAndSelect

        reward = 0.0  # 中间步骤的奖励通常为0，除非有特殊设计
        done = False
        info = {} # 可以用来存储额外信息

        if action == 0: # ContinueRefinement
            solution_after_refinement, feedback, cost_this_step = self._execute_one_refinement_step()
            self.remaining_budget -= cost_this_step

            if solution_after_refinement is None: # 比如发生错误无法继续
                done = True
                reward = -1.0 # 惩罚错误
                info['status'] = 'Error in refinement'
            else:
                # 评估新生成的解 (这里简化，只看是否在feedback中被标记为correct)
                # 实际中，可以调用 M_eval 对 solution_after_refinement 打分
                temp_reward, _ = evaluate_solution_and_cost(solution_after_refinement, self.ground_truth_answer)
                
                # 更新S集合 (简单起见，只保留最新的解和其评估)
                # 更复杂的S可以存储历史路径和评分
                if not self.solution_set_S or temp_reward > self.current_best_solution_score:
                    self.current_best_solution = solution_after_refinement
                    self.current_best_solution_score = temp_reward
                
                self.solution_set_S.append({
                    "path": solution_after_refinement, 
                    "score": temp_reward, # 假设的评估分数
                    "feedback_received": feedback
                })

                if "it is correct" in feedback.lower() or self.current_iteration + 1 >= self.max_iterations:
                    # 如果反馈说正确，或者达到最大迭代，下一状态应该倾向于终止
                    # 但最终决策由RL agent在下一个状态做出
                    # 这里可以设置一个标志，或者在状态表示中体现出来
                    pass 
                if self.remaining_budget <= 0:
                    done = True # 预算耗尽
                    # 最终奖励在 done=True 时计算
                    final_reward_val, _ = evaluate_solution_and_cost(self.current_best_solution if self.current_best_solution else "", self.ground_truth_answer)
                    reward = final_reward_val - (self.total_budget - self.remaining_budget) / self.total_budget * 0.1 # 简单惩罚预算消耗
                    info['status'] = 'Budget depleted'
        
        elif action == 1: # TerminateAndSelect
            done = True
            if not self.solution_set_S: # 如果没有任何解就终止了
                final_solution = ""
                reward = -0.5 # 或者更大的惩罚
                info['status'] = 'Terminated without solution'
            else:
                # 从 S 中选择最终答案，这里简化为选择当前记录的最好解
                # 实际可以实现更复杂的选择逻辑，比如基于M_eval的最终评估
                final_solution = self.current_best_solution if self.current_best_solution else self.solution_set_S[-1]["path"] # fallback
            
            final_reward_val, _ = evaluate_solution_and_cost(final_solution, self.ground_truth_answer)
            # 奖励 = 最终答案的正确性 - (一定比例的 * 已消耗预算 / 总预算)
            # 这里的 0.1 是一个例子，表示对预算消耗的惩罚权重，可以调整
            reward = final_reward_val - (self.total_budget - self.remaining_budget) / self.total_budget * 0.1
            info['status'] = 'Terminated by agent'
            info['final_solution'] = final_solution

        self.current_iteration += 1
        if self.current_iteration >= self.max_iterations and not done:
            done = True
            # 如果达到最大迭代次数，也需要计算最终奖励
            final_solution = self.current_best_solution if self.current_best_solution else (self.solution_set_S[-1]["path"] if self.solution_set_S else "")
            final_reward_val, _ = evaluate_solution_and_cost(final_solution, self.ground_truth_answer)
            reward = final_reward_val - (self.total_budget - self.remaining_budget) / self.total_budget * 0.1
            info['status'] = 'Max iterations reached'
            info['final_solution'] = final_solution
            
        next_state = self._get_state()
        return next_state, reward, done, info