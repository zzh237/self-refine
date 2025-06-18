import pandas as pd
# from prompt_lib.backends import openai_api
from ..backends.local_llm_api import LocalHFAPIWrapper
# from src.utils import Prompt
from utils import Prompt, retry_parse_fail_prone_cmd,ENGINE_PATH


class GSMFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 600) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###\n\n",
            engine = engine,
            temperature = temperature
        )
        
        self.max_tokens = max_tokens
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good." if "naive" not in prompt_examples else "# There is an error in the code above."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def __call__(self, solution: str):
        generation_query = self.make_query(solution=solution)
        print(generation_query)
        # print(1/0)
        output = LocalHFAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END",
            temperature=self.temperature,
        )
        
        
        entire_output = LocalHFAPIWrapper.get_first_response(output)
        print(entire_output)
        if "### END" in entire_output:
            entire_output = entire_output.split("### END")[0]

        improved_soln = entire_output.split("def solution():")[1]
        feedback = entire_output.split("def solution():")[0]
        improved_soln = "def solution():" + improved_soln.rstrip()
        self.update_prompt(solution=solution, improved_soln=improved_soln, feedback=feedback)
        return {"solution": improved_soln, "feedback": feedback}

    def make_query(self, solution: str):
        
        solution = f"""{self.question_prefix}{solution}{self.intra_example_sep}{self.instruction}{self.answer_prefix}"""
        return f"{self.prompt}{solution}"
    
    
    def update_prompt(self, solution: str, improved_soln: str, feedback: str):
        prefix = f"""{self.question_prefix}{solution}{self.intra_example_sep}{self.instruction}{self.answer_prefix}"""
        
        gen_ans = f"""

{feedback}

{improved_soln.rstrip()}{self.inter_example_sep}"""

        new_example = f"{prefix}{gen_ans}"
        self.prompt = f"{self.prompt}{new_example}"
    

def test():
    task_fb = GSMFeedback(
        prompt_examples="data/prompt/gsm/pal/feedback.txt",
        engine=ENGINE_PATH,
        temperature=0.7,
    )

    wrong_soln = """def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = (plates * plate_cost) / cups - 1200
    result = cup_cost
    return result"""
    feedback_and_solution = task_fb(wrong_soln)
    print(feedback_and_solution["feedback"])
    print(feedback_and_solution["solution"])
    
    print(task_fb.prompt)
    

if __name__ == '__main__':
    test()