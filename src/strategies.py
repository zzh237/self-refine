# src/strategies.py

# import logging
# from abc import ABC, abstractmethod
# from pathlib import Path
# from collections import Counter
# import json

# from src.backends.llm_interface import LlamaInterface
# from src.prompt_utils import load_prompt_from_file, format_prompt

# logger = logging.getLogger(__name__)

# class BaseStrategy(ABC):
#     """Abstract base class for an inference strategy."""
#     def __init__(self, llm: LlamaInterface, prompt_dir: Path, dataset_name: str, budget: int, n_samples: int):
#         self.llm = llm
#         self.prompt_dir = prompt_dir
#         self.dataset_name = dataset_name
#         self.budget = budget
#         self.n_samples = n_samples
#         self.tokens_per_step = budget // n_samples
#         logger.info(f"Strategy initialized: Budget={budget}, N={n_samples}, Tokens per step={self.tokens_per_step}")

#     @abstractmethod
#     def generate(self, item: dict) -> dict:
#         """
#         Generates a final answer for a given data item using the specific strategy.

#         Args:
#             item (dict): A single data item from the dataset.

#         Returns:
#             dict: A JSON-like dictionary representing the final answer.
#         """
#         pass

# class ParallelStrategy(BaseStrategy):
#     """
#     Implements the parallel generation strategy (Best-of-N with majority voting).
#     """
#     def generate(self, item: dict) -> dict:
#         prompt_template = load_prompt_from_file(self.prompt_dir / f"{self.dataset_name}_parallel.txt")
#         prompt = format_prompt(prompt_template, item, self.dataset_name)
        
#         logger.info(f"Generating {self.n_samples} parallel responses...")
#         responses = self.llm.generate_parallel_responses(prompt, n=self.n_samples, max_tokens=self.tokens_per_step)
        
#         # Perform majority voting on the 'answer' field
#         answers = [resp.get('answer') for resp in responses if resp and 'answer' in resp and resp.get('answer')]
#         if not answers:
#             logger.warning("No valid answers found in parallel responses.")
#             return {"answer": "Error: No valid answer generated.", "supporting_facts": []}
            
#         # Use Counter to find the most common answer
#         answer_counts = Counter(answers)
#         most_common_answer = answer_counts.most_common(1)
        
#         logger.info(f"Majority vote winner: '{most_common_answer}'")
        
#         # For simplicity, we return the majority answer. We don't aggregate supporting facts.
#         return {"answer": most_common_answer, "supporting_facts":[]}


# class SequentialStrategy(BaseStrategy):
#     """
#     Implements the sequential refinement strategy.
#     """
#     def generate(self, item: dict) -> dict:
#         step1_template = load_prompt_from_file(self.prompt_dir / f"{self.dataset_name}_sequential_step1.txt")
#         refine_template = load_prompt_from_file(self.prompt_dir / f"{self.dataset_name}_sequential_refine.txt")

#         # Step 1: Initial Generation
#         logger.info("Generating initial response (Step 1)...")
#         prompt = format_prompt(step1_template, item, self.dataset_name)
#         current_attempt_json = self.llm.generate_structured_response(prompt, max_tokens=self.tokens_per_step)
        
#         # Refinement Loop
#         for i in range(1, self.n_samples):
#             logger.info(f"Performing refinement step {i+1}/{self.n_samples}...")
#             if "error" in current_attempt_json:
#                 logger.error(f"Cannot refine due to error in previous step: {current_attempt_json}")
#                 return current_attempt_json # Propagate the error

#             previous_attempt_str = json.dumps(current_attempt_json, indent=2)
#             prompt = format_prompt(refine_template, item, self.dataset_name, previous_attempt=previous_attempt_str)
            
#             current_attempt_json = self.llm.generate_structured_response(prompt, max_tokens=self.tokens_per_step)

#         logger.info(f"Final response after {self.n_samples} steps: {current_attempt_json}")
#         return current_attempt_json


# def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
#     """Factory function to get an instance of a strategy."""
#     if strategy_name == 'parallel':
#         return ParallelStrategy(**kwargs)
#     elif strategy_name == 'sequential':
#         return SequentialStrategy(**kwargs)
#     else:
#         raise ValueError(f"Unknown strategy: {strategy_name}")


# src/strategies.py
# src/strategies.py

import logging
import re
import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from collections import Counter
from src.backends.llm_interface import LlamaInterface
from src.prompt_utils import load_prompt_templates_from_json, format_prompt
# from src.evaluation import parse_llm_answer  # Use the parser we created
from src.parsing_utils import parse_llm_json_answer, parse_llm_string_answer

logger = logging.getLogger(__name__)





class BaseStrategy(ABC):
    """Abstract base class for an inference strategy."""
    def __init__(self, llm: LlamaInterface, prompt_dir: Path, dataset_name: str, **kwargs):
        self.llm = llm
        self.dataset_name = dataset_name
        self.config = kwargs
        prompt_file = prompt_dir / f"{self.dataset_name}.json"
        self.prompts = load_prompt_templates_from_json(prompt_file)
        self.system_prompt = self.prompts.get("system_prompt", "You are a helpful assistant.")
        logger.info(f"Strategy for '{dataset_name}' initialized.")

    @abstractmethod
    def generate(self, item: dict) -> dict:
        """Generates a final answer dictionary for a given data item."""
        pass

class ParallelStrategy(BaseStrategy):
    """Implements Best-of-N with majority voting."""
    def generate(self, item: dict) -> dict:
        n_samples = self.config.get('n_samples', 1)
        temperature = self.config.get('temperature', 0.5)
        max_tokens = self.config.get('compute_budget', 1024)

        user_prompt_template = self.prompts["parallel_user_prompt"]
        user_prompt = format_prompt(user_prompt_template, item)
        full_prompt = self.llm.create_prompt(self.system_prompt, user_prompt)
        
        logger.info(f"Generating {n_samples} parallel responses...")
        # Call the new, simpler generate method
        raw_responses = self.llm.generate(full_prompt, temperature, max_tokens, n=n_samples)
        
        # # --- MAJORITY VOTE LOGIC ---
        # parsed_answers = [parse_llm_answer(resp, self.dataset_name) for resp in raw_responses]
        # valid_answers = [ans for ans in parsed_answers if ans]

        # if not valid_answers:
        #     logger.warning("No valid answers found in parallel responses.")
        #     return {"answer": "Error: No valid answer generated."}
            
        # answer_counts = Counter(valid_answers)
        # winner_answer = answer_counts.most_common(1)[0][0]
        
        # logger.info(f"Majority vote winner: '{winner_answer}'")
        # return {"answer": winner_answer}
        if self.dataset_name in ['hotpotqa', 'musique', '2wikimultihopqa']:
            # For JSON datasets, we vote on the 'answer' field
            parsed_responses = [parse_llm_json_answer(resp) for resp in raw_responses]
            valid_responses = [r for r in parsed_responses if r and r.get('answer')]
            
            if not valid_responses:
                return {"answer": "Error: No valid JSON answers."}
            
            answer_strings = [r['answer'] for r in valid_responses]
            winner_answer_str = Counter(answer_strings).most_common(1)[0][0]
            
            # Return the full JSON object of the winner
            final_answer_obj = next(r for r in valid_responses if r['answer'] == winner_answer_str)
            logger.info(f"Majority vote winner: '{winner_answer_str}'")
            return final_answer_obj
        else:
            # For string-based datasets (gsm8k, math), we vote on the parsed string
            parsed_answers = [parse_llm_string_answer(resp, self.dataset_name) for resp in raw_responses]
            valid_answers = [ans for ans in parsed_answers if ans]

            if not valid_answers:
                return {"answer": "Error: No valid answers found."}
            
            winner_answer_str = Counter(valid_answers).most_common(1)[0][0]
            logger.info(f"Majority vote winner: '{winner_answer_str}'")
            return {"answer": winner_answer_str}

class SequentialStrategy(BaseStrategy):
    """Implements sequential refinement."""
    def generate(self, item: dict) -> dict:
        n_samples = self.config.get('n_samples', 1)
        temperature = self.config.get('temperature', 0.0)
        max_tokens_per_step = self.config.get('compute_budget', 1024) // n_samples

        initial_user_template = self.prompts["parallel_user_prompt"]
        refine_user_template = self.prompts["sequential_user_prompt_refine"]

        previous_answer_raw = ""

        logger.info(f"Executing SequentialStrategy with {n_samples} steps...")
        for i in range(n_samples):
            if i == 0:
                user_prompt = format_prompt(initial_user_template, item)
            else:
                user_prompt = format_prompt(refine_user_template, item, previous_answer=previous_answer_raw)
            
            full_prompt = self.llm.create_prompt(self.system_prompt, user_prompt)
            # Generate a single response for this step
            current_generation_raw = self.llm.generate(full_prompt, temperature, max_tokens_per_step, n=1)[0]
            previous_answer_raw = current_generation_raw
        
        logger.info("Sequential refinement finished. Parsing final answer.")
        # final_parsed_answer = parse_llm_answer(previous_answer_raw, self.dataset_name)
        # return {"answer": final_parsed_answer}
        # Use the correct parser based on dataset type
        if self.dataset_name in ['hotpotqa', 'musique', '2wikimultihopqa']:
            final_answer_obj = parse_llm_json_answer(previous_answer_raw)
            return final_answer_obj if final_answer_obj else {"answer": "Error: Failed to parse final JSON."}
        else:
            final_parsed_answer = parse_llm_string_answer(previous_answer_raw, self.dataset_name)
            return {"answer": final_parsed_answer}
        
class ParallelRRMStrategy(BaseStrategy):
    """
    Implements Best-of-N using an LLM as a judge in a knockout tournament.
    """
    def __init__(self, llm: LlamaInterface, prompt_dir: Path, dataset_name: str, **kwargs):
        super().__init__(llm, prompt_dir, dataset_name, **kwargs)
        
        # --- THE FIX IS HERE: Load the correct judge prompt based on the dataset ---
        if self.dataset_name in ['hotpotqa', 'musique', '2wikimultihopqa']:
            judge_prompt_file = prompt_dir / "rrm_judge.json"
        else: # For gsm8k, math_500, etc.
            judge_prompt_file = prompt_dir / "rrm_judge_math.json"
        
        logger.info(f"RRM Strategy loading judge prompt from: {judge_prompt_file}")
        self.judge_prompts = load_prompt_templates_from_json(judge_prompt_file)
        # --- END OF FIX ---

    def _judge_pair(self, item: dict, answer_a: str, answer_b: str) -> str:
        """Uses the LLM to judge which of two answers is better."""
        system_prompt = self.judge_prompts["system_prompt"]
        user_template = self.judge_prompts["user_prompt"]
        
        # The format call will now work because the math template doesn't expect a {context}
        user_prompt = format_prompt(user_template, item, answer_a=answer_a, answer_b=answer_b)
        full_prompt = self.llm.create_prompt(system_prompt, user_prompt)
        
        judgment_text = self.llm.generate(full_prompt, temperature=0.0, max_tokens=512, n=1)[0]
        
        if "Winner: [[B]]" in judgment_text:
            return answer_b
        return answer_a

    def generate(self, item: dict) -> dict:
        # ... (The rest of this function remains exactly the same)
        n_samples = self.config.get('n_samples', 1)
        temperature = self.config.get('temperature', 0.5)
        max_tokens = self.config.get('compute_budget', 1024)

        user_prompt_template = self.prompts["parallel_user_prompt"]
        user_prompt = format_prompt(user_prompt_template, item)
        full_prompt = self.llm.create_prompt(self.system_prompt, user_prompt)
        
        candidates = self.llm.generate(full_prompt, temperature, max_tokens, n=n_samples)
        
        tournament_round = list(candidates)
        round_num = 1
        
        while len(tournament_round) > 1:
            logger.info(f"RRM Tournament Round {round_num}: {len(tournament_round)} candidates remaining.")
            winners = []
            random.shuffle(tournament_round)
            
            for i in range(0, len(tournament_round), 2):
                if i + 1 < len(tournament_round):
                    winner = self._judge_pair(item, tournament_round[i], tournament_round[i+1])
                    winners.append(winner)
                else:
                    winners.append(tournament_round[i])
            
            tournament_round = winners
            round_num += 1

        final_raw_response = tournament_round[0]
        final_prediction = {}
        
        if self.dataset_name in ['hotpotqa', 'musique', '2wikimultihopqa']:
            final_prediction = parse_llm_json_answer(final_raw_response) or {"answer": "Error: RRM winner was unparsable."}
        else:
            final_parsed_answer = parse_llm_string_answer(final_raw_response, self.dataset_name)
            final_prediction = {"answer": final_parsed_answer}
            
        return {"prediction": final_prediction, "raw_responses": candidates}





def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """Factory function to get an instance of a strategy."""
    if strategy_name.lower() == 'parallel':
        return ParallelStrategy(**kwargs)
    elif strategy_name.lower() == 'sequential':
        return SequentialStrategy(**kwargs)
    elif strategy_name == 'parallel-rrm': 
        return ParallelRRMStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")