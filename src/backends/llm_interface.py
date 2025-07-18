# src/llm_interface.py

# import json
# import logging
# from vllm import LLM
# from vllm import SamplingParams # From your existing setup



# logger = logging.getLogger(__name__)

# class LlamaInterface:
#     """
#     A wrapper class for interacting with a vLLM-powered language model.
#     This class handles model initialization, prompt formatting, and response generation.
#     """
#     def __init__(self, model_path: str, temp: float, tp_size: int, trust_remote_code: bool = True):
#         """
#         Initializes the LLM and tokenizer using vLLM.

#         Args:
#             model_path (str): The path to the locally cached model weights.
#             temp (float): The default sampling temperature.
#             tp_size (int): The tensor parallel size for vLLM.
#             trust_remote_code (bool): Whether to trust remote code when loading the model.
#         """
#         logger.info(f"Initializing vLLM with model from: {model_path}")
#         self.llm = LLM(
#             model=model_path,
#             tensor_parallel_size=tp_size,
#             trust_remote_code=trust_remote_code,
#             guided_decoding_backend="outlines" # Required for JSON schema enforcement
#         )
#         self.tokenizer = self.llm.get_tokenizer()
#         self.default_temp = temp
#         logger.info("LLM and tokenizer initialized successfully.")

#     def generate_structured_response(self, prompt: str, max_tokens: int) -> dict:
#         """
#         Generates a single structured (JSON) response for a given prompt.

#         Args:
#             prompt (str): The fully formatted prompt string.
#             max_tokens (int): The maximum number of tokens to generate.

#         Returns:
#             dict: The parsed JSON object from the model's response.
#                   Returns a dictionary with an 'error' key if parsing fails.
#         """
#         # Define a generic JSON schema for the response
#         json_schema = {
#             "type": "object",
#             "properties": {
#                 "answer": {"type": "string"},
#                 "supporting_facts": {
#                     "type": "array",
#                     "items": {
#                         "type": "array",
#                         "items": [{"type": "string"}, {"type": "integer"}],
#                         "minItems": 2,
#                         "maxItems": 2
#                     }
#                 }
#             },
#             "required": ["answer"]
#         }
        
#         sampling_params = SamplingParams(
#             temperature=self.default_temp,
#             max_tokens=max_tokens,
#             guided_decoding_backend='outlines',
#             guided_json=json_schema
#         )

#         outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        
#         # vLLM returns a list of outputs, one for each prompt. We sent one.
#         response_text = outputs.outputs.text
        
#         try:
#             return json.loads(response_text)
#         except json.JSONDecodeError:
#             logger.error(f"Failed to parse JSON despite guided decoding: {response_text}")
#             return {"error": "Failed to parse JSON", "raw_output": response_text}

#     def generate_parallel_responses(self, prompt: str, n: int, max_tokens: int) -> list[dict]:
#         """
#         Generates N parallel responses for a single prompt.

#         Args:
#             prompt (str): The fully formatted prompt string.
#             n (int): The number of parallel responses to generate.
#             max_tokens (int): The maximum number of tokens for each response.

#         Returns:
#             list[dict]: A list of parsed JSON objects from the model's responses.
#         """
#         sampling_params = SamplingParams(
#             n=n,
#             temperature=self.default_temp,
#             max_tokens=max_tokens
#         )

#         outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        
#         parsed_responses = []
#         for i in range(n):
#             response_text = outputs.outputs[i].text
#             try:
#                 # Find the first '{' and the last '}' to extract the JSON object
#                 start = response_text.find('{')
#                 end = response_text.rfind('}') + 1
#                 if start!= -1 and end!= 0:
#                     json_str = response_text[start:end]
#                     parsed_responses.append(json.loads(json_str))
#                 else:
#                     parsed_responses.append({"error": "No JSON object found", "raw_output": response_text})
#             except json.JSONDecodeError:
#                 logger.warning(f"Could not parse parallel response {i+1}: {response_text}")
#                 parsed_responses.append({"error": "Failed to parse JSON", "raw_output": response_text})
        
#         return parsed_responses


# src/llm_interface.py

import logging
from typing import List

# Use the same vLLM imports as your working projects
from vllm import LLM as VLLM_Engine, SamplingParams

logger = logging.getLogger(__name__)

class LlamaInterface:
    """
    A simple, direct wrapper for the vLLM engine.
    Its only job is to initialize the model and generate raw text.
    """
    def __init__(self, model_path: str, temp: float, tp_size: int, trust_remote_code: bool = True):
        logger.info(f"Initializing vLLM engine from path: {model_path}")
        self.llm_engine = VLLM_Engine(
            model=model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=trust_remote_code
        )
        self.tokenizer = self.llm_engine.get_tokenizer()
        self.default_temp = temp
        logger.info("LLM and tokenizer initialized successfully.")

    def create_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Applies the chat template to system and user prompts.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, temperature: float, max_tokens: int, n: int = 1) -> List[str]:
        """
        Generates N text responses for a SINGLE prompt.
        This is a much simpler and clearer interface.
        """
        if not prompt:
            return [""] * n

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
        )

        logger.info(f"Generating {n} response(s) for the prompt...")
        # We send a single prompt string and get back a list of request outputs
        request_outputs = self.llm_engine.generate(prompt, sampling_params, use_tqdm=False)
        
        # The output for a single prompt is the first element in the list
        output = request_outputs[0]
        generated_texts = [completion.text.strip() for completion in output.outputs]

        return generated_texts