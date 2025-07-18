# src/parsing_utils.py

import logging
import re
import json
import ast 
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def clean_invalid_json_escapes(json_str: str) -> str:
    # """
    # Replace invalid escape sequences in JSON with valid ones.
    # For example, \x, \z, etc. will be replaced with escaped backslashes.
    # """
    # Match \ followed by a non-valid escape character
    json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)
    return json_str

def parse_llm_json_answer(raw_text: str) -> Optional[Dict]:
    """
    Robustly parses a JSON object from LLM output, even if it contains invalid escapes.
    """

    # 1. Try to extract markdown ```json ... ``` block
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if not match:
        # 2. Fallback: try to extract any JSON-looking substring
        match = re.search(r'(\{.*\})', raw_text, re.DOTALL)

    if not match:
        logger.warning(f"No JSON object found in response: {raw_text[:200]}...")
        return None

    json_str = match.group(1)

    # Step A: First try json.loads with strict=False
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as e1:
        logger.warning(f"json.loads failed: {e1}. Attempting to clean invalid escapes...")

    # Step B: Clean up escape characters and retry json.loads
    cleaned_str = clean_invalid_json_escapes(json_str)
    try:
        return json.loads(cleaned_str, strict=False)
    except json.JSONDecodeError as e2:
        logger.warning(f"json.loads still failed after cleaning: {e2}. Trying ast.literal_eval...")

    # Step C: Try using ast.literal_eval as a last resort
    try:
        return ast.literal_eval(cleaned_str)
    except Exception as e3:
        logger.warning(f"ast.literal_eval also failed: {e3}. Raw string: {json_str[:200]}...")
        return None

# def parse_llm_json_answer(raw_text: str) -> Optional[Dict]:
#     """
#     Finds and parses a JSON object from a raw string, handling common LLM mistakes
#     like markdown fences and invalid escape characters.
#     """
#     # Attempt 1: Look for a markdown-fenced JSON block first.
#     match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
#     if match:
#         json_str = match.group(1)
#         try:
#             # --- THE FIX IS HERE ---
#             # Use strict=False to allow for minor syntax errors like invalid escapes.
#             return json.loads(json_str, strict=False)
#         except json.JSONDecodeError as e:
#             logger.warning(f"Failed to parse JSON from markdown block: {e}. Raw block: {json_str[:200]}...")
#             # Fall through to the next attempt if markdown block is found but still fails
    
#     # Fallback Attempt 2: If no markdown block, find the first '{' and last '}'
#     match = re.search(r'\{.*\}', raw_text, re.DOTALL)
#     if not match:
#         logger.warning(f"No JSON object found in response: {raw_text[:100]}...")
#         return None
    
#     json_str = match.group(0)
#     try:
#         # --- THE FIX IS HERE ---
#         # Also apply strict=False to the fallback parser.
#         return json.loads(json_str, strict=False)
#     except json.JSONDecodeError as e:
#         logger.warning(f"Fallback parsing also failed: {e}. Raw string: {json_str[:200]}...")
#         return None

def parse_llm_string_answer(raw_text: str, dataset_name: str) -> str:
    """
    Parses the raw text output from the LLM to extract a clean string answer.
    Used for datasets like gsm8k and math.
    """
    dataset_name = dataset_name.lower()
    
    # Handle gsm8k format: "The final answer is #### 123"
    if dataset_name == 'gsm8k':
        parts = raw_text.split("####")
        if len(parts) > 1:
            return parts[-1].replace(",", "").strip()

    # Handle competition math format: "... \boxed{123} ..."
    elif dataset_name in ['google_math', 'math_500']:
        match = re.search(r"\\boxed\{(.*?)\}", raw_text)
        if match:
            return match.group(1).replace(",", "").strip()
    
    # Fallback if no special format is found
    return raw_text.strip()