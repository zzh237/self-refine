# import logging
# from typing import Dict, Any

# logger = logging.getLogger(__name__)


# def load_prompt_from_file(file_path: str) -> str:
#     """
#     Loads a text-based prompt template from a file.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return f.read()
#     except FileNotFoundError:
#         logger.error(f"Prompt template file not found at: {file_path}")
#         raise
#     except Exception as e:
#         logger.error(f"Error loading prompt template from {file_path}: {e}")
#         raise

# def format_prompt(template: str, data_point: Dict[str, Any], dataset_name: str, **kwargs) -> str:
#     """
#     Formats a prompt template with data from a single data point
#     and any additional dynamic keyword arguments.
#     """
#     # Combine the data point and any extra kwargs into one dictionary
#     # kwargs will overwrite keys from data_point if they overlap
#     format_dict = {**data_point, **kwargs}
    
#     # Handle cases where context might not exist (like in MATH dataset)
#     if 'context' not in format_dict:
#         format_dict['context'] = "No external context provided."
        
#     return template.format(**format_dict)



# src/prompt_utils.py
import logging
import json
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_prompt_templates_from_json(file_path: Path) -> Dict[str, str]:
    """
    Loads a JSON file containing multiple named prompt templates.
    """
    logger.info(f"Loading prompt templates from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Prompt template file not found at: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        raise

def format_prompt(template: str, data_item: Dict[str, Any], **kwargs) -> str:
    """
    Formats a prompt template with data from a single data item
    and any additional dynamic keyword arguments.
    """
    # Combine the data item and any extra kwargs into one dictionary
    format_dict = {**data_item, **kwargs}
    return template.format(**format_dict)