# src/evaluation.py

import logging
import re
import string
from collections import Counter
import evaluate # Use the new library

logger = logging.getLogger(__name__)


# --- The ONLY Answer Parser You Need ---
# This single function replaces get_boxed_answer and get_gsm8k_answer
def parse_llm_answer(raw_text: str, dataset_name: str) -> str:
    """
    Parses the raw text output from the LLM to extract the clean answer
    based on the dataset's specific format.
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
    
    # Fallback for QA datasets or if no special format is found
    return raw_text.strip()


# --- Evaluation logic for google/math ---

def get_boxed_answer(solution: str) -> str:
    """Extracts the final answer from a \\boxed{} environment in LaTeX."""
    match = re.search(r"\\boxed\{(.*?)\}", solution)
    if match:
        # Normalize by removing commas from numbers
        return match.group(1).replace(",", "").strip()
    return ""

def evaluate_competition_math(prediction: dict, ground_truth_item: dict) -> dict:
    """Evaluates a prediction for competition-style math datasets."""
    try:
        math_metric = evaluate.load("competition_math")
        pred_answer_str = str(prediction.get("answer", ""))
        ref_solution = ground_truth_item.get("answer", "") # For math, 'answer' field holds the full solution
        
        results = math_metric.compute(predictions=[pred_answer_str], references=[ref_solution])
        return {"accuracy": results.get('accuracy', 0.0)}
    except Exception as e:
        logger.error(f"Error during competition_math evaluation: {e}")
        return {"accuracy": 0.0}
# --- Evaluation logic for gsm8k ---

def get_gsm8k_answer(answer_str: str) -> str:
    """Extracts the final numerical answer from a gsm8k solution string."""
    # The answer is the number that comes after "####"
    parts = answer_str.split("####")
    if len(parts) > 1:
        # Get the last part, remove commas and whitespace
        return parts[-1].replace(",", "").strip()
    return ""
def evaluate_gsm8k(prediction: dict, ground_truth_item: dict) -> dict:
    """Evaluates a prediction for the gsm8k dataset."""
    pred_answer_str = str(prediction.get("answer", ""))
    gold_answer_str = get_gsm8k_answer(ground_truth_item.get("answer", ""))

    # Normalize prediction as well, in case it includes reasoning
    if "####" in pred_answer_str:
        pred_answer_str = get_gsm8k_answer(pred_answer_str)
        
    # Simple exact match after normalization
    is_correct = pred_answer_str == gold_answer_str
    return {"accuracy": 1.0 if is_correct else 0.0}
# --- Evaluation logic for HotpotQA and MuSiQue ---

def normalize_answer(s: str) -> str:
    # ... (code from previous turn)
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> tuple:
    """Computes F1, precision, and recall scores."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    if not normalized_prediction or not normalized_ground_truth:
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Computes exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def supporting_facts_f1(pred_sp: list, gold_sp: list) -> tuple:
    """Computes F1, precision, and recall for supporting facts."""
    pred_set = set(map(tuple, pred_sp))
    gold_set = set(map(tuple, gold_sp))
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    if tp == 0:
        return 0.0, 0.0, 0.0
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def evaluate_qa(prediction: dict, ground_truth_item: dict) -> dict:
    """Evaluates a prediction for QA datasets like HotpotQA and MuSiQue."""
    pred_ans = prediction.get("answer", "")
    gold_ans = ground_truth_item.get("answer", "")
    
    # For MuSiQue, supporting facts are in 'answer_supporting_paragraphs'
    # For HotpotQA, they are in 'supporting_facts'
    gold_sp_raw = ground_truth_item.get("supporting_facts")
    if gold_sp_raw is None: # Try MuSiQue format
        gold_sp_raw = ground_truth_item.get("answer_supporting_paragraphs")
    
    # --- FIX 1: Added an empty list [] for the 'else' case ---
    # Normalize supporting facts to a consistent list of tuples format
    gold_sp = [tuple(fact) for fact in gold_sp_raw] if isinstance(gold_sp_raw, list) else []

    # --- FIX 2: Added a default empty list [] to the .get() call ---
    # This prevents an error if the prediction has no supporting facts.
    pred_sp = prediction.get("supporting_facts", [])
    
    em = exact_match_score(pred_ans, gold_ans)
    f1, prec, recall = f1_score(pred_ans, gold_ans)
    sp_f1, sp_prec, sp_recall = supporting_facts_f1(pred_sp, gold_sp)
    
    return {
        "answer_em": em,
        "answer_f1": f1,
        "answer_precision": prec,
        "answer_recall": recall,
        "support_f1": sp_f1,
        "support_precision": sp_prec,
        "support_recall": sp_recall
    }


def evaluate_qa(prediction: dict, ground_truth_item: dict) -> dict:
    """Evaluates a prediction for QA datasets, now including support metrics."""
    pred_ans = prediction.get("answer", "")
    gold_ans = ground_truth_item.get("answer", "")
    
    pred_sp = prediction.get("supporting_facts", [])
    gold_sp = ground_truth_item.get("supporting_facts", [])
    
    em = exact_match_score(pred_ans, gold_ans)
    f1, prec, recall = f1_score(pred_ans, gold_ans)
    sp_f1, sp_prec, sp_recall = supporting_facts_f1(pred_sp, gold_sp)
    
    return {
        "answer_em": em,
        "answer_f1": f1,
        "answer_precision": prec,
        "answer_recall": recall,
        "support_f1": sp_f1,
        "support_precision": sp_prec,
        "support_recall": sp_recall
    }

# --- Main Evaluation Dispatcher ---

def evaluate_prediction(prediction: dict, ground_truth_item: dict, dataset_name: str) -> dict:
    """Dispatches to the correct evaluation function based on the dataset name."""
    # if "error" in prediction:
    #     logger.warning(f"Skipping evaluation for item due to prediction error: {prediction['error']}")
    #     return {"error": 1}
    if not prediction or "error" in prediction.get("answer", "").lower():
        # Return a dictionary with zero scores for all metrics
        if dataset_name in ['hotpotqa', 'musique', '2wikimultihopqa']:
             return {"answer_em": 0.0, "answer_f1": 0.0, "answer_precision": 0.0, "answer_recall": 0.0,
                    "support_f1": 0.0, "support_precision": 0.0, "support_recall": 0.0, "error": 1}
        else:
            return {"accuracy": 0.0, "error": 1}
    
    # --- UPDATED SECTION ---
    dataset_name = dataset_name.lower()
    if dataset_name ==  'math_500':
        return evaluate_competition_math(prediction, ground_truth_item)
    elif dataset_name == 'gsm8k':
        return evaluate_gsm8k(prediction, ground_truth_item)
    elif dataset_name in ['hotpotqa', 'musique','2wikimultihopqa']:
        # This part requires the full evaluate_qa function from the previous step
        # return evaluate_qa(prediction, ground_truth_item)
        # For now, let's just return a placeholder for QA
        # return {"answer_f1": 0.0, "support_f1": 0.0} # Replace with your full evaluate_qa
        return evaluate_qa(prediction, ground_truth_item)
    else:
        logger.error(f"No evaluation function available for dataset: {dataset_name}")
        return {"error": 1, "reason": f"Unknown dataset {dataset_name}"}