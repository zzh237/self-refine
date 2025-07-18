import pandas as pd
from pathlib import Path
import logging
import json
import sys
import re

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import evaluate_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_difficulty_scores(df: pd.DataFrame) -> pd.Series:
    """
    Calculates a robust difficulty score for each unique question (item_id)
    by averaging its accuracy across all experimental conditions.

    Args:
        df (pd.DataFrame): The master DataFrame containing all results.

    Returns:
        pd.Series: A pandas Series where the index is 'item_id' and the
                   value is the calculated difficulty score.
    """
    if 'item_id' not in df.columns or 'accuracy' not in df.columns:
        logging.warning("Cannot calculate difficulty scores. 'item_id' or 'accuracy' column is missing.")
        return None

    logger.info("Calculating a robust difficulty score for each question by averaging across all runs...")
    # This single line performs the multi-level aggregation you described:
    # It groups by each unique question and finds its mean accuracy across all strategies, budgets, and n_samples.
    difficulty_map = df.groupby('item_id')['accuracy'].mean()
    return difficulty_map


def analyze_results():
    """
    Walks through the results directory, reads the raw .jsonl files,
    re-calculates metrics with the updated evaluation logic, and saves a master CSV.
    """
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    output_file = project_root / "notebooks" / "master_results2.csv"
    output_file.parent.mkdir(exist_ok=True)

    all_results = []
    logging.info(f"Scanning for results.jsonl files in: {results_dir}")
    jsonl_files = list(results_dir.glob("**/results.jsonl"))

    if not jsonl_files:
        logging.error("No results.jsonl files found. Please run experiments first.")
        return

    # --- Logic to find only the LATEST run for each experiment setting ---
    latest_runs = {}
    for file_path in jsonl_files:
        run_name = file_path.parent.parent.name
        timestamp = file_path.parent.name
        if run_name not in latest_runs or timestamp > latest_runs[run_name].parent.name:
            latest_runs[run_name] = file_path
    
    logging.info(f"Found {len(latest_runs)} unique experiment settings to process from the latest runs.")

    
    for run_name, file_path in latest_runs.items():
        try:
            # Extract parameters from the directory path
            parts = run_name.split('_')
            dataset, strategy = parts[0], parts[1]
            # budget_str = next((p for p in parts if p.startswith('b')), 'b0')
            # n_samples_str = next((p for p in parts if p.startswith('n')), 'n0')
            # compute_budget = int(budget_str[1:])
            # n_samples = int(n_samples_str[1:])
            # --- NEW, ROBUST PARSING LOGIC ---
            budget_str = next((p for p in parts if p.startswith('b') or p.startswith('budget')), 'b0')
            n_samples_str = next((p for p in parts if p.startswith('n')), 'n0')
            
            # Use regex to find the first sequence of digits in the string
            compute_budget = int(re.search(r'\d+', budget_str).group()) if re.search(r'\d+', budget_str) else 0
            n_samples = int(re.search(r'\d+', n_samples_str).group()) if re.search(r'\d+', n_samples_str) else 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue # Skip empty lines
                    record = json.loads(line)
                    
                    # --- RE-EVALUATE HERE ---
                    # Use the corrected evaluation function on the saved prediction
                    corrected_metrics = evaluate_prediction(
                        record['prediction'], 
                        {"answer": record['ground_truth']}, # Create a mock ground_truth_item
                        dataset
                    )
                    
                    new_record = {
                        'item_id': record['item_id'],
                        'dataset': dataset,
                        'strategy': strategy,
                        'compute_budget': compute_budget,
                        'n_samples': n_samples,
                        **corrected_metrics # Add all the corrected metric columns
                    }
                    all_results.append(new_record)

        except Exception as e:
            logging.warning(f"Could not process file {file_path}: {e}")
    
    master_df = pd.DataFrame(all_results)
    
        
    # logging.info(f"Aggregated {len(master_df)} total results from {len(summary_files)} summary files.")

    
    master_df.to_csv(output_file, index=False)
    logging.info(f"Master summary saved to {output_file}")

if __name__ == "__main__":
    analyze_results()