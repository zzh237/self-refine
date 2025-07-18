import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_per_dataset_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates difficulty based on the average performance of each question
    within its specific dataset. This implements your requested averaging method.

    Args:
        df (pd.DataFrame): The master DataFrame containing all results.

    Returns:
        pd.DataFrame: The DataFrame with new columns for per-dataset difficulty.
    """
    print("Calculating difficulty bins by averaging performance within each dataset...")
    
    all_binned_groups = []
    # Group the entire master dataframe by the 'dataset' column first.
    for dataset_name, group_df in df.groupby('dataset'):
        if group_df.empty:
            continue
            
        # --- NEW AVERAGING LOGIC ---
        # For each item_id within this dataset group, calculate its mean accuracy
        # across all strategies, budgets, and n_samples.
        difficulty_map = group_df.groupby('item_id')['accuracy'].mean()
        
        # Map this robust score back to every row for this dataset group.
        group_df['difficulty_score_per_dataset'] = group_df['item_id'].map(difficulty_map)
        
        try:
            # pd.qcut will try to split the questions into 3 equal-sized groups based on score
            group_df['difficulty_bin_per_dataset'] = pd.qcut(
                group_df['difficulty_score_per_dataset'], 
                q=3, 
                labels=['hard', 'medium', 'easy'], 
                duplicates='drop' # This helps if there are many identical scores
            )
            logging.info(f"Successfully created 3 difficulty bins for {dataset_name}.")
        except ValueError:
            # This happens if there are not enough unique scores to make 3 bins
            logging.warning(f"Could not create 3 bins for {dataset_name}. Falling back to 2 bins (easy/hard).")
            group_df['difficulty_bin_per_dataset'] = group_df['difficulty_score_per_dataset'].apply(
                lambda score: 'easy' if score >= 0.5 else 'hard'
            )
        
        all_binned_groups.append(group_df)
    
    if not all_binned_groups:
        return df # Return original df if no groups were processed
        
    return pd.concat(all_binned_groups, ignore_index=True)


def calculate_overall_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a difficulty score by averaging performance across ALL experiments.
    This was your proposed method.
    """
    print("Calculating difficulty bins based on overall average performance...")
    # For each item, calculate its average accuracy across all runs
    difficulty_map = df.groupby('item_id')['accuracy'].mean()
    df['difficulty_score'] = df['item_id'].map(difficulty_map)
    try:
        # pd.qcut will try to split the questions into 3 equal-sized groups based on score
        df['difficulty_bin'] = pd.qcut(
            df['difficulty_score'], 
            q=3, 
            labels=['hard', 'medium', 'easy'], 
            duplicates='drop' # This helps if there are many identical scores
        )
        logging.info(f"Successfully created 3 difficulty bins regardless which dataset.")
    except ValueError:
        # This happens if there are not enough unique scores to make 3 bins
        logging.warning(f"Could not create 3 bins for regardless which dataset. Falling back to 2 bins (easy/hard).")
        df['difficulty_bin'] = df['difficulty_score'].apply(
            lambda score: 'easy' if score >= 0.5 else 'hard'
        )
    return df


def add_difficulty_bin(df: pd.DataFrame, score_col: str, bin_col: str) -> pd.DataFrame:
    """Helper function to add a binned column from a score column."""
    try:
        # Try to create 3 bins (hard, medium, easy) using quantiles
        df[bin_col] = pd.qcut(df[score_col], q=3, labels=['hard', 'medium', 'easy'], duplicates='drop')
    except ValueError:
        # Fallback to a simple 2-bin split if 3 is not possible
        logging.warning(f"Could not create 3 bins for {score_col}. Falling back to 2 bins (easy/hard).")
        df[bin_col] = df[score_col].apply(lambda score: 'easy' if score >= 0.5 else 'hard')
    return df

def calculate_all_difficulties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds all four types of difficulty metrics to the DataFrame.
    """
    if 'accuracy' not in df.columns:
        logging.error("Accuracy column not found, cannot calculate difficulty.")
        return df

    # --- 1. Overall Difficulty (grouped by item_id only) ---
    logging.info("Calculating Overall Difficulty...")
    difficulty_map = df.groupby('item_id')['accuracy'].mean()
    df['difficulty_score_overall'] = df['item_id'].map(difficulty_map)
    df = add_difficulty_bin(df, 'difficulty_score_overall', 'difficulty_bin_overall')

    # --- 2. Per-Dataset Difficulty (grouped by dataset, then item_id) ---
    logging.info("Calculating Per-Dataset Difficulty...")
    difficulty_map = df.groupby(['dataset', 'item_id'])['accuracy'].transform('mean')
    df['difficulty_score_per_dataset'] = difficulty_map
    df = add_difficulty_bin(df, 'difficulty_score_per_dataset', 'difficulty_bin_per_dataset')

    # --- 3. By-Strategy Difficulty (grouped by strategy, then item_id) ---
    logging.info("Calculating By-Strategy Difficulty...")
    difficulty_map = df.groupby(['strategy', 'item_id'])['accuracy'].transform('mean')
    df['difficulty_score_by_strategy'] = difficulty_map
    df = add_difficulty_bin(df, 'difficulty_score_by_strategy', 'difficulty_bin_by_strategy')

    # --- 4. Per-Dataset-Strategy Difficulty (most granular) ---
    logging.info("Calculating Per-Dataset-Strategy Difficulty...")
    difficulty_map = df.groupby(['dataset', 'strategy', 'item_id'])['accuracy'].transform('mean')
    df['difficulty_score_per_dataset_strategy'] = difficulty_map
    df = add_difficulty_bin(df, 'difficulty_score_per_dataset_strategy', 'difficulty_bin_per_dataset_strategy')
    
    return df



def main():
    """Main function to load data, calculate difficulties, and save."""
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "notebooks" / "master_results.csv"
    output_file = project_root / "notebooks" / "master_results_with_difficulty.csv"

    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully loaded {input_file}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}. Please run create_master_summary.py first.")
        return


    # Apply both difficulty calculation methods
    # df_final = calculate_per_dataset_difficulty(df.copy())
    # df_final = calculate_overall_difficulty(df_final)

    df_final = calculate_all_difficulties(df)
    df_final.sort_values(
        by=['dataset', 'item_id', 'strategy', 'compute_budget', 'n_samples'],
        inplace=True
    )

    df_final.to_csv(output_file, index=False)
    logging.info(f"Successfully saved enriched data to {output_file}")

if __name__ == "__main__":
    main()

