import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_results():
    """
    Walks through the results directory, aggregates all summary.csv files,
    calculates difficulty bins, and saves a master CSV file.
    """
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    output_file = project_root / "notebooks" / "master_results.csv"

    all_summaries = []

    logging.info(f"Scanning for summary files in: {results_dir}")
    summary_files = list(results_dir.glob("**/summary.csv"))

    if not summary_files:
        logging.error("No summary.csv files found. Please run the experiments first.")
        return

    for file_path in summary_files:
        try:
            # Extract parameters from the directory path
            parts = file_path.parent.parent.name.split('_')
            if len(parts) < 3:
                continue
            
            dataset, strategy, n_samples_str = parts[0], parts[1], parts[2]
            n_samples = int(n_samples_str.replace('n', ''))

            df = pd.read_csv(file_path)
            df['dataset'] = dataset
            df['strategy'] = strategy
            df['n_samples'] = n_samples
            all_summaries.append(df)
        except Exception as e:
            logging.warning(f"Could not process file {file_path}: {e}")

    master_df = pd.concat(all_summaries, ignore_index=True)
    logging.info(f"Aggregated {len(master_df)} total results from {len(summary_files)} summary files.")

    # --- Calculate Difficulty Bins (for Snell et al. Figure 3, right) ---
    logging.info("Calculating difficulty bins for each dataset...")
    binned_dfs = []
    for dataset_name, group_df in master_df.groupby('dataset'):
        # Use the parallel run with the highest sample count as the reference for difficulty
        ref_run = group_df[(group_df['strategy'] == 'parallel') & (group_df['n_samples'] == group_df['n_samples'].max())]
        
        if ref_run.empty:
            logging.warning(f"No reference run found for dataset {dataset_name} to create difficulty bins.")
            group_df['difficulty'] = 'medium' # Default
        else:
            # Create a map from item_id to its accuracy in the reference run
            difficulty_map = ref_run.set_index('item_id')['accuracy']
            # Map this back to the entire group
            group_df['difficulty_score'] = group_df['item_id'].map(difficulty_map)
            # Create bins using quantiles
            group_df['difficulty'] = pd.qcut(group_df['difficulty_score'], 3, labels=['easy', 'medium', 'hard'], duplicates='drop')
        
        binned_dfs.append(group_df)
    
    if binned_dfs:
        master_df = pd.concat(binned_dfs, ignore_index=True)
        logging.info("Finished calculating difficulty bins.")

    master_df.to_csv(output_file, index=False)
    logging.info(f"Master summary saved to {output_file}")

if __name__ == "__main__":
    analyze_results()