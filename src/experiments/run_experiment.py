# experiments/run_experiment.py (Conceptual)
import argparse
import os
import torch
# import pandas as pd
import json # 
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm # 引入tqdm来显示进度条


# Import custom modules from the 'src' package
# from src.llm_interface import LlamaInterface
from src.backends.llm_interface import LlamaInterface
from src.data_loader import load_dataset_by_name
from src.strategies import get_strategy
from src.evaluation import evaluate_prediction

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_available_gpu_memory_resume(): # Renamed to avoid conflict if imported elsewhere
    """Prints available and total GPU memory for all GPUs."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")
        for i in range(num_gpus):
            torch.cuda.set_device(i) # Select GPU i
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"[GPU {i}] Available memory: {free_mem / (1024 ** 3):.2f} GiB / {total_mem / (1024 ** 3):.2f} GiB")
    else:
        print("No CUDA-capable GPU detected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference scaling experiments for 'abra'.")
    
    # Experiment-specific parameters
    parser.add_argument("--dataset", type=str, required=True, choices=["google_math", "gsm8k", 'hotpotqa', 'musique', '2wikimultihopqa'], help="Dataset to use.")
    parser.add_argument("--strategy", type=str, required=True, choices=['parallel', 'sequential', 'parallel-rrm'], help="Generation strategy to use.")
    parser.add_argument("--compute_budget", type=int, required=True, help="Total token budget per problem.")
    
    # Model and Path parameters (mirroring your script)
    parser.add_argument("--model_name", type=str, default="Llama-3.3-70B-Instruct", help="Name of the model for logging.")
    parser.add_argument("--model_cache_path", type=str, required=True, help="Path to the LLM cache.")
    parser.add_argument("--base_path", type=str, default=".", help="Base path of the project.")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results and logs.")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of parallel samples or sequential steps.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size for vLLM.")
    
    # Debugging
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode to run on a small subset.")
    parser.add_argument("--debug_sample_size", type=int, default=10, help="Number of samples for debug mode.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of samples to process. -1 means no limit.")
    return parser.parse_args()

def main():
    print_available_gpu_memory_resume()
    args = parse_args()

    # 1. Setup paths
    output_dir = Path(args.output_dir)
    results_file = output_dir / "results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting experiment with arguments: {args}")
    logger.info(f"Results will be saved to: {results_file}")

    # 2. Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    effective_limit = -1
    if args.debug_mode:
        effective_limit = args.debug_sample_size
        logger.info(f"--- DEBUG MODE: Processing only {effective_limit} samples. ---")
    elif args.limit > 0:
        effective_limit = args.limit
        logger.info(f"--- Sample limit set to {effective_limit} samples. ---")

    # --- Load Dataset ---
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset_by_name(
        args.dataset,
        base_data_path=Path(args.base_path) / "data",
        limit=effective_limit # Pass the final calculated limit
    )
    
    # 3. Initialize LLM backend
    logger.info(f"Initializing LLM: {args.model_name}")
    llm = LlamaInterface(
        model_path=args.model_cache_path,
        temp=args.temperature,
        tp_size=args.tensor_parallel_size
    )

    # 4. Initialize generation strategy
    logger.info(f"Using strategy: {args.strategy}")
    strategy = get_strategy(
        strategy_name=args.strategy,
        llm=llm,
        prompt_dir=Path(args.base_path) / "prompts",
        dataset_name=args.dataset,
        budget=args.compute_budget,
        n_samples=args.n_samples
    )

    # 5. Main experiment loop
    all_results = []
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {args.dataset}"):
        logger.info(f"Processing item {i+1}/{len(dataset)}...")
        
        # Generate prediction using the selected strategy
        prediction_json = strategy.generate(item) #{"aswer":""}
        # pred_answer_str = prediction_json.get("answer", "")

        # Evaluate the prediction
        metrics = evaluate_prediction(prediction_json, item, args.dataset)

        # Log results constructing the results, prediction has the answer and supporting facts used to generate the answer
        result_record = {
            "item_id": item.get("id") or item.get("_id") or i,
            "question": item.get("question"),
            "prediction": prediction_json,
            "ground_truth": item.get("answer"),
            "metrics": metrics
        }
        all_results.append(result_record)

        # Append to file incrementally
        with open(results_file, "a") as f:
            f.write(json.dumps(result_record) + "\n")

    logger.info("Experiment finished successfully.")
    # Optionally save a final summary CSV
    pd.DataFrame(all_results).to_csv(output_dir / "summary.csv", index=False)

if __name__ == "__main__":
    main()