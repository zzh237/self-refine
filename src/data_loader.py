# src/data_loader.py
import os 
import logging
import json # Import Python's built-in JSON library
from datasets import load_dataset, Dataset, DatasetDict, DownloadConfig, Features, Value, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# Updated list of supported datasets
SUPPORTED_DATASETS = ["gsm8k", "hotpotqa", "musique","2wikimultihopqa"]




def _preprocess_musique(dataset_split: Dataset) -> Dataset:
    """Standardizes the MuSiQue dataset format."""
    logger.info("Preprocessing MuSiQue data to create a standard 'context' field..")
    # logger.info(f"Preprocessing '{dataset_name}' to create a standard 'context' field...")
    # --- FIX FOR MUSIQUE IS HERE ---
    processed_data = []
    def format_musique_item(item):
        # The 'paragraphs' field is a list of dicts.
        context_str = "\n\n".join(
            [f"Title: {p['title']}\n{p['paragraph_text']}" for p in item.get("paragraphs", [])]
        )
        return {
            'id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
            'context': context_str,
            'supporting_facts': [] # MuSiQue format doesn't have the same support facts structure
        }
    for item in dataset_split:
        # The 'paragraphs' field is a list of dicts. We join them into a single string.
        # We create a new, standardized dictionary for each item
        processed_item = format_musique_item(item)
        processed_data.append(processed_item)
    # Convert the list of processed dicts back into a Dataset object
    data_to_process = Dataset.from_list(processed_data)
    return data_to_process

# def _preprocess_2wikimultihopqa(dataset_split: Dataset) -> Dataset:
#     """Standardizes the 2WikiMultiHopQA dataset format."""
#     logger.info("Preprocessing 2WikiMultiHopQA data...")

#     def format_2wiki_item(item):
#         context_data = item.get("context", [])
#         # --- THE FIX IS HERE ---
#         # First, check if the context is a string that needs to be parsed
#         if isinstance(context_data, str):
#             try:
#                 context_data = json.loads(context_data)
#             except json.JSONDecodeError:
#                 logger.warning(f"Could not parse context JSON string for item {item.get('id')}")
#                 context_data = [] # Default to an empty list on failure
#         # --- END OF FIX ---

#         # The 'context' field is a list of dicts with 'title' and 'content' keys.
#         context_str = "\n\n".join(
#             [f"Title: {p.get('title', '')}\n{' '.join(p.get('content', []))}" for p in context_data if isinstance(p, dict)]
#         )
#         return {
#             'id': str(item.get('_id') or item.get('id', '')),
#             'question': item['question'],
#             'answer': item['answer'],
#             'context': context_str,
#             'supporting_facts': item.get('supporting_facts', [])
#         }
#     return dataset_split.map(format_2wiki_item, load_from_cache_file=False)

def _preprocess_2wikimultihopqa(dataset_split: Dataset) -> Dataset:
    """
    Standardizes the 2WikiMultiHopQA dataset format by parsing JSON strings
    in its columns and creating a unified context field.
    """
    logger.info("Preprocessing 2WikiMultiHopQA data...")

    def parse_and_format_item(item):
        """
        Parses all JSON-string fields and formats the item into the standard structure.
        """
        def _safe_json_loads(data_str, default_value=None):
            if default_value is None:
                default_value = []
            if not isinstance(data_str, str):
                return data_str
            try:
                return json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse JSON string for item {item.get('_id', 'N/A')}. Content: '{data_str[:100]}...'")
                return default_value

        context_list = _safe_json_loads(item.get("context"))
        supporting_facts_list = _safe_json_loads(item.get("supporting_facts"))
        
        context_str = "\n\n".join(
            f"Title: {ctx[0]}\n{' '.join(ctx[1])}"
            for ctx in context_list if isinstance(ctx, (list, tuple)) and len(ctx) == 2
        )

        # --- THE FIX IS HERE ---
        # Ensure all elements within supporting_facts are strings to match the target Feature schema.
        # It converts lists like [['title_A', 0]] to [['title_A', '0']].
        sanitized_supporting_facts = [
            [str(part) for part in fact] 
            for fact in supporting_facts_list if isinstance(fact, list)
        ]
        # --- END OF FIX ---

        return {
            'id': str(item.get('_id', '')),
            'question': item['question'],
            'answer': item['answer'],
            'context': context_str,
            'supporting_facts': sanitized_supporting_facts  # Use the sanitized version
        }

    # Since we are modifying the structure, it's safer to remove old columns
    # and then cast to a new, well-defined feature set.
    processed_dataset = dataset_split.map(
        parse_and_format_item, 
        remove_columns=dataset_split.column_names,
        load_from_cache_file=False
    )
    
    final_features = Features({
        'id': Value("string"),
        'question': Value("string"),
        'answer': Value("string"),
        'context': Value("string"),
        'supporting_facts': Sequence(
            Sequence(Value("string")) 
        )
    })

    # The final cast ensures everything aligns perfectly.
    return processed_dataset.cast(final_features)


# def _preprocess_2wikimultihopqa(dataset_split: Dataset) -> Dataset:
#     """Standardizes the 2WikiMultiHopQA dataset format."""
#     logger.info("Preprocessing 2WikiMultiHopQA data...")

#     def parse_json_field(field, default):
#         """Robustly parse a JSON field, fallback to default if needed."""
#         if isinstance(field, str):
#             try:
#                 return json.loads(field)
#             except json.JSONDecodeError:
#                 logger.warning(f"Failed to parse JSON: {field[:100]}...")
#                 return default
#         elif isinstance(field, list):
#             return field
#         else:
#             return default

#     def format_2wiki_item(item):
#         context_data = parse_json_field(item.get("context", []), [])
#         supporting_facts = parse_json_field(item.get("supporting_facts", []), [])
#         evidences = parse_json_field(item.get("evidences", []), [])

#         context_str = "\n\n".join([
#             f"Title: {p.get('title', '')}\n{' '.join(p.get('content', []))}"
#             for p in context_data if isinstance(p, dict)
#         ])

#         return {
#             'id': str(item.get('_id') or item.get('id', '')),
#             'question': item.get('question', ''),
#             'answer': item.get('answer', ''),
#             'context': context_str,
#             'supporting_facts': supporting_facts,
#             # Optionally include evidences if needed downstream
#             # 'evidences': evidences
#         }

#     return dataset_split.map(format_2wiki_item, load_from_cache_file=False)



def _preprocess_hotpotqa(dataset_split: Dataset) -> Dataset:
    """
    Standardizes the HotpotQA dataset format, robustly handling malformed context entries.
    """
    logger.info("Preprocessing HotpotQA data...")
    
    def format_item(item):
        context_parts = []
        # The context is a list of lists: [["title", ["sent1", "sent2"]], ...]
        for context_item in item.get("context", []):
            # This defensive check prevents crashes on malformed data
            if isinstance(context_item, list) and len(context_item) == 2:
                title, sentences = context_item
                if isinstance(sentences, list):
                    context_parts.append(f"Title: {title}\n{' '.join(sentences)}")
        
        return {
            'id': str(item.get('_id') or item.get('id', '')),
            'question': item['question'],
            'answer': item['answer'],
            'context': "\n\n".join(context_parts),
            'supporting_facts': item.get('supporting_facts', [])
        }
    return dataset_split.map(format_item, load_from_cache_file=False)




def _load_local_hotpotqa(cache_dir: Path) -> Dataset:
    """
    Robustly loads the HotpotQA dataset.
    It first tries to download from the Hub. If it fails, it falls back
    to loading from local JSON files.
    """
    # --- Attempt 1: Online Method ---
    
    # --- Attempt 2: Offline Fallback Method ---
    logger.info("Manually loading HotpotQA from local JSON files...")
    
    def _sanitize_hotpotqa_data(data: list) -> list:
        """Helper function to ensure all parts of supporting_facts are strings."""
        for item in data:
            if 'supporting_facts' in item and item['supporting_facts'] is not None:
                item['supporting_facts'] = [
                    [str(part) for part in fact] for fact in item['supporting_facts']
                ]
            # Also sanitize the context field, which can cause similar issues
            if 'context' in item and item['context'] is not None:
                item['context'] = [
                    [str(part) for part in para] for para in item['context']
                ]
        return data
    
    hotpotqa_features = Features({
        '_id': Value('string'),
        'question': Value('string'),
        'answer': Value('string'),
        'supporting_facts': Sequence(Sequence(Value('string'))),
        'context': Sequence(Sequence(Value('string'))),
        'type': Value('string'),
        'level': Value('string')
    })
    

    try:
        train_file = cache_dir / "hotpot_qa" / "hotpot_train_v1.1.json"
        validation_file = cache_dir / "hotpot_qa" / "hotpot_dev_fullwiki_v1.json"
        # validation_file = cache_dir / "hotpot_qa" / "hotpot_dev_fullwiki_v1.json"
        with open(train_file, "r", encoding="utf-8") as f:
            train_data = _sanitize_hotpotqa_data(json.load(f))
        with open(validation_file, "r", encoding="utf-8") as f:
            validation_data = _sanitize_hotpotqa_data(json.load(f))

        # Manually create Dataset objects from the loaded lists
        train_dataset = Dataset.from_list(train_data, features=hotpotqa_features)
        validation_dataset = Dataset.from_list(validation_data, features=hotpotqa_features)
        # Combine them into a DatasetDict, which is what load_dataset usually returns
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset
        })
        return dataset
    except Exception as e_local:
        logger.error(f"FATAL: Local file loading failed for HotpotQA. Error: {e_local}")
        raise e_local





def load_dataset_by_name(dataset_name: str, base_data_path: Path, split: str = 'test', limit: int = -1):
    """
    Loads a specified dataset using the Hugging Face datasets library,
    configured with a specific proxy.
    """
    logger.info(f"Attempting to load dataset: '{dataset_name}' via proxy.")
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from {SUPPORTED_DATASETS}.")

     # Read username and password from environment variables
    # proxy_user = os.environ.get("PROXY_USER")
    # proxy_pass = os.environ.get("PROXY_PASS")
    
    # if not proxy_user or not proxy_pass:
    #     logger.error("PROXY_USER and PROXY_PASS environment variables are not set.")
    #     raise ValueError("Please set PROXY_USER and PROXY_PASS in your run script.")

    # Build the authenticated proxy URL
    # proxy_host = "httpproxy-tcop.vip.ebay.com:80"
    # proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}"
    
    # # --- Proxy Configuration ---
    proxy_url = "http://httpproxy-tcop.vip.ebay.com:80"
    d_config = DownloadConfig(proxies={'http': proxy_url, 'https': proxy_url})
    
    cache_dir = base_data_path / "hf_cache"

    try:
        # --- UPDATED SECTION ---
        if dataset_name == 'google_math':
            # Direct replacement for the old math dataset
            dataset = load_dataset("google/math", cache_dir=str(cache_dir), download_config=d_config)
            data_to_process = dataset['test']

        elif dataset_name == 'math_500': # Give it a new name
            # Use the H4 dataset
            dataset = load_dataset("HuggingFaceH4/MATH-500", cache_dir=str(cache_dir), download_config=d_config)
            data_to_process = dataset['test'] # It has train and test splits

        elif dataset_name == 'gsm8k':
            # New standard math reasoning benchmark
            dataset = load_dataset("gsm8k", "main", cache_dir=str(cache_dir), download_config=d_config)
            data_to_process = dataset['test']
        
        elif dataset_name == 'hotpotqa':
            try:
                dataset = load_dataset("hotpot_qa", "distractor", cache_dir=str(cache_dir), download_config=d_config, trust_remote_code=True)
                data_to_process = dataset['validation']
            except Exception as e:
                logger.warning(f"Online loading for HotpotQA failed: {e}. Falling back to local file method.")
            # dataset = load_dataset("json", 
            #         data_files={
            #         "train": str(cache_dir / "hotpot_qa" / "hotpot_train_v1.1.json"),
            #         "validation": str(cache_dir / "hotpot_qa" / "hotpot_dev_distractor_v1.json")
            #         }
            #         )
            
            dataset = _load_local_hotpotqa(cache_dir)
            data_to_process = dataset['validation']
            data_to_process = _preprocess_hotpotqa(data_to_process)
        elif dataset_name == 'musique':
            dataset = load_dataset("dgslibisey/MuSiQue", cache_dir=str(cache_dir), download_config=d_config)
            data_to_process = dataset['validation']
            data_to_process = _preprocess_musique(data_to_process)
            # --- END OF FIX ---
        elif dataset_name == '2wikimultihopqa':
            # --- ADD THIS BLOCK FOR THE NEW DATASET ---
            dataset = load_dataset("xanhho/2wikimultihopqa", cache_dir=str(cache_dir), download_config=d_config, trust_remote_code=True)
            data_to_process = dataset['validation']
            data_to_process = _preprocess_2wikimultihopqa(data_to_process)

        # --- END OF UPDATED SECTION ---

    # --- FAIRNESS FIX: Shuffle the dataset with a fixed seed ---
    # This ensures that for a given dataset, every experimental run
    # gets the exact same subset of questions when a limit is applied.
    # The number 42 is a standard convention for a fixed random seed.    
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}'. Error: {e}")
        raise
    
     

    
    logger.info(f"Successfully loaded split from '{dataset_name}'. Contains {len(data_to_process)} examples.")

    logger.info("Shuffling dataset with a fixed seed 42 to ensure fair sampling.")
    data_to_process = data_to_process.shuffle(seed=42)
    # --- END OF FIX ---
    
    if limit > 0 and len(data_to_process) > limit:
        logger.info(f"Limiting dataset to {limit} examples.")
        return data_to_process.select(range(limit))
        
    return data_to_process


