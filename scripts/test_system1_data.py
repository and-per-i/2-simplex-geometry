import sys
import os
import torch
from transformers import AutoTokenizer

# Aggiunge src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.system1_dataset import System1Dataset
from tokenizer.hf_tokenizer import load_tokenizer

def test_pivot_dataset():
    tokenizer_path = "tokenizer/weights/geometry.757.model"
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    dataset_path = "data/curriculum/stage_1_very_easy.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print("--- Testing System1Dataset Pivot ---")
    dataset = System1Dataset(dataset_path, tokenizer, max_length=512)
    
    if len(dataset) == 0:
        print("No samples found in dataset (maybe no <aux> tags?).")
        return

    sample = dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Input IDs length: {len(input_ids)}")
    print(f"Labels length: {len(labels)}")
    
    # Check labels: should have many -100 at the beginning
    neg_100_count = labels.count(-100)
    print(f"Number of -100 labels (prompt): {neg_100_count}")
    
    # Decode the part where labels are NOT -100
    target_ids = [ids for ids in labels if ids != -100]
    decoded_target = tokenizer.decode(target_ids)
    print(f"Decoded Target: {decoded_target}")
    
    # Decode the full input
    decoded_full = tokenizer.decode(input_ids)
    print(f"Full Sequence: {decoded_full[:200]}...")

if __name__ == "__main__":
    test_pivot_dataset()
