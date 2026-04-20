import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class FinetuneDataset(Dataset):
    """
    Simple dataset for fine-tuning the 2-simplex model.
    Supports .txt and .parquet formats.
    """
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
        add_bos: bool = True,
        add_eos: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.samples = []

        if not os.path.exists(path):
            print(f"⚠️ Warning: Dataset file not found at {path}")
            return

        if path.endswith(".parquet"):
            print(f"📦 Loading Parquet dataset: {path}")
            df = pd.read_parquet(path)
            # Combine question and solution
            if "question" in df.columns and "solution" in df.columns:
                self.samples = (df["question"] + " " + df["solution"]).tolist()
            else:
                # Try to find any text column if standard names are missing
                text_cols = [c for c in df.columns if df[c].dtype == "object"]
                if text_cols:
                    self.samples = df[text_cols[0]].tolist()
        else:
            with open(path, "r", encoding="utf-8") as f:
                self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Simple tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length - (2 if self.add_bos and self.add_eos else 1),
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        
        input_ids = encoding["input_ids"]
        
        if self.add_bos and self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            
        input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
