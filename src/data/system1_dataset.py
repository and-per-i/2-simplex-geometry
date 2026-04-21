import os
import torch
import pandas as pd
import re
from torch.utils.data import Dataset

class System1Dataset(Dataset):
    """
    Pivot Dataset for 'System 1' training with Iterative Splitting:
    For a problem with N auxiliary points, generates N training samples.
    
    Train on Completions Only: labels are masked with -100 for the prompt part.
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
            print(f"📦 Loading and Splitting Parquet dataset: {path}")
            df = pd.read_parquet(path)
            
            # Filter for proofs that HAVE auxiliary constructions
            mask = df["solution"].str.contains("<aux>", na=False)
            df_filtered = df[mask].copy()
            
            for _, row in df_filtered.iterrows():
                question = row["question"]
                solution = row["solution"]
                
                # 1. Extract and split <aux> constructions
                aux_match = re.search(r"<aux>(.*?)</aux>", solution)
                if not aux_match: continue
                aux_full_content = aux_match.group(1).strip()
                
                # Split by ';' to get individual constructions
                # Format is usually: x00 point : constructor [tag] ;
                # We remove the x00 start token if present in each split
                raw_constructions = [c.strip() for c in aux_full_content.split(";") if ":" in c]
                
                # 2. Extract <proof> steps
                proof_match = re.search(r"<proof>(.*?)</proof>", solution)
                all_proof_steps = []
                if proof_match:
                    all_proof_steps = [s.strip() for s in proof_match.group(1).split(";") if s.strip()]
                
                # 3. Iterative Splitting Logic
                accumulated_constructions = []
                for i, target_construction in enumerate(raw_constructions):
                    # Identify the point introduced in THIS construction
                    # e.g. "j : coll e g j [011]" -> point is "j"
                    point_match = re.search(r"(\w)\s:", target_construction)
                    if not point_match: continue
                    target_point = point_match.group(1)
                    
                    # Deductions found so far (System 2):
                    # They are steps that DO NOT use the target_point or any subsequent aux points
                    future_points = []
                    for future_c in raw_constructions[i:]:
                        p_match = re.search(r"(\w)\s:", future_c)
                        if p_match: future_points.append(p_match.group(1))
                    
                    pre_steps = []
                    for step in all_proof_steps:
                        # Step is valid as context if it doesn't mention the point we are about to introduce
                        # or any points that haven't been introduced yet.
                        if not any(pt in step for pt in future_points):
                            pre_steps.append(step)
                    
                    # Construct Prompt: Question + Previous Aux + Pre-steps
                    prev_aux_str = " ; ".join(accumulated_constructions)
                    if prev_aux_str:
                        prev_aux_str = f" <prev_aux> {prev_aux_str} </prev_aux>"
                    
                    input_prompt = f"{question}{prev_aux_str} <pre_steps> {' ; '.join(pre_steps)} </pre_steps>"
                    
                    # Target: The single construction (cleaned)
                    target_text = target_construction.replace("x00", "").strip()
                    
                    self.samples.append({
                        "input": input_prompt,
                        "target": target_text
                    })
                    
                    # Add this construction to context for next point
                    accumulated_constructions.append(target_text)
            
            print(f"✨ Splitting complete. Generated {len(self.samples)} samples from {len(df_filtered)} proofs.")
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input"]
        target_text = sample["target"]
        
        # Tokenize target
        target_encoding = self.tokenizer(target_text, add_special_tokens=False)
        target_ids = target_encoding["input_ids"]
        
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            target_ids = target_ids + [self.tokenizer.eos_token_id]
            
        # Max space for input
        max_input_len = self.max_length - len(target_ids) - (1 if self.add_bos else 0)
        
        # Tokenize input
        input_encoding = self.tokenizer(input_text, add_special_tokens=False)
        input_ids = input_encoding["input_ids"]
        
        # Truncate input from the LEFT (preserve recent context)
        if len(input_ids) > max_input_len:
            input_ids = input_ids[-max_input_len:]
            
        # Combine IDs
        full_input_ids = input_ids + target_ids
        
        # Labels: -100 for input (MASKED), original IDs for target
        labels = [-100] * len(input_ids) + target_ids
        
        if self.add_bos and self.tokenizer.bos_token_id is not None:
            full_input_ids = [self.tokenizer.bos_token_id] + full_input_ids
            labels = [-100] + labels
            
        # Final safety truncation/padding
        full_input_ids = full_input_ids[:self.max_length]
        labels = labels[:self.max_length]
        attention_mask = [1] * len(full_input_ids)
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
