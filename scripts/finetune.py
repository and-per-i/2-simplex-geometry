import os
import sys
import torch
import argparse
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.models.checkpoint_loader import load_from_phase1, load_checkpoint
from src.data.finetune_dataset import FinetuneDataset
from tokenizer.hf_tokenizer import load_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Phase 3 for 2-simplex model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to any .pt checkpoint (Phase 1 or Phase 2 format) — auto-detected")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a Phase 2 HF checkpoint directory (from_pretrained format)")
    parser.add_argument("--phase1_checkpoint", type=str, default=None, help="Path to simplex-geometry_Final.pt produced by Phase 1 (Simplex Distillery)")
    parser.add_argument("--dataset_path", type=str, default="data/hard_dataset/olympiad_problems.txt", help="Path to the hard dataset")
    parser.add_argument("--output_dir", type=str, default="./alphageometry-edge-finetuned", help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (optimized for large batches)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device (multiplied by 4 internally)")
    parser.add_argument("--grad_acc", type=int, default=8, help="Gradient accumulation steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. Load Tokenizer
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/weights/geometry.757.model"))
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 # Default for AlphaGeometry
    print(f"📝 Tokenizer loaded (Vocab: {tokenizer.vocab_size})")

    # 2. Load Model
    if args.checkpoint:
        # Generic path: auto-detect Phase 1 vs Phase 2 format
        print(f"📦 Loading checkpoint from {args.checkpoint}...")
        model = load_checkpoint(args.checkpoint, device=device, strict=True)
        model = model.to(torch.bfloat16)
        config = model.config
    elif args.phase1_checkpoint:
        # Explicit Phase 1 path (legacy option, kept for back-compat)
        print(f"📦 Loading Phase 1 checkpoint from {args.phase1_checkpoint}...")
        model = load_from_phase1(args.phase1_checkpoint, device=device)
        model = model.to(torch.bfloat16)
        config = model.config
    elif args.model_path:
        # Phase 2 path: load a previously saved Phase 2 HF checkpoint directory
        print(f"📦 Loading Phase 2 checkpoint from {args.model_path}...")
        config = StudentConfig.from_pretrained(args.model_path)
        model = StudentForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        ).to(device)
    else:
        raise ValueError("Provide --checkpoint (any .pt), --phase1_checkpoint, or --model_path.")
    
    print(f"✅ Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Load Hard Dataset
    print(f"📂 Loading dataset from {args.dataset_path}...")
    hard_dataset = FinetuneDataset(
        path=args.dataset_path,
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings
    )
    
    if len(hard_dataset) == 0:
        print("❌ Error: Dataset is empty.")
        return

    # 4. Custom Data Collator for dynamic padding
    def custom_collator(features):
        batch = {}
        # Find max length in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        for key in ["input_ids", "attention_mask", "labels"]:
            padded_seqs = []
            for f in features:
                seq = f[key]
                if isinstance(seq, torch.Tensor):
                    seq = seq.tolist()
                
                # Pad with 0 for input_ids/mask, and -100 for labels
                pad_val = 0 if key != "labels" else -100
                padded = seq + [pad_val] * (max_len - len(seq))
                padded_seqs.append(padded)
            batch[key] = torch.tensor(padded_seqs, dtype=torch.long)
        
        return batch


    # 5. Training Arguments - Optimized for 5090 (32GB VRAM)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size * 4,  # Increased for 32GB VRAM
        gradient_accumulation_steps=args.grad_acc // 4 if args.grad_acc >= 4 else 1,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        bf16=True,                   # Best for 5090
        fp16=False,
        save_strategy="steps",
        save_steps=1000,             # Save every 1000 steps
        save_total_limit=3,          # Keep only last 3 checkpoints to save space
        logging_steps=10,            # Standard logging
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=False, # Disabled for speed (32GB VRAM is enough)
        optim="adamw_torch_fused",   # Faster optimizer
        remove_unused_columns=False, # Important for custom data
    )

    # 6. Initialize Trainer with Custom Collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hard_dataset,
        data_collator=custom_collator,
    )

    # 7. Start Fine-Tuning
    print("🔥 Starting Phase 3: Fine-Tuning...")
    trainer.train()

    # 8. Save Final Model
    print(f"💾 Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✨ Phase 3 Completed!")

if __name__ == "__main__":
    main()
