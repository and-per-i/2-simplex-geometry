import os
import sys
import torch
import argparse
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.data.finetune_dataset import FinetuneDataset
from tokenizer.hf_tokenizer import load_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Phase 3 for 2-simplex model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint from Phase 2")
    parser.add_argument("--dataset_path", type=str, default="data/hard_dataset/olympiad_problems.txt", help="Path to the hard dataset")
    parser.add_argument("--output_dir", type=str, default="./alphageometry-edge-finetuned", help="Output directory")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (should be very low)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_acc", type=int, default=8, help="Gradient accumulation steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. Load Tokenizer
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/weights/geometry.757.model"))
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"📝 Tokenizer loaded (Vocab: {tokenizer.vocab_size})")

    # 2. Load Model (Pre-trained from Phase 2)
    print(f"📦 Loading pre-trained model from {args.model_path}...")
    config = StudentConfig.from_pretrained(args.model_path)
    
    # Cloud GPU optimization: Enable Triton for 2-simplicial attention
    config.use_simplex_attention = True
    config.use_triton = True
    
    model = StudentForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16  # Optimized for 5090 (Ampere/Blackwell)
    ).to(device)
    
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

    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

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
        save_strategy="epoch",
        logging_steps=1,             # More frequent logging
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=True, # Extra safety for memory
        optim="adamw_torch_fused",   # Faster optimizer
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hard_dataset,
        data_collator=data_collator,
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
