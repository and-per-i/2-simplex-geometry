"""
finetune_imo.py — Pipeline unificata curriculum IMO con un singolo Trainer.

Rispetto a curriculum_finetune.py (che lancia N Trainer separati), questo script
usa un Trainer singolo con CurriculumSchedulerCallback che ricostruisce il dataset
all'inizio di ogni epoca secondo lo schedule definito in EPOCH_SCHEDULE.

Vantaggi:
  - Unico run wandb/tensorboard — curva di loss continua
  - LR cosine schedule uniforme su tutte le epoch
  - Nessun overhead di re-inizializzazione dell'ottimizzatore

Schema epoch default (6 epoch totali):
  Epoch 0  max_len=512   stage 1+2+3 ugualmente pesati   — fondamenta
  Epoch 1  max_len=640   stage 4 (70%) + 3 (20%) + 1 (10%)  — ponte
  Epoch 2  max_len=896   stage 5 (75%) + 2 (15%) + 1 (10%)  — IMO push
  Epoch 3  max_len=896   (idem epoch 2)                       — consolidamento push
  Epoch 4  max_len=1024  stage 5 (85%) + 3 (10%) + 1 (5%)   — consolidamento finale
  Epoch 5  max_len=1024  (idem epoch 4)

Uso
---
  # Training completo
  python scripts/finetune_imo.py \\
      --checkpoint checkpoints/simplex-geometry_Final_v2.pt \\
      --data_dir data \\
      --output_dir runs/imo_unified

  # Test rapido (100 campioni per stage, 2 epoch)
  python scripts/finetune_imo.py \\
      --checkpoint checkpoints/simplex-geometry_Final_v2.pt \\
      --data_dir data \\
      --output_dir runs/test \\
      --total_epochs 2 \\
      --max_samples_per_stage 100
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.models.checkpoint_loader import load_checkpoint
from src.data.curriculum_dataset import DynamicCurriculumDataset, EPOCH_SCHEDULE
from tokenizer.hf_tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class CurriculumSchedulerCallback(TrainerCallback):
    """Ricostruisce il dataset all'inizio di ogni epoca con il mix appropriato."""

    def __init__(self, dataset: DynamicCurriculumDataset):
        self.dataset = dataset

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.dataset.set_epoch(epoch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum fine-tuning IMO con singolo Trainer"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Checkpoint di partenza (.pt Phase 1/2 o directory HF)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory contenente curriculum/stage_*.parquet"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/imo_unified",
        help="Directory di output per checkpoint e log"
    )
    parser.add_argument(
        "--total_epochs", type=int, default=6,
        help="Numero totale di epoch (default=6, mappa allo schedule 0-5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size per device"
    )
    parser.add_argument(
        "--grad_acc", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5,
        help="Learning rate massimo (cosine schedule)"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Frazione di step per il warmup LR"
    )
    parser.add_argument(
        "--max_samples_per_stage", type=int, default=None,
        help="Cap campioni per stage (debug/test rapido)"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    bf16 = device in ("cuda", "mps")
    print(f"🚀 Device: {device}  BF16: {bf16}")

    # Tokenizer
    tok_path = ROOT_DIR / "tokenizer" / "weights" / "geometry.757.model"
    tokenizer = load_tokenizer(str(tok_path), vocab_size=1024)
    print(f"📝 Tokenizer caricato (vocab={tokenizer.vocab_size})")

    # Modello
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else str(ROOT_DIR / args.checkpoint)
    print(f"📦 Caricamento checkpoint: {ckpt_path}")
    model = load_checkpoint(ckpt_path, device=device)
    if bf16:
        model = model.to(torch.bfloat16)
    model.train()

    max_pos = model.config.max_position_embeddings
    print(f"   Parametri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Config: {model.config.num_hidden_layers}L, hidden={model.config.hidden_size}, "
          f"max_pos={max_pos}, simplex_layers={model.config.simplex_layers}")

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else str(ROOT_DIR / args.data_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else str(ROOT_DIR / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Dataset — carica tutti gli stage una volta sola
    print(f"\n📚 Caricamento dataset da {data_dir}/curriculum/ ...")
    dataset = DynamicCurriculumDataset.load_from_dir(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_pos=max_pos,
        max_samples_per_stage=args.max_samples_per_stage,
        seed=args.seed,
    )
    print(f"   Dataset epoch 0: {len(dataset):,} campioni  max_length={dataset.max_length}")

    # Stampa schedule
    print(f"\n📋 Schedule curriculum ({args.total_epochs} epoch):")
    for ep in range(args.total_epochs):
        weights, raw_ml = EPOCH_SCHEDULE[min(ep, len(EPOCH_SCHEDULE) - 1)]
        ml = min(raw_ml, max_pos)
        mix_str = " + ".join(f"{s.split('_')[1]}({w:.0%})" for s, w in weights.items())
        print(f"   Epoch {ep}: max_len={ml}  {mix_str}")

    # Training arguments — bf16 usato solo se CUDA o MPS disponibile
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.total_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=bf16,
        fp16=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_drop_last=True,
    )

    callback = CurriculumSchedulerCallback(dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DynamicCurriculumDataset.collate_fn
            if hasattr(DynamicCurriculumDataset, "collate_fn")
            else _collate_fn,
        callbacks=[callback],
    )

    print(f"\n🔥 Avvio training IMO curriculum ({args.total_epochs} epoch)...")
    trainer.train()

    # Salva modello finale
    final_path = os.path.join(output_dir, "final_imo_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n🏆 Training completato! Modello finale: {final_path}")
    print(f"   Inferenza: python scripts/cloud_deep_search.py --checkpoint {final_path}")


def _collate_fn(batch):
    """Dynamic padding fallback (usato se DynamicCurriculumDataset non eredita collate_fn)."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids      = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L]      = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        labels[i, :L]         = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    main()
