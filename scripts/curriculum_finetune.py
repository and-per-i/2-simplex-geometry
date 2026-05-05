"""
curriculum_finetune.py — Fine-tuning IMO-level con curriculum learning.

Pipeline in 4 fasi progressive che portano il modello da un livello base
a problemi di difficoltà olimpica, prevenendo il catastrophic forgetting
mediante mixing con esempi semplici nelle fasi avanzate.

Schema fasi
-----------
  Fase 0  (1 epoch)  : stage 1+2+3          — rinforzo fondamenta
  Fase 1  (1 epoch)  : stage 4+3+1           — ponte verso il difficile
  Fase 2  (2 epoch)  : stage 5+2+1 (75/15/10%) — push IMO con retention
  Fase 3  (2 epoch)  : stage 5+3+1 (85/10/5%)  — consolidamento finale

Uso tipico
----------
  # Curriculum completo (fasi 0→3)
  python scripts/curriculum_finetune.py \\
      --checkpoint checkpoints/simplex-geometry_Final_v2.pt \\
      --data_dir data \\
      --output_dir runs/imo_curriculum

  # Solo fasi finali (se il modello è già forte sulle basi)
  python scripts/curriculum_finetune.py \\
      --checkpoint checkpoints/simplex-geometry_Final_v2.pt \\
      --data_dir data \\
      --output_dir runs/imo_curriculum \\
      --start_phase 2

  # Test rapido (100 campioni per stage)
  python scripts/curriculum_finetune.py \\
      --checkpoint checkpoints/simplex-geometry_Final_v2.pt \\
      --data_dir data \\
      --output_dir runs/test \\
      --max_samples_per_stage 100 \\
      --epochs_per_phase 1
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.models.checkpoint_loader import load_checkpoint
from src.data.curriculum_dataset import CurriculumDataset, CURRICULUM_PHASES
from tokenizer.hf_tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Configurazione fasi
# ---------------------------------------------------------------------------

# Numero di epoch per ogni fase (indice = numero fase)
DEFAULT_EPOCHS = [1, 1, 2, 2]

# Learning rate per fase (decrescente: fasi avanzate richiedono LR minore)
DEFAULT_LR = [5e-5, 3e-5, 1e-5, 5e-6]

# max_length per fase — progressivo per warm-up dei positional embedding.
# Il modello è stato trainato con max_length=512: posizioni 0-511 sono sicure.
# Fasi successive estendono gradualmente il contesto verso 1024.
# Nota: superare max_position_embeddings (1024) causa IndexError immediato.
DEFAULT_MAX_LENGTH = [512, 640, 896, 1024]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum fine-tuning IMO-level per simplex-geometry"
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
        "--output_dir", type=str, default="runs/imo_curriculum",
        help="Directory di output per checkpoint e log"
    )
    parser.add_argument(
        "--start_phase", type=int, default=0,
        choices=[0, 1, 2, 3],
        help="Fase di partenza (0=fondamenta, 2=IMO push diretto)"
    )
    parser.add_argument(
        "--end_phase", type=int, default=3,
        choices=[0, 1, 2, 3],
        help="Ultima fase da eseguire (inclusa)"
    )
    parser.add_argument(
        "--epochs_per_phase", type=int, default=None,
        help="Override uniforme del numero di epoch per ogni fase"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size per device"
    )
    parser.add_argument(
        "--grad_acc", type=int, default=4,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_acc)"
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Override uniforme di max_length per tutte le fasi. "
             "Default: progressivo [512, 640, 896, 1024] per fase. "
             "Non superare max_position_embeddings del modello (solitamente 1024)."
    )
    parser.add_argument(
        "--max_samples_per_stage", type=int, default=None,
        help="Cap campioni per stage (utile per test rapidi)"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Frazione di step per il warmup LR"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Salva un checkpoint ogni N step"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50,
        help="Logga metriche ogni N step"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training di una singola fase
# ---------------------------------------------------------------------------

def train_phase(
    phase: int,
    model,
    tokenizer,
    data_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_acc: int,
    max_length: int,
    max_samples_per_stage: int | None,
    warmup_ratio: float,
    save_steps: int,
    logging_steps: int,
    seed: int,
    bf16: bool,
) -> None:
    phase_output = os.path.join(output_dir, f"phase_{phase}")
    os.makedirs(phase_output, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FASE {phase}: {_phase_name(phase)}")
    print(f"  LR={lr:.0e}  epochs={epochs}  batch={batch_size}×{grad_acc}={batch_size*grad_acc}  max_len={max_length}")
    print(f"{'='*70}\n")

    dataset = CurriculumDataset.for_phase(
        data_dir=data_dir,
        phase=phase,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples_per_stage=max_samples_per_stage,
        seed=seed,
    )

    training_args = TrainingArguments(
        output_dir=phase_output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        fp16=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CurriculumDataset.collate_fn,
    )

    trainer.train()

    # Salva il modello al termine della fase
    final_path = os.path.join(phase_output, "model_final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Fase {phase} completata. Modello salvato in: {final_path}\n")


def _phase_name(phase: int) -> str:
    names = [
        "Rinforzo fondamenta (stage 1+2+3)",
        "Ponte verso il difficile (stage 4+3+1)",
        "IMO push con retention (stage 5+2+1)",
        "Consolidamento IMO (stage 5+3+1)",
    ]
    return names[phase] if phase < len(names) else f"Fase {phase}"


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

    # Modello di partenza
    print(f"📦 Caricamento checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device)
    if bf16:
        model = model.to(torch.bfloat16)
    model.train()
    print(f"   Parametri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Config: {model.config.num_hidden_layers}L, hidden={model.config.hidden_size}, "
          f"simplex_layers={model.config.simplex_layers}")

    data_dir = os.path.join(ROOT_DIR, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(ROOT_DIR, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    epochs_schedule = DEFAULT_EPOCHS[:]
    lr_schedule = DEFAULT_LR[:]

    if args.epochs_per_phase is not None:
        epochs_schedule = [args.epochs_per_phase] * len(CURRICULUM_PHASES)

    max_len_schedule = DEFAULT_MAX_LENGTH[:]
    if args.max_length is not None:
        max_pos = model.config.max_position_embeddings
        if args.max_length > max_pos:
            raise ValueError(
                f"--max_length {args.max_length} supera max_position_embeddings={max_pos}. "
                f"Causerebbe un IndexError durante il training."
            )
        max_len_schedule = [args.max_length] * len(CURRICULUM_PHASES)
    else:
        # Clamp al limite architetturale del modello caricato
        max_pos = model.config.max_position_embeddings
        max_len_schedule = [min(ml, max_pos) for ml in DEFAULT_MAX_LENGTH]

    print(f"\n📋 Piano curriculum: fasi {args.start_phase}→{args.end_phase}")
    for ph in range(args.start_phase, args.end_phase + 1):
        print(f"   Fase {ph}: {_phase_name(ph)}  "
              f"({epochs_schedule[ph]} epoch, LR={lr_schedule[ph]:.0e}, "
              f"max_len={max_len_schedule[ph]})")

    # Esegui le fasi in sequenza
    for phase in range(args.start_phase, args.end_phase + 1):
        train_phase(
            phase=phase,
            model=model,
            tokenizer=tokenizer,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs_schedule[phase],
            lr=lr_schedule[phase],
            batch_size=args.batch_size,
            grad_acc=args.grad_acc,
            max_length=max_len_schedule[phase],
            max_samples_per_stage=args.max_samples_per_stage,
            warmup_ratio=args.warmup_ratio,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            seed=args.seed,
            bf16=bf16,
        )

    # Checkpoint finale consolidato
    final_output = os.path.join(output_dir, "final_imo_model")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\n🏆 Curriculum completato! Modello finale: {final_output}")
    print(f"   Per l'inferenza: python scripts/cloud_deep_search.py --checkpoint {final_output}")


if __name__ == "__main__":
    main()
