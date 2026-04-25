"""
Fine-Tuning Post-Distillazione — Self-Contained
================================================

Script autonomo per il fine-tuning di Davide_8L_distilled.pt.
Non dipende da moduli interni: tutto il necessario è inline.

Prerequisiti:
    - modello_distillato/Davide_8L_distilled.pt
    - modello_distillato/tokenizer/vocab.model
    - modello_distillato/student_progressive.py
    - data/finetune_clean.parquet  (generato da preprocess_dataset.py)

Usage (Mac M4 — inferenza/test):
    python modello_distillato/finetune_clean.py \\
        --model_path modello_distillato/Davide_8L_distilled.pt \\
        --data_path data/finetune_clean.parquet \\
        --output_dir runs/finetune_clean \\
        --num_epochs 2

Usage (Cloud RTX 5090 Ti — training reale):
    python finetune_clean.py \\
        --model_path Davide_8L_distilled.pt \\
        --data_path data/finetune_clean.parquet \\
        --output_dir runs/finetune_clean \\
        --num_epochs 2 \\
        --fp16
"""

import sys
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# ─── Aggiungi modello_distillato/ al path per importare student_progressive ───
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from student_progressive import StudentModelProgressive

# ─── Aggiungi il tokenizer dal path corretto ───────────────────────────────────
TOKENIZER_PATH = str(SCRIPT_DIR / "tokenizer" / "vocab.model")
sys.path.insert(0, str(SCRIPT_DIR))
from tokenizer.hf_tokenizer import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune_clean.log"),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset inline — legge la colonna "text" dal parquet preprocessato
# ══════════════════════════════════════════════════════════════════════════════
class FinetuneDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512):
        df = pd.read_parquet(parquet_path)

        # Supporta sia colonna "text" (output di preprocess_dataset.py)
        # sia il vecchio formato "question"+"solution"
        if "text" in df.columns:
            texts = df["text"].dropna().tolist()
        else:
            logger.warning("Colonna 'text' non trovata — uso question+solution")
            texts = (df["question"].fillna("") + " " + df["solution"].fillna("")).tolist()

        logger.info(f"Dataset: {len(texts):,} campioni da {parquet_path}")

        self.examples = []
        skipped = 0
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length - 1,
                padding=False,
                return_tensors=None,
            )
            ids = enc["input_ids"]
            # IMPORTANTE: Aggiungiamo l'EOS token così impara a fermarsi!
            ids.append(tokenizer.eos_token_id)
            
            if len(ids) < 4:          # scarta sequenze troppo corte
                skipped += 1
                continue
            self.examples.append(torch.tensor(ids, dtype=torch.long))

        logger.info(f"Esempi validi: {len(self.examples):,} ({skipped} scartati)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ══════════════════════════════════════════════════════════════════════════════
# Collator inline — padding dinamico
# ══════════════════════════════════════════════════════════════════════════════
class FinetuneCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        labels    = [b["labels"]    for b in batch]
        max_len   = max(x.size(0) for x in input_ids)

        padded_ids    = []
        padded_labels = []
        for ids, lbl in zip(input_ids, labels):
            pad_len = max_len - ids.size(0)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_token_id)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100)]))

        return {
            "input_ids":      torch.stack(padded_ids),
            "labels":         torch.stack(padded_labels),
            "attention_mask": (torch.stack(padded_ids) != self.pad_token_id).long(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Trainer CE-only (no Knowledge Distillation)
# ══════════════════════════════════════════════════════════════════════════════
class FineTuneTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default=str(SCRIPT_DIR / "Davide_8L_distilled.pt"))
    parser.add_argument("--data_path",  type=str,
                        default="data/finetune_clean.parquet")
    parser.add_argument("--output_dir", type=str,
                        default="runs/finetune_clean")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--fp16", action="store_true",
                        help="Abilita FP16 (solo su GPU CUDA, NON su Mac MPS)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"\n🎯 FINE-TUNING POST-DISTILLAZIONE")
    logger.info(f"   Device:  {device}")
    logger.info(f"   Model:   {args.model_path}")
    logger.info(f"   Data:    {args.data_path}")
    logger.info(f"   Output:  {args.output_dir}")
    logger.info(f"   Epochs:  {args.num_epochs}")
    logger.info(f"   FP16:    {args.fp16}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"\n🔑 Caricamento tokenizer: {TOKENIZER_PATH}")
    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=1024)
    logger.info(f"   vocab_size={tokenizer.vocab_size}  BOS={tokenizer.bos_token_id}  EOS={tokenizer.eos_token_id}")

    # ── Modello ───────────────────────────────────────────────────────────────
    logger.info(f"\n📦 Caricamento modello: {args.model_path}")
    # IMPORTANTE: il checkpoint ha 8 layer fisici (0-7), non 12.
    # I layer bypassed (1,2,8,9 nell'originale) sono già stati rimossi
    # dalla Fase 4 di estrazione. I simplicial layer rimappano a [3, 7].
    model = StudentModelProgressive(
        vocab_size=1024,
        dim_hidden=384,
        num_layers=8,
        simplicial_layers=[3, 7],
    )
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    bypassed = [i + 1 for i, l in enumerate(model.layers) if l.is_bypassed]
    active   = [i + 1 for i, l in enumerate(model.layers) if not l.is_bypassed]
    total_p  = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parametri: {total_p / 1e6:.2f}M")
    logger.info(f"   Layer attivi:   {active}")
    logger.info(f"   Layer bypassed: {bypassed} (nessun gradiente li attraversa)")
    model.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"\n📚 Caricamento dataset: {args.data_path}")
    dataset = FinetuneDataset(args.data_path, tokenizer, max_length=512)
    collator = FinetuneCollator(pad_token_id=tokenizer.pad_token_id)

    # ── Training Args ─────────────────────────────────────────────────────────
    # NOTA: fp16 solo su CUDA (RTX 5090 Ti). Su Mac MPS → False sempre.
    use_fp16 = args.fp16 and device == "cuda"

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,          # batch virtuale = 64

        learning_rate=1e-4,                     # Aumentato per ri-addestramento attenzione causale
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=0.5,

        warmup_steps=1000,                      # Più warmup visto il dataset grande
        lr_scheduler_type="cosine",

        fp16=use_fp16,
        bf16=False,                             # BF16 non supportato su MPS

        logging_steps=100,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=3,

        eval_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        report_to="none",

        dataloader_num_workers=4,
    )

    logger.info(f"\n⚙️  Config:")
    logger.info(f"   LR={training_args.learning_rate}  Batch effettivo=64")
    logger.info(f"   FP16={use_fp16}  Scheduler=cosine  Warmup=1000")

    # ── Avvio ─────────────────────────────────────────────────────────────────
    trainer = FineTuneTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info(f"\n🔥 AVVIO FINE-TUNING...")
    trainer.train()

    # ── Salvataggio ───────────────────────────────────────────────────────────
    final_path = Path(args.output_dir) / "pytorch_model_finetuned.bin"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n✅ Fine-tuning completato!")
    logger.info(f"   Modello salvato: {final_path}")
    logger.info(f"\n💡 Prossimo step: python verify_learning.py")


if __name__ == "__main__":
    main()
