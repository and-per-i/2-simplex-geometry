"""
Script di calibrazione Q-Filters — da eseguire una sola volta dopo il training.

Esegue forward pass su un dataset geometrico, calcola la SVD delle Query
per ogni layer standard e salva i vettori filtro (Q-Filters) su disco.

Uso:
    python scripts/calibrate_q_filters.py \
        --checkpoint checkpoints/simplex-geometry_Final.pt \
        --data_path data/finetune_clean.parquet \
        --n_samples 512 \
        --output checkpoints/q_filters.pt

I Q-Filters vengono automaticamente cercati al caricamento del modello in
cloud_deep_search.py (cerca <checkpoint_dir>/q_filters.pt).
"""

import os
import sys
import argparse
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.models.checkpoint_loader import load_from_phase1
from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.cache.q_filters import calibrate_q_filters, save_q_filters
from tokenizer.hf_tokenizer import load_tokenizer


def load_calibration_sequences(data_path: str, tokenizer, n_samples: int, max_length: int = 256):
    """Carica sequenze di calibrazione dal dataset geometrico."""
    from pathlib import Path
    path = Path(data_path)
    sequences = []

    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(data_path)
        # Colonne attese: 'question' e/o 'solution'
        text_col = "question" if "question" in df.columns else df.columns[0]
        texts = df[text_col].dropna().tolist()
    elif path.suffix == ".txt":
        with open(data_path) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Formato non supportato: {path.suffix} (usa .parquet o .txt)")

    for text in texts[:n_samples]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].squeeze(0)   # (S,)
        if ids.shape[0] >= 4:               # scarta sequenze troppo corte
            sequences.append(ids)

    print(f"✅ {len(sequences)} sequenze di calibrazione caricate da {data_path}")
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Calibra Q-Filters per KV cache eviction")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path al checkpoint: Phase 1 .pt oppure directory HF Phase 2"
    )
    parser.add_argument(
        "--data_path", type=str,
        default=str(ROOT_DIR / "data" / "finetune_clean.parquet"),
        help="Dataset geometrico per calibrazione (.parquet o .txt)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=512,
        help="Numero di sequenze da usare per la calibrazione (default: 512)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path di output per q_filters.pt (default: accanto al checkpoint)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Lunghezza massima sequenza di calibrazione"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Dispositivo: {device}")

    # --- Carica modello ---
    checkpoint_path = Path(args.checkpoint)
    print(f"📦 Carico modello da {checkpoint_path}...")
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        model = load_from_phase1(str(checkpoint_path), device=device)
    else:
        config = StudentConfig.from_pretrained(str(checkpoint_path))
        model = StudentForCausalLM.from_pretrained(str(checkpoint_path), config=config)
        model = model.to(device)
    model.eval()

    # --- Carica tokenizer ---
    tok_path = ROOT_DIR / "tokenizer" / "weights" / "geometry.757.model"
    tokenizer = load_tokenizer(str(tok_path), vocab_size=1024)
    print(f"📝 Tokenizer caricato (vocab={tokenizer.vocab_size})")

    # --- Carica sequenze di calibrazione ---
    sequences = load_calibration_sequences(
        args.data_path, tokenizer, args.n_samples, args.max_length
    )

    if not sequences:
        print("❌ Nessuna sequenza di calibrazione caricata. Controlla --data_path.")
        sys.exit(1)

    # --- Calibrazione Q-Filters ---
    print(f"🔍 Calibrazione Q-Filters su {len(sequences)} sequenze...")
    filters = calibrate_q_filters(model, sequences, n_samples=len(sequences), device=device)

    if not filters:
        print("❌ Nessun Q-Filter prodotto. Verifica l'architettura del modello.")
        sys.exit(1)

    # --- Salva ---
    if args.output:
        output_path = args.output
    else:
        if checkpoint_path.is_file():
            output_path = str(checkpoint_path.parent / "q_filters.pt")
        else:
            output_path = str(checkpoint_path / "q_filters.pt")

    save_q_filters(filters, output_path)
    print(f"\n✨ Calibrazione completata. Q-Filters per {len(filters)} layer salvati in:")
    print(f"   {output_path}")
    print("\nUso in inferenza:")
    print(f"   python scripts/cloud_deep_search.py --checkpoint {args.checkpoint} --token_budget 16")


if __name__ == "__main__":
    main()
