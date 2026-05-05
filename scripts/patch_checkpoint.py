"""
patch_checkpoint.py — Ripara il checkpoint V2 dai danni del kernel Triton backward.

Il kernel Triton backward buggato (commits 17abebf, 7969be8 per il fix) produceva
gradienti con magnitudo errata durante il training delle fasi di distillazione.
Questo ha corrotto progressivamente:

  - layers.*.ln2.weight  → gamma esplosi fino a 142x (reset a 1.0)
  - token_embedding.weight → std 30x sopra il normale (rescale a 0.02)
  - lm_head.weight         → std 31x sopra il normale (rescale a 0.02)

74 tensori su 83 sono integri. Questo script sistema i 9 corrotti preservando
le direzioni apprese dall'embedding/LM head.

Il rescaling preserva il forward perché LayerNorm ricalcola la propria
normalizzazione sugli input. Il backward torna stabile con magnitudo normale.

Uso
---
  python scripts/patch_checkpoint.py \
      --input checkpoints/simplex-geometry_Final_v2.pt \
      --output checkpoints/simplex-geometry_Final_v2_patched.pt
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


def patch_checkpoint(input_path: str, output_path: str, embed_std: float = 0.02):
    print(f"📦 Caricamento checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print(f"❌ Formato non riconosciuto: {type(ckpt).__name__}")
        sys.exit(1)

    total = len(ckpt)
    fixed_ln = []
    fixed_embed = []
    fixed_lm_head = []

    for key in sorted(ckpt):
        tensor = ckpt[key]
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 1:
            continue

        std_val = tensor.float().std().item()
        max_val = tensor.float().abs().max().item()

        # ── LayerNorm gamma esplosi ──────────────────────────────────────
        if ("ln" in key and "weight" in key and max_val > 3.0):
            old_max = max_val
            fixed_ln.append((key, old_max))

        # ── token_embedding gonfiato ─────────────────────────────────────
        elif key == "token_embedding.weight" and std_val > 1.0:
            old_std = std_val
            ckpt[key] = tensor.float() * (embed_std / old_std)
            fixed_embed.append((key, old_std, ckpt[key].float().std().item()))

        # ── lm_head gonfiato ─────────────────────────────────────────────
        elif key == "lm_head.weight" and std_val > 1.0:
            old_std = std_val
            ckpt[key] = tensor.float() * (embed_std / old_std)
            fixed_lm_head.append((key, old_std, ckpt[key].float().std().item()))

    # ── Applica fix LayerNorm (in un secondo passaggio per sicurezza) ────
    for key, old_max in fixed_ln:
        ckpt[key] = torch.ones_like(ckpt[key])

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n🔍 Analisi completata: {total} chiavi totali")
    print(f"   LN corrotti:          {len(fixed_ln)}")
    print(f"   token_embedding:      {len(fixed_embed)}")
    print(f"   lm_head:              {len(fixed_lm_head)}")
    print(f"   Pesi intatti:         {total - len(fixed_ln) - len(fixed_embed) - len(fixed_lm_head)}")

    if fixed_ln:
        print(f"\n🔧 LayerNorm con gamma esploso (reset a 1.0):")
        for key, old_max in fixed_ln:
            print(f"   {key}: max was {old_max:.1f} → reset a 1.0")

    if fixed_embed:
        print(f"\n🔧 token_embedding.weight riscalato:")
        for key, old_std, new_std in fixed_embed:
            factor = old_std / new_std
            print(f"   std: {old_std:.1f} → {new_std:.4f}  (÷{factor:.0f})")

    if fixed_lm_head:
        print(f"\n🔧 lm_head.weight riscalato:")
        for key, old_std, new_std in fixed_lm_head:
            factor = old_std / new_std
            print(f"   std: {old_std:.1f} → {new_std:.4f}  (÷{factor:.0f})")

    # ── Salva ────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"\n✅ Checkpoint patchato salvato: {output_path}")

    # ── Verifica rapida ──────────────────────────────────────────────────
    ckpt2 = torch.load(output_path, map_location="cpu")
    all_ok = True
    for key in sorted(ckpt2):
        tensor = ckpt2[key]
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 1:
            continue
        std_val = tensor.float().std().item()
        max_val = tensor.float().abs().max().item()
        if ("ln" in key and "weight" in key and max_val > 3.0):
            print(f"   ❌ {key}: max={max_val:.1f} ANCORA CORROTTO")
            all_ok = False
        if key in ("token_embedding.weight", "lm_head.weight") and std_val > 1.0:
            print(f"   ❌ {key}: std={std_val:.1f} ANCORA CORROTTO")
            all_ok = False
    if all_ok:
        print("   ✅ Verifica: nessun peso anomalo rilevato")


def main():
    parser = argparse.ArgumentParser(
        description="Ripara checkpoint V2 dai danni del kernel Triton backward"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Percorso del checkpoint da riparare")
    parser.add_argument("--output", type=str, required=True,
                        help="Percorso dove salvare il checkpoint patchato")
    parser.add_argument("--embed_std", type=float, default=0.02,
                        help="Target std per embedding/LM head (default: 0.02)")
    args = parser.parse_args()

    patch_checkpoint(args.input, args.output, args.embed_std)


if __name__ == "__main__":
    main()
