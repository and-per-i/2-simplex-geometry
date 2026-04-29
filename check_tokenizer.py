"""
check_tokenizer.py — Verifica tokenizer prima del fine-tuning

Lancia questo script PRIMA di qualsiasi training nel nuovo progetto.
Se passa tutti gli assert, il tokenizer è identico a quello usato in distillazione.

Usage:
    python check_tokenizer.py

Requisiti:
    - modello_distillato/tokenizer/vocab.model  (14 KB, copiato da pt_ckpt/vocab.model del cloud)
    - modello_distillato/tokenizer/hf_tokenizer.py  (copiato dal progetto simplex-distillery)
"""

import sys

# ─────────────────────────────────────────────────────────────
# 1. Import
# ─────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, "modello_distillato")
    from tokenizer.hf_tokenizer import load_tokenizer
except ImportError as e:
    print(f"❌ ERRORE IMPORT: {e}")
    print("   Assicurati che modello_distillato/tokenizer/hf_tokenizer.py esista nel progetto.")
    print("   (E di aver installato 'sentencepiece' nel tuo venv: pip install sentencepiece)")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# 2. Caricamento
# ─────────────────────────────────────────────────────────────
VOCAB_MODEL_PATH = "tokenizer/weights/geometry.757.model"

try:
    tok = load_tokenizer(VOCAB_MODEL_PATH, vocab_size=1024)
except Exception as e:
    print(f"❌ ERRORE caricamento tokenizer: {e}")
    print(f"   Verifica che {VOCAB_MODEL_PATH} esista (14 KB, copiato dal cloud).")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# 3. Verifica token ancora (devono matchare esattamente)
# ─────────────────────────────────────────────────────────────
ANCHOR_IDS = {
    "▁a": 261,
    "▁b": 260,
    "▁c": 262,
    "▁d": 264,
    "▁e": 265,
    "▁;": 263,
    "▁:": 266,
}

print("=" * 55)
print("  TOKEN ANCORA — devono matchare la distillazione")
print("=" * 55)

errors = []
for piece, expected_id in ANCHOR_IDS.items():
    actual_id = tok.sp_model.PieceToId(piece)
    ok = actual_id == expected_id
    status = "✅" if ok else "❌ MISMATCH!"
    print(f"  {piece:8s} → ID {actual_id:4d}  (atteso: {expected_id})  {status}")
    if not ok:
        errors.append(f"Token '{piece}': atteso {expected_id}, trovato {actual_id}")

# ─────────────────────────────────────────────────────────────
# 4. Verifica parametri fissi
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  PARAMETRI FISSI DEL TOKENIZER")
print("=" * 55)

CHECKS = [
    ("SP tokens reali",    tok.sp_model.GetPieceSize(), 757),
    ("vocab_size override",tok.vocab_size,              1024),
    ("bos_token_id",       tok.bos_token_id,            2),
    ("eos_token_id",       tok.eos_token_id,            1),
    ("pad_token_id",       tok.pad_token_id,            0),
]

for label, actual, expected in CHECKS:
    ok = actual == expected
    status = "✅" if ok else "❌ MISMATCH!"
    print(f"  {label:22s} = {actual:6}  (atteso: {expected})  {status}")
    if not ok:
        errors.append(f"{label}: atteso {expected}, trovato {actual}")

# ─────────────────────────────────────────────────────────────
# 5. Risultato finale
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
if errors:
    print("❌ TOKENIZER NON VALIDO — NON avviare il fine-tuning!")
    print()
    for err in errors:
        print(f"   • {err}")
    print()
    print("   Soluzione: copia pt_ckpt/vocab.model dal cloud")
    print("   e assicurati che hf_tokenizer.py sia identico.")
    sys.exit(1)
else:
    print("✅ Tokenizer verificato — identico alla distillazione.")
    print("   Puoi procedere con il fine-tuning.")
    sys.exit(0)
