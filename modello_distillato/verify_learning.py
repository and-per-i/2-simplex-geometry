"""
Verifica Finale post-Finetuning
===============================

Esegue una generazione di prova sul modello appena fine-tunato
per verificare che abbia appreso il nuovo formato "clean" dei token.
"""

import sys
import torch
from pathlib import Path

# Setup paths per importare dal progetto
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer

def generate_text(model, input_ids, max_new_tokens, eos_token_id, repetition_penalty=1.2):
    """Semplice generazione auto-regressiva con repetition penalty"""
    device = input_ids.device
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            
        logits = out["logits"] if isinstance(out, dict) else out
        next_token_logits = logits[:, -1, :]
        
        # Applica repetition penalty
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[1]):
                token = input_ids[0, i].item()
                if next_token_logits[0, token] < 0:
                    next_token_logits[0, token] *= repetition_penalty
                else:
                    next_token_logits[0, token] /= repetition_penalty
                    
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == eos_token_id:
            break
            
    return input_ids

def main():
    print("="*60)
    print("  VERIFICA LEARNING — Modello 8 Layer (Clean Syntax)")
    print("="*60)

    # 1. Trova l'ultimo modello fine-tunato
    model_path = SCRIPT_DIR.parent / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"
    if not model_path.exists():
        print(f"❌ Modello non trovato in: {model_path}")
        print("Assicurati di aver scaricato la cartella runs/ dal cloud o modifica il path.")
        return

    tokenizer_path = SCRIPT_DIR / "tokenizer" / "vocab.model"
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Caricamento
    print(f"\n📦 Caricamento Tokenizer...")
    tok = load_tokenizer(str(tokenizer_path), vocab_size=1024)

    print(f"📦 Caricamento Modello a 8 Layer...")
    model = StudentModelProgressive(
        vocab_size=1024,
        dim_hidden=384,
        num_layers=8,
        simplicial_layers=[3, 7]
    )
    
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 3. Test Prompts (nel formato pulito!)
    # Niente XML, niente bracket numerici, sintassi esatta
    prompts = [
        # Test 1: Triangolo con punti medi (Teorema: segmento dei punti medi è parallelo alla base)
        "a : ; b : ; c : ; d : coll a b d cong a d b d ; e : coll a c e cong a e c e ? para d e b c",
        
        # Test 2: Angoli alla base di triangolo isoscele
        "a : ; b : ; c : cong a b a c ; d : coll b c d ? eqangle a b c a c b"
    ]

    print("\n" + "—"*60)
    for i, p in enumerate(prompts):
        print(f"\n🎯 TEST {i+1}:")
        print(f"PROMPT: {p}")
        
        inputs = tok(p, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        print("\n⏳ Generazione in corso...")
        out_ids = generate_text(
            model=model, 
            input_ids=input_ids, 
            max_new_tokens=50, 
            eos_token_id=tok.eos_token_id,
            repetition_penalty=1.5
        )
            
        generated_text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
        
        print("\n✨ OUTPUT GENERATO:")
        print(generated_text)
        print("—"*60)


if __name__ == "__main__":
    main()
