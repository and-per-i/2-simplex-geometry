
import torch
import re
from modello_distillato.student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer

# --- CONFIGURAZIONE ---
CHECKPOINT_PATH = "runs/finetune_clean/pytorch_model_finetuned.bin"
TOKENIZER_PATH = "tokenizer/weights/geometry.757.model"

def load_davide():
    # Ottimizzazione per Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡️ Utilizzo accelerazione Metal (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Utilizzo accelerazione NVIDIA (CUDA)")
    else:
        device = torch.device("cpu")
        print("🐌 Utilizzo CPU (Nessuna accelerazione hardware trovata)")
        
    model = StudentModelProgressive(vocab_size=1024, dim_hidden=384, num_layers=8, num_heads=8)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Importante: load_tokenizer vuole il percorso del file .model
    tok = load_tokenizer(TOKENIZER_PATH, vocab_size=1024)
    return model, tok, device

# --- TRADUZIONE "PENSIERO DAVIDE" ---
DAVIDE_VOCAB = {
    "x00": "[Aggiungi Punto Ausiliario]",
    "00": "[Relazione Appartenenza]",
    "01": "[Relazione Congruenza]",
    "02": "[Relazione Parallelismo]",
    "05": "[Relazione Perpendicolarità]",
    "i": "[Intersezione Strategica]",
    "r49": "[Uguaglianza Raggi/Rapporti]",
    "c": "[Centro/Circonferenza]",
    "e": "[Punto Esterno]",
    ":": "→ DEFINITO COME:",
    ";": "PROSSIMO STEP:",
}

def decode_thought(raw_text):
    tokens = raw_text.split()
    decoded = []
    for t in tokens:
        if t in DAVIDE_VOCAB:
            decoded.append(f"\033[94m{DAVIDE_VOCAB[t]}\033[0m")
        else:
            decoded.append(t)
    return " ".join(decoded)

@torch.inference_mode()
def get_suggestions(model, tok, prompt, device, k=5):
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Batch generation for speed
    input_ids = input_ids.repeat(k, 1)
    finished = torch.zeros(k, dtype=torch.bool, device=device)
    
    for _ in range(30):
        logits = model(input_ids)
        if isinstance(logits, dict): logits = logits['logits']
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits / 0.8, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        finished |= (next_tokens.squeeze(-1) == tok.eos_token_id)
        if finished.all(): break
        
    results = []
    for i in range(k):
        decoded = tok.decode(input_ids[i].tolist(), skip_special_tokens=True)
        # Removed destructive .replace("▁", " ")
        raw_sug = decoded[len(prompt):].strip().split(";")[0]
        results.append(raw_sug)
    return results

def print_showcase(title, prompt, model, tok, device):
    print(f"\n{'-'*80}")
    print(f"🌟 SCENARIO: {title}")
    print(f"📝 Prompt Iniziale: {prompt}")
    print(f"{'-'*80}")
    
    suggestions = get_suggestions(model, tok, prompt, device)
    for i, s in enumerate(suggestions):
        thought = decode_thought(s)
        print(f"   💡 Suggerimento {i+1}:")
        print(f"      Raw: {s}")
        print(f"      Pensiero Davide: {thought}")

def main():
    print("\n" + "="*80)
    print("🚀 AVVIO SHOWCASE: DAVIDE 8L - L'Intuizione Geometrica")
    print("="*80)
    model, tok, device = load_davide()
    
    # 1. Triangolo e Punto Medio
    print_showcase(
        "Proprietà delle Mediane",
        "a b c = triangle a b c ; m = midpoint a b ? cong c m m a",
        model, tok, device
    )
    
    # 2. Perpendicolarità e Altezze
    print_showcase(
        "Intersezione delle Altezze (Ortocentro)",
        "a b c = triangle a b c ; h = perp h a b c ; k = perp k b a c ? coll h k a",
        model, tok, device
    )
    
    # 3. Cerchi e Concliclicità
    print_showcase(
        "Punti su una Circonferenza",
        "a b c d = square a b c d ; o = circle o a b c ? on_circle d o",
        model, tok, device
    )

    print("\n" + "="*80)
    print("✅ Showcase completato. Davide ha generato suggerimenti basati sulla sua 'intuizione' dei 2-Simplessi.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
