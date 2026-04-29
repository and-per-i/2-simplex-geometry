
import torch
import re
import sys
import os

# Aggiungi la root del progetto al path per trovare i moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modello_distillato.student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer

# --- CONFIGURAZIONE ---
CHECKPOINT_PATH = "runs/finetune_clean/pytorch_model_finetuned.bin"
TOKENIZER_PATH = "tokenizer/weights/geometry.757.model"

def load_model():
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
    
    tok = load_tokenizer(TOKENIZER_PATH, vocab_size=1024)
    return model, tok, device

# --- TRADUZIONE VOCABOLARIO GEOMETRICO ---
GEO_VOCAB = {
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

# --- SPIEGAZIONI PROBLEMI ---
PROBLEMS_INFO = {
    "Proprietà delle Mediane": {
        "desc": "Dato un triangolo ABC e il punto medio M del lato AB, dimostra una proprietà legata alla mediana CM.",
        "goal": "Verificare se il modello suggerisce costruzioni utili per legare la mediana ai lati o ad altri punti medi."
    },
    "Intersezione delle Altezze (Ortocentro)": {
        "desc": "Dato un triangolo ABC e due sue altezze (da A su BC e da B su AC) che si intersecano in H.",
        "goal": "Il modello deve intuire che H è l'ortocentro e suggerire proprietà legate alla terza altezza (da C su AB)."
    },
    "Punti su una Circonferenza": {
        "desc": "Dato un quadrato ABCD e una circonferenza passante per A, B e C.",
        "goal": "Dimostrare che anche il punto D giace sulla stessa circonferenza (proprietà di conciclicità del quadrato)."
    }
}

def decode_thought(raw_text):
    tokens = raw_text.split()
    decoded = []
    for t in tokens:
        if t in GEO_VOCAB:
            decoded.append(f"\033[94m{GEO_VOCAB[t]}\033[0m")
        else:
            decoded.append(t)
    return " ".join(decoded)

def explain_suggestion_it(raw_text):
    """Fornisce una spiegazione discorsiva in italiano del suggerimento."""
    explanation = []
    if "x00" in raw_text:
        explanation.append("Propone di aggiungere un punto ausiliario per facilitare la dimostrazione.")
    if "02" in raw_text:
        explanation.append("Cerca di stabilire una relazione di parallelismo tra segmenti.")
    if "05" in raw_text:
        explanation.append("Suggerisce di sfruttare la perpendicolarità (es. altezze o tangenti).")
    if "01" in raw_text:
        explanation.append("Tenta di dimostrare la congruenza tra segmenti o angoli.")
    if "i" in raw_text:
        explanation.append("Individua un punto di intersezione critico per la costruzione.")
    
    if not explanation:
        return "Il modello sta esplorando relazioni di base tra i punti esistenti."
    return " ".join(explanation)

@torch.inference_mode()
def get_suggestions(model, tok, prompt, device, k=5):
    raw_ids = tok.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([raw_ids]).to(device)
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
        raw_sug = decoded[len(prompt):].strip().split(";")[0]
        results.append(raw_sug)
    return results

def print_showcase(title, prompt, model, tok, device):
    info = PROBLEMS_INFO.get(title, {"desc": "Problema geometrico.", "goal": ""})
    
    print(f"\n{'='*80}")
    print(f"🌟 SCENARIO: {title}")
    print(f"{'='*80}")
    print(f"📝 PROMPT INIZIALE: {prompt}")
    print(f"📖 COSA CHIEDE IL PROBLEMA: {info['desc']}")
    if info['goal']:
        print(f"🎯 OBIETTIVO MODELLO: {info['goal']}")
    print(f"{'-'*80}")
    
    suggestions = get_suggestions(model, tok, prompt, device)
    for i, s in enumerate(suggestions):
        if not s: continue
        thought = decode_thought(s)
        explanation = explain_suggestion_it(s)
        print(f"   💡 Suggerimento {i+1}:")
        print(f"      [Output Modello]: {s}")
        print(f"      [Logica Interna]: {thought}")
        print(f"      [Spiegazione IT]: {explanation}\n")

def main():
    print("\n" + "*"*80)
    print("🚀 AVVIO SHOWCASE: Intuizione Geometrica 2-Simplessi")
    print("*"*80)
    model, tok, device = load_model()
    
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

    print("\n" + "*"*80)
    print("✅ Showcase completato. Il modello ha generato suggerimenti basati sull'intuizione geometrica.")
    print("*"*80 + "\n")

if __name__ == "__main__":
    main()
