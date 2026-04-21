import sys
import os
print("DEBUG: Script avviato...", flush=True)
import torch
import sentencepiece as spm

# Configurazione Percorsi
NEWCLID_ROOT = "/Users/andrea/Documents/Newclid_Transformer"
NEWCLID_SRC = os.path.join(NEWCLID_ROOT, "src")
WEIGHTS_DIR = os.path.join(NEWCLID_ROOT, "pt_ckpt")

# Aggiungi Newclid al path di sistema per importare alphageo
if os.path.exists(NEWCLID_SRC):
    sys.path.append(NEWCLID_SRC)
else:
    print(f"ERRORE: Percorso sorgente Newclid non trovato in {NEWCLID_SRC}")

try:
    from alphageo.model import Decoder
    from alphageo.inference import priority_beam_search
except ImportError:
    print("ERRORE: Impossibile importare moduli alphageo. Assicurati che PYTHONPATH sia corretto.")
    sys.exit(1)

def load_newclid_model():
    print(f"\n--- Caricamento Modello Newclid (Teacher) da {WEIGHTS_DIR} ---")
    
    cfg_path = os.path.join(WEIGHTS_DIR, "cfg.sav")
    params_path = os.path.join(WEIGHTS_DIR, "params.sav")
    vocab_path = os.path.join(WEIGHTS_DIR, "vocab.model")
    
    if not all(os.path.exists(p) for p in [cfg_path, params_path, vocab_path]):
        print(f"ERRORE: Pesi non trovati in {WEIGHTS_DIR}. Controlla la cartella pt_ckpt.")
        sys.exit(1)

    cfg = torch.load(cfg_path, weights_only=False)
    model = Decoder(cfg)
    params = torch.load(params_path, weights_only=False)
    model.load_state_dict(params)
    model.eval()
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model.bfloat16()
    elif torch.backends.mps.is_available():
        device = "mps"
        # model.bfloat16() # MPS a volte ha problemi con bfloat16, usiamo float32 se fallisce
    
    model.to(device)
    tokenizer = spm.SentencePieceProcessor(vocab_path)
    return model, tokenizer, device

def run_newclid_inference(model, tokenizer, device, prompt_raw):
    # Pulizia del prompt (rimuove i tag XML)
    clean_prompt = prompt_raw.replace("<problem>", "").replace("</problem>", "").strip()
    
    # Mapping verso i token AlphaGeometry originali (grammar compatibile)
    mapping = {
        "midpoint": "M",
        "perp": "T",
        "para": "P",
        "altitude": "T",
        "on_circle": "O",
        "inter": "I",
        "eqangle": "^",
        "triangle": "triangle", # Alcuni modelli lo riconoscono
    }
    
    tokens = clean_prompt.split()
    ag_tokens = []
    for t in tokens:
        t_low = t.lower().strip(":;")
        if t_low in mapping:
            ag_tokens.append(mapping[t_low])
        elif t.startswith("[") and t.endswith("]"):
            # Converte [001] in 01 (formato AG)
            val = t[1:-1]
            if val.isdigit(): ag_tokens.append(f"{int(val):02d}")
            else: ag_tokens.append(t)
        else:
            ag_tokens.append(t)
    
    # Formato finale atteso dall'LLM: {S} ... {F1} x00
    full_prompt = "{S} " + " ".join(ag_tokens)
    if not full_prompt.endswith("x00"):
        if "{F1}" not in full_prompt:
            full_prompt += " {F1}"
        full_prompt += " x00"
    
    print(f"\nPROMPT ORIGINALE: {clean_prompt}")
    print(f"PROMPT TRADOTTO (AG): {full_prompt}")
    
    input_ids = tokenizer.encode(full_prompt)
    inp = torch.LongTensor([input_ids]).to(device)
    
    with torch.no_grad():
        # Eseguiamo una beam search per trovare le costruzioni migliori
        outs = priority_beam_search(model, inp, beam_width=4, num_return_sequences=2)
    
    print("\nSUGGERIMENTI NEWCLID (Top 2):")
    if not outs:
        print("Nessuna costruzione generata.")
    for i, (seq, score) in enumerate(outs):
        # Decodifica solo la parte generata (nuovi token dopo l'input)
        decoded = tokenizer.decode(seq[len(input_ids):]).strip()
        print(f"{i+1}. {decoded} (score: {score:.4f})")

if __name__ == "__main__":
    model, tokenizer, device = load_newclid_model()
    
    # Esempi presi da inference_test.py
    theorems = [
        "<problem> a : ; b : ; c : ; d : midpoint a b d [000] ; e : midpoint a c e [001] ? para d e b c </problem> ",
        "<problem> a : ; b : ; c : ; d : on_circle d a b c [000] ? eqangle a c b a d b </problem> ",
        "<problem> a : ; b : ; c : ; h1 : altitude a b c h1 [000] ; h2 : altitude b a c h2 [001] ; h : inter h1 h2 h [002] ? perp c h a b </problem> "
    ]
    
    for i, t in enumerate(theorems):
        print("\n" + "="*70)
        print(f"TEST {i+1}")
        run_newclid_inference(model, tokenizer, device, t)
    print("\n" + "="*70 + "\n")
