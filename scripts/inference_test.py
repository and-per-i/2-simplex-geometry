import torch
import sys
import os
import re

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.translator.geometry_translator import GeometryTranslator
from tokenizer.hf_tokenizer import load_tokenizer

def run_pro_inference(prompt, model_path, tokenizer_path):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tokenizer_path)
    translator = GeometryTranslator()
    
    print(f"\n--- Loading Model on {device} ---")
    config = StudentConfig.from_pretrained(model_path)
    config.use_triton = False  # Set to True if kernel is available and on GPU
    
    model = StudentForCausalLM.from_pretrained(
        model_path, 
        config=config,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()

    print(f"\nPROMPT: {prompt}")
    inputs = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt").items()}

    print("Generating proof (Beam Search, beams=5)...")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=512,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    solution_raw = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Simple extraction of steps for the translator
    # (Note: This is a heuristic based on the AlphaGeometry token format)
    # The translator expects a list of dicts like {'premises': [], 'rule': '', 'conclusion': ''}
    # For now, let's just print the raw solution and the translator output for the whole thing if possible
    # Actually, the model outputs tokens like 'perp a b c d ; para e f g h'
    
    print("\n================ RAW OUTPUT ================")
    print(solution_raw)
    
    print("\n================ TRANSLATED PROOF (Master) ================")
    translated_proof = translator.translate_proof(prompt, solution_raw)
    print(translated_proof)
    print("==================================================\n")

if __name__ == "__main__":
    # Automatic detection of the latest fine-tuned checkpoint
    base_finetuned_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints_finetuned"))
    MODEL_DIR = os.path.join(base_finetuned_dir, "checkpoint-final")
    
    if not os.path.exists(MODEL_DIR):
        # Look for the folder with the highest number if checkpoint-final is not found
        subdirs = [d for d in os.listdir(base_finetuned_dir) if d.startswith("checkpoint-")]
        if subdirs:
            latest = sorted(subdirs, key=lambda x: int(x.split("-")[1]))[-1]
            MODEL_DIR = os.path.join(base_finetuned_dir, latest)
            print(f"🔍 Auto-detected latest checkpoint: {latest}")
    
    TOKENIZER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/weights/geometry.757.model"))

    theorems = [
        "<problem> a : ; b : ; c : ; d : midpoint a b d [000] ; e : midpoint a c e [001] ? para d e b c </problem> ",
        "<problem> a : ; b : ; c : ; d : on_circle d a b c [000] ? eqangle a c b a d b </problem> ",
        "<problem> a : ; b : ; c : ; h1 : altitude a b c h1 [000] ; h2 : altitude b a c h2 [001] ; h : inter h1 h2 h [002] ? perp c h a b </problem> "
    ]

    print("Scegli un test:")
    for i, t in enumerate(theorems): print(f"{i+1}. {t}")
    
    # Non-interactive mode if running in script
    prompt = theorems[0]
    
    if os.isatty(sys.stdin.fileno()):
        choice = input("\nInserisci il numero o scrivi un nuovo prompt (invio per default): ")
        if choice.isdigit() and 1 <= int(choice) <= len(theorems):
            prompt = theorems[int(choice)-1]
        elif choice:
            prompt = choice

    run_pro_inference(prompt, MODEL_DIR, TOKENIZER_FILE)
