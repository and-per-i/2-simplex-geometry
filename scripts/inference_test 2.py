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

    print("Generating proof (Greedy Search)...")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
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
    # Using the new GeometryTranslator logic
    # We need to split the solution into steps. AlphaGeometry steps usually end with ';'
    raw_steps = solution_raw.split(';')
    proof_steps = []
    
    for i, step_str in enumerate(raw_steps):
        step_str = step_str.strip()
        if not step_str: continue
        
        # Heuristic: find the rule. In AlphaGeometry, it's often the last token before ':' or part of the premises
        # Actually, let's use a simpler approach: translate each predicate found
        predicates = re.findall(r'[a-z0-9]+\s+[a-z\s]+', step_str) # This is too simple
        
        # Let's just use the translator's translate_predicate on each part
        translated_step = translator.translate_predicate(step_str)
        print(f"STEP {i+1:02d}: {translated_step}")

    print("==================================================\n")

if __name__ == "__main__":
    # Update these paths to match your project structure
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints/checkpoint-56000"))
    TOKENIZER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/weights/geometry.757.model"))

    theorems = [
        "a b c = triangle a b c; d = midpoint a b; e = midpoint a c; ? parallel d e b c",
        "a b c d = circle a b c d; ? cong angle a c b angle a d b",
        "a b c = triangle a b c; h1 = altitude a b c; h2 = altitude b a c; h = intersection h1 h2; ? perpendicular c h a b"
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
