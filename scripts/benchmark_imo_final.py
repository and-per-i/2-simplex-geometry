import os
import sys
import torch
import time
import re
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src/symbolic"))
sys.path.insert(0, str(ROOT_DIR / "modello_distillato"))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from tokenizer.hf_tokenizer import load_tokenizer
from newclid.api import GeometricSolverBuilder, PythonDefault
from newclid.jgex.formulation import JGEXFormulation
from newclid.jgex.problem_builder import JGEXProblemBuilder
import signal

PREDICATE_MAP = {
    'coll': '00', 'cong': '01', 'perp': '02', 'para': '03',
    'midpoint': '04', 'eqangle': '05', 'eqratio': '06', 'sameclock': '07',
    'sameside': '08', 'simtri': '09', 'contri': '10', 'cyclic': '11', 'circle': '12',
    'midp': '04'
}

REVERSE_MAP = {v: k for k, v in PREDICATE_MAP.items()}

def parse_ag_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    points = []
    assumes = []
    proves = []
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split()
        if parts[0] == "point": points.append(parts[1])
        elif parts[0] == "assume": assumes.append(parts[1:])
        elif parts[0] == "prove": proves.append(parts[1:])
    return points, assumes, proves

def ag_to_model_prompt(points, assumes, proves):
    target_assumes = {p: [] for p in points}
    for assume in assumes:
        pred = assume[0]
        args = assume[1:]
        target = max(args, key=lambda p: points.index(p) if p in points else -1)
        mapped_pred = PREDICATE_MAP.get(pred, pred)
        target_assumes[target].append(f"{mapped_pred} {' '.join(args)}")
        
    clauses = []
    for p in points:
        if target_assumes[p]:
            clauses.append(f"{p} : {' '.join(target_assumes[p])}")
        else:
            clauses.append(f"{p} :")
            
    setup_str = " ; ".join(clauses)
    
    goal_str = ""
    if proves:
        pred = proves[0][0]
        args = proves[0][1:]
        mapped_pred = PREDICATE_MAP.get(pred, pred)
        goal_str = f" ? {mapped_pred} {' '.join(args)} x00 "
        
    return setup_str + goal_str

def solve_with_newclid(problem_str, timeout=10):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        initial_problem = JGEXFormulation.from_text(problem_str)
        builder = JGEXProblemBuilder().with_problem(initial_problem)
        setup = builder.build(max_attempts_to_satisfy_goals_numerically=200)
        api_def = PythonDefault(use_sympy_ar=False)
        solver = GeometricSolverBuilder(api_default=api_def).build(setup)
        success = solver.run()
        signal.alarm(0)
        return success
    except TimeoutError:
        return False
    except Exception as e:
        signal.alarm(0)
        return False

@torch.inference_mode()
def get_model_suggestions(model, tok, prompt, device, k=3):
    eos_id = tok.eos_token_id
    suggestions = []
    
    for attempt in range(k):
        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        for _ in range(30):
            out = model(input_ids)
            logits = out["logits"] if isinstance(out, dict) else out
            next_token_logits = logits[:, -1, :]
            
            # Simple temperature sampling
            temp = 0.8
            probs = torch.softmax(next_token_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_id:
                break
                
        decoded = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
        # Get just the generated part
        gen = decoded[len(prompt):].strip()
        
        # Replace tokens back to words
        for num, eng in REVERSE_MAP.items():
            gen = re.sub(r'\b' + num + r'\b', eng, gen)
            
        gen = gen.replace("▁", " ")
        # Take the first construction (e.g. "midp e a c")
        first_step = gen.split(";")[0].strip()
        if first_step:
            suggestions.append(first_step)
            
    return list(set(suggestions))

def translate_suggestion_to_newclid(suggestion):
    import re
    # Ripuliamo i token tipo r54, a01, 1101, x00
    parts = [p for p in suggestion.split() if not re.match(r'^[ra]\d+$', p) and not p.isdigit() and p != 'x00']
    if len(parts) < 3: return ""
    pred = parts[0]
    target = parts[1]
    
    # Prendi solo i punti (lettere) come argomenti
    args = [p for p in parts[1:] if len(p) <= 2]
    
    mapping = {
        "midp": "midpoint",
        "midpoint": "midpoint",
        "coll": "on_line",
        "perp": "on_tline",
        "para": "on_pline",
        "cong": "eqdistance",
        "circle": "on_circum"
    }
    
    if pred not in mapping:
        return "" # ignora non-costruzioni
        
    n_pred = mapping[pred]
    
    # Adjust args
    if n_pred == "midpoint": final_args = args[:3]
    elif n_pred == "on_line": final_args = args[:3]
    elif n_pred in ["on_tline", "on_pline", "eqdistance"]: final_args = args[:4]
    elif n_pred == "on_circum": final_args = args[:4]
    else: return ""
    
    if len(final_args) < 3: return "" # Scarta roba rotta
    
    return f"; {target} = {n_pred} {' '.join(final_args)}"

def optimize_for_m4(model, device):
    model.to(device)
    model.half()
    model.eval()
    return model

def main():
    print("="*80)
    print("🏆 IMO-AG-30 BENCHMARK: Newclid vs Davide 8L M4")
    print("="*80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = ROOT_DIR / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"
    # Unified tokenizer path
    tok_path = ROOT_DIR / "tokenizer" / "weights" / "geometry.757.model"

    print("📦 Inizializzazione acceleratori Metal (FP16)...")
    tok = load_tokenizer(str(tok_path), vocab_size=1024)
    config = StudentConfig(
        vocab_size=1024, 
        hidden_size=512, 
        num_hidden_layers=6, 
        use_simplex_attention=True,
        w1=8,
        w2=8
    )
    model = StudentForCausalLM(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = optimize_for_m4(model, device)

    trans_dir = ROOT_DIR / "imo_translated"
    raw_dir = ROOT_DIR / "imo_ag_30"
    
    files = [f for f in os.listdir(trans_dir) if f.endswith(".txt")]
    files.sort()
    
    results = {"pure": 0, "ai": 0, "fail": 0}
    
    for idx, fname in enumerate(files):
        print(f"\n📝 [{idx+1}/30] Problema: {fname.replace('translated_', '').replace('.txt', '').upper()}")
        
        with open(trans_dir / fname, "r") as f:
            jgex_str = f.read().strip()
            
        print("   🔹 Pure DDARN: ", end="", flush=True)
        # FASE 1: Niente AI
        if solve_with_newclid(jgex_str, timeout=10):
            print("✅ Risolto!")
            results["pure"] += 1
            continue
        print("❌ Fallito (Timeout/Impossibile)")
        
        # FASE 2: Modello AI
        print("   🔥 Davide 8L AI: Richiesta costruzioni ausiliarie...")
        points, assumes, proves = parse_ag_file(raw_dir / fname)
        llm_prompt = ag_to_model_prompt(points, assumes, proves)
        
        suggestions = get_model_suggestions(model, tok, llm_prompt, device, k=10)
        solved = False
        
        for sugg in suggestions:
            newclid_sugg = translate_suggestion_to_newclid(sugg)
            if not newclid_sugg: continue
            
            # Inject auxiliary construction before the goal
            setup, goal = jgex_str.split("?")
            aug_jgex = f"{setup.strip()} {newclid_sugg} ? {goal.strip()}"
            
            print(f"      [Test Aux]: {sugg} ", end="", flush=True)
            if solve_with_newclid(aug_jgex, timeout=10):
                print(" -> ✅ DIMOSTRATO!")
                solved = True
                break
            else:
                print(" -> ❌")
                
        if solved:
            results["ai"] += 1
        else:
            results["fail"] += 1
            
    print("\n" + "="*80)
    print("📊 RISULTATI FINALI IMO-AG-30")
    print("="*80)
    print(f"🧊 Pure Newclid (Senza AI): {results['pure']}/30")
    print(f"🔥 Davide 8L (Con AI):     {results['pure'] + results['ai']}/30")
    print("="*80)

if __name__ == "__main__":
    main()
