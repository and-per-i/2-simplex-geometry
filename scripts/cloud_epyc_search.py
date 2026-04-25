import os
import sys
import torch
import time
import re
import multiprocessing as mp
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "src/symbolic"))
sys.path.insert(0, str(ROOT_DIR / "modello_distillato"))

from student_progressive import StudentModelProgressive
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

def solve_worker(task):
    """
    Worker function to run in a separate process on the massive CPU cores.
    task = (aug_jgex, timeout, sugg_idx, total_sugg)
    """
    aug_jgex, timeout, sugg_idx, total_sugg = task
    
    def handler(signum, frame):
        raise TimeoutError()
        
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        initial_problem = JGEXFormulation.from_text(aug_jgex)
        builder = JGEXProblemBuilder().with_problem(initial_problem)
        setup = builder.build(max_attempts_to_satisfy_goals_numerically=200)
        api_def = PythonDefault(use_sympy_ar=False)
        solver = GeometricSolverBuilder(api_default=api_def).build(setup)
        success = solver.run()
        signal.alarm(0)
        return (success, sugg_idx, aug_jgex)
    except TimeoutError:
        return (False, sugg_idx, aug_jgex)
    except Exception:
        signal.alarm(0)
        return (False, sugg_idx, aug_jgex)

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

@torch.inference_mode()
def get_model_suggestions(model, tok, prompt, device, k=256, temp=0.9):
    eos_id = tok.eos_token_id
    suggestions = []
    
    for attempt in range(k):
        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        for _ in range(30):
            out = model(input_ids)
            logits = out["logits"] if isinstance(out, dict) else out
            next_token_logits = logits[:, -1, :]
            
            probs = torch.softmax(next_token_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_id:
                break
                
        decoded = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
        gen = decoded[len(prompt):].strip()
        
        for num, eng in REVERSE_MAP.items():
            gen = re.sub(r'\b' + num + r'\b', eng, gen)
            
        gen = gen.replace("▁", " ")
        first_step = gen.split(";")[0].strip()
        if first_step:
            suggestions.append(first_step)
            
    seen = set()
    unique_sugg = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_sugg.append(s)
            
    return unique_sugg

def translate_suggestion_to_newclid(suggestion):
    parts = [p for p in suggestion.split() if not re.match(r'^[ra]\d+$', p) and not p.isdigit() and p != 'x00']
    if len(parts) < 3: return ""
    pred = parts[0]
    target = parts[1]
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
    
    if pred not in mapping: return ""
    n_pred = mapping[pred]
    
    if n_pred == "midpoint": final_args = args[:3]
    elif n_pred == "on_line": final_args = args[:3]
    elif n_pred in ["on_tline", "on_pline", "eqdistance"]: final_args = args[:4]
    elif n_pred == "on_circum": final_args = args[:4]
    else: return ""
    
    if len(final_args) < 3: return ""
    return f"; {target} = {n_pred} {' '.join(final_args)}"

def load_model(device):
    model_path = ROOT_DIR / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"
    tok_path = ROOT_DIR / "modello_distillato" / "tokenizer" / "vocab.model"

    tok = load_tokenizer(str(tok_path), vocab_size=1024)
    model = StudentModelProgressive(vocab_size=1024, dim_hidden=384, num_layers=8, simplicial_layers=[3, 7])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    if "cuda" in str(device) or "mps" in str(device):
        model.half()
    else:
        model.float()
    model.eval()
    return model, tok

def main():
    print("="*80)
    print("☁️ CLOUD EPYC BEAM SEARCH: CPU Massively Parallel DDARN")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"📦 Utilizzo acceleratore AI: {device}")
    
    model, tok = load_model(device)

    # Scelta del file (può essere iterato su tutto il benchmark)
    fname = "translated_imo_2019_p2_easy.txt"
    trans_dir = ROOT_DIR / "imo_translated"
    raw_dir = ROOT_DIR / "imo_ag_30"
    
    with open(trans_dir / fname, "r") as f:
        jgex_str = f.read().strip()
        
    print("\n🔥 Generazione massiva costruzioni sull'acceleratore (k=256, Temp=0.9)...")
    points, assumes, proves = parse_ag_file(raw_dir / fname)
    llm_prompt = ag_to_model_prompt(points, assumes, proves)
    
    t0 = time.time()
    suggestions = get_model_suggestions(model, tok, llm_prompt, device, k=256, temp=0.9)
    print(f"   Trovate {len(suggestions)} costruzioni uniche (Time: {time.time()-t0:.1f}s)")
    
    # Preparazione Task per Multiprocessing
    setup, goal = jgex_str.split("?")
    tasks = []
    
    for i, sugg in enumerate(suggestions):
        newclid_sugg = translate_suggestion_to_newclid(sugg)
        if not newclid_sugg: continue
        aug_jgex = f"{setup.strip()} {newclid_sugg} ? {goal.strip()}"
        tasks.append((aug_jgex, 20, i+1, len(suggestions))) # timeout = 20s
        
    num_cores = min(mp.cpu_count(), len(tasks))
    print(f"\n🚀 Avvio Beam Search Parallelo con {num_cores} Core della CPU...")
    
    solved = False
    winning_sugg = ""
    
    # Esegui i task sui core della CPU in parallelo!
    t1 = time.time()
    with mp.Pool(processes=num_cores) as pool:
        try:
            from tqdm import tqdm
            iterator = tqdm(pool.imap_unordered(solve_worker, tasks), total=len(tasks), desc="🌳 Rami DDARN")
            use_tqdm = True
        except ImportError:
            iterator = pool.imap_unordered(solve_worker, tasks)
            use_tqdm = False
            
        for result in iterator:
            success, sugg_idx, aug_jgex = result
            if success:
                solved = True
                winning_sugg = aug_jgex
                print(f"\n   ✅ [Worker {sugg_idx}] DIMOSTRATO!!! Trovata soluzione valida!")
                pool.terminate() # Abbiamo vinto, ferma tutti gli altri!
                break
            else:
                if not use_tqdm:
                    print(f"   ❌ [Worker {sugg_idx}] Fallito           ", end="\r")

    if solved:
        print("\n\n" + "="*80)
        print("🎉 MIRACOLO COMPIUTO DAL CLOUD!")
        print(f"Costruzione Vincente: {winning_sugg}")
        print(f"Tempo totale ricerca: {time.time()-t1:.1f}s")
        print("="*80)
    else:
        print("\n\n" + "="*80)
        print("Nessun worker ha trovato la soluzione a profondità 1.")
        print(f"Tempo totale ricerca: {time.time()-t1:.1f}s")
        print("="*80)

if __name__ == "__main__":
    main()
