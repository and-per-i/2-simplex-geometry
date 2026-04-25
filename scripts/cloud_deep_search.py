import os
import sys
import torch
import time
import re
import multiprocessing as mp
from pathlib import Path

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
    aug_jgex, timeout, task_id = task
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
        return (success, task_id, aug_jgex)
    except Exception:
        signal.alarm(0)
        return (False, task_id, aug_jgex)

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
def get_model_suggestions(model, tok, prompt, device, k=64, temp=0.9):
    eos_id = tok.eos_token_id
    suggestions = []
    for attempt in range(k):
        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        for _ in range(30):
            out = model(input_ids)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_id: break
                
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
        "midp": "midpoint", "midpoint": "midpoint", "coll": "on_line",
        "perp": "on_tline", "para": "on_pline", "cong": "eqdistance", "circle": "on_circum"
    }
    if pred not in mapping: return ""
    n_pred = mapping[pred]
    
    if n_pred == "midpoint": final_args = args[:3]
    elif n_pred == "on_line": final_args = args[:3]
    elif n_pred in ["on_tline", "on_pline", "eqdistance", "on_circum"]: final_args = args[:4]
    else: return ""
    
    if len(final_args) < 3: return ""
    return f"; {target} = {n_pred} {' '.join(final_args)}"

def append_suggestion_to_prompt(prompt, sugg):
    parts = [p for p in sugg.split() if not re.match(r'^[ra]\d+$', p) and not p.isdigit() and p != 'x00']
    if len(parts) < 3: return prompt
    pred = parts[0]
    target = parts[1]
    args = [p for p in parts[2:] if len(p) <= 2]
    mapped_pred = PREDICATE_MAP.get(pred, pred)
    new_clause = f"{target} : {mapped_pred} {' '.join(args)}"
    
    setup, goal = prompt.split("?")
    new_prompt = f"{setup.strip()} ; {new_clause} ? {goal.strip()}"
    return new_prompt

def load_model(device):
    model_path = ROOT_DIR / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"
    tok_path = ROOT_DIR / "modello_distillato" / "tokenizer" / "vocab.model"
    tok = load_tokenizer(str(tok_path), vocab_size=1024)
    model = StudentModelProgressive(vocab_size=1024, dim_hidden=384, num_layers=8, simplicial_layers=[3, 7])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    if "cuda" in str(device) or "mps" in str(device): model.half()
    else: model.float()
    model.eval()
    return model, tok

def run_depth_search_for_problem(fname, model, tok, device, max_depth=5):
    trans_dir = ROOT_DIR / "imo_translated"
    raw_dir = ROOT_DIR / "imo_ag_30"
    
    with open(trans_dir / fname, "r") as f:
        jgex_str = f.read().strip()
        
    points, assumes, proves = parse_ag_file(raw_dir / fname)
    initial_prompt = ag_to_model_prompt(points, assumes, proves)
    
    print(f"\n📝 Analisi: {fname} (Max Depth={max_depth})")
    
    # Nodi = [(sugg_chain_string, aug_jgex, prompt)]
    current_nodes = [("", jgex_str, initial_prompt)]
    
    for depth in range(1, max_depth + 1):
        print(f"   [Depth {depth}] Generazione espansioni da {len(current_nodes)} nodi...")
        next_candidates = []
        
        for sugg_chain, jgex, prompt in current_nodes:
            # Piu k al depth 1, meno k ai successivi per contenere l'esplosione combinatoria
            k_val = 64 if depth == 1 else 16
            suggs = get_model_suggestions(model, tok, prompt, device, k=k_val, temp=0.9)
            
            jgex_setup, jgex_goal = jgex.split("?")
            for sugg in suggs:
                newclid_sugg = translate_suggestion_to_newclid(sugg)
                if newclid_sugg:
                    aug_jgex = f"{jgex_setup.strip()} {newclid_sugg} ? {jgex_goal.strip()}"
                    new_prompt = append_suggestion_to_prompt(prompt, sugg)
                    new_chain = f"{sugg_chain} -> {sugg}" if sugg_chain else sugg
                    next_candidates.append((new_chain, aug_jgex, new_prompt))
                    
        if not next_candidates:
            print(f"   [Depth {depth}] Nessuna costruzione valida generata. Fine ramo.")
            return False
            
        print(f"   [Depth {depth}] Test di {len(next_candidates)} rami composti con DDARN...")
        tasks = [(item[1], 15, i) for i, item in enumerate(next_candidates)]
        num_cores = max(1, mp.cpu_count())
        
        with mp.Pool(num_cores) as pool:
            try:
                from tqdm import tqdm
                iterator = tqdm(pool.imap_unordered(solve_worker, tasks), total=len(tasks), desc=f"D{depth}", leave=False)
            except ImportError:
                iterator = pool.imap_unordered(solve_worker, tasks)
                
            depth_solved = False
            for success, task_id, _ in iterator:
                if success:
                    print(f"\n   ✅✅ [Depth {depth}] DIMOSTRATO!!! 🏆")
                    print(f"   Catena risolutiva: {next_candidates[task_id][0]}")
                    pool.terminate()
                    return True
                    
        print(f"   ❌ [Depth {depth}] Nessun successo.")
        if depth < max_depth:
            print("   Tengo i migliori 16 rami per espanderli al prossimo livello...")
            current_nodes = next_candidates[:16]

    print(f"   ❌ Fallito anche a Depth {max_depth}.")
    return False

def main():
    print("="*80)
    print("🚀 ESPERIMENTO FINALE: BEAM SEARCH (DEPTH=100) SU UN SINGOLO PROBLEMA")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tok = load_model(device)

    # Esploriamo fino al centro della terra con un solo problema che ha dimostrato di generare tanti rami validi
    problems = [
        "translated_imo_2004_p1.txt"
    ]
    
    solved_count = 0
    for prob in problems:
        success = run_depth_search_for_problem(prob, model, tok, device, max_depth=100)
        if success:
            solved_count += 1
            print("\n🎉 ABBIAMO UN VINCITORE! Il sistema ha sbloccato una IMO!")
            break # L'utente voleva risolverne ALMENO UNA. Se la risolviamo, festeggiamo e fermiamo.
            
    if solved_count == 0:
        print("\nNessun problema risolto in questa run limitata (Depth=2).")

if __name__ == "__main__":
    main()
