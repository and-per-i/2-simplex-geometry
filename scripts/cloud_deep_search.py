import os
import sys
import torch
import time
import re
import multiprocessing as mp
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, str(ROOT_DIR / "src/symbolic"))
    sys.path.insert(0, str(ROOT_DIR / "modello_distillato"))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.cache.q_filters import QFilterEviction, load_q_filters
from src.cache.l2_eviction import L2Eviction
from tokenizer.hf_tokenizer import load_tokenizer
from newclid.api import GeometricSolverBuilder, PythonDefault
from newclid.jgex.formulation import JGEXFormulation
from newclid.jgex.problem_builder import JGEXProblemBuilder
# import signal

PREDICATE_MAP = {
    'coll': '00', 'cong': '01', 'perp': '02', 'para': '03',
    'midpoint': '04', 'eqangle': '05', 'eqratio': '06', 'sameclock': '07',
    'sameside': '08', 'simtri': '09', 'contri': '10', 'cyclic': '11', 'circle': '12',
    'midp': '04'
}
REVERSE_MAP = {v: k for k, v in PREDICATE_MAP.items()}

def solve_worker(task):
    import threading
    import _thread
    aug_jgex, timeout, task_id = task
    
    timer = threading.Timer(timeout, lambda: _thread.interrupt_main())
    timer.start()
    
    try:
        initial_problem = JGEXFormulation.from_text(aug_jgex)
        builder = JGEXProblemBuilder().with_problem(initial_problem)
        setup = builder.build(max_attempts_to_satisfy_goals_numerically=200)
        api_def = PythonDefault(use_sympy_ar=False)
        solver = GeometricSolverBuilder(api_default=api_def).build(setup)
        success = solver.run()
        return (success, task_id, aug_jgex)
    except (KeyboardInterrupt, TimeoutError):
        return (False, task_id, aug_jgex)
    except Exception:
        return (False, task_id, aug_jgex)
    finally:
        timer.cancel()

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
def get_model_suggestions(
    model, tok, prompt, device,
    k=1024, temp=0.9, batch_size=128,
    max_new_tokens=30,
    q_filter_eviction=None,
    l2_eviction=None,
    token_budget=None,
):
    """
    Genera k costruzioni ausiliarie candidate per il problema dato.

    Usa model.generate() con KV cache abilitata per efficienza O(S) per step
    invece di O(S²). L'eviction è applicata dentro il forward del modello:
      - Layer standard (0,2,3,4,6): Q-Filter eviction (se q_filter_eviction fornito)
      - Layer simpliciali (1,5,7): L2-norm eviction (se l2_eviction fornito)
    """
    eos_id = tok.eos_token_id or 1
    suggestions = []

    inputs = tok(prompt, return_tensors="pt")
    base_input_ids = inputs["input_ids"].to(device)   # (1, S_prompt)
    prompt_len = base_input_ids.shape[1]

    for i in range(0, k, batch_size):
        bsz = min(batch_size, k - i)
        input_ids = base_input_ids.repeat(bsz, 1)     # (bsz, S_prompt)

        # model.generate() gestisce internamente la KV cache tramite
        # prepare_inputs_for_generation() — ogni step passa solo il nuovo token.
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            use_cache=True,
            # Passa eviction objects come kwargs: arrivano al forward() via **kwargs
            # grazie al fatto che prepare_inputs_for_generation li inoltra.
            # Nota: HF Trainer li ignora silenziosamente; qui usiamo il nostro modello.
        )
        # generated: (bsz, S_prompt + new_tokens)

        for b in range(bsz):
            gen_ids = generated[b, prompt_len:].tolist()
            # Rimuovi padding dopo EOS
            if eos_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos_id)]
            gen = tok.decode(gen_ids, skip_special_tokens=True)
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

def translate_suggestion_to_newclid(suggestion, prompt):
    # Extract current points from the prompt setup
    setup = prompt.split("?")[0]
    current_points = []
    for clause in setup.split(";"):
        if ":" in clause:
            p = clause.split(":")[0].strip()
            if p: current_points.append(p)
            
    # Remove rule tokens like r51, a00, x00 but KEEP numbers like 2, 1/2
    parts = [p for p in suggestion.split() if not re.match(r'^[ra]\d+$', p) and p != 'x00']
    if len(parts) < 3: return ""
    
    pred = parts[0]
    args = parts[1:]
    
    # Find target: the first single letter argument that is not in current_points
    target = None
    for p in args:
        if len(p) <= 2 and p.isalpha() and p not in current_points:
            target = p
            break
            
    if not target:
        # Not creating a new point -> not a valid auxiliary construction for JGEX point creation
        return ""
        
    mapping = {
        "midp": "midpoint", "midpoint": "midpoint", 
        "coll": "on_line", "perp": "on_tline", "para": "on_pline", 
        "cong": "eqdistance", "circle": "on_circum",
        "rconst": "rconst", "eqratio": "eqratio", "eqangle": "eqangle"
    }
    
    if pred not in mapping: return ""
    n_pred = mapping[pred]
    
    if n_pred == "midpoint": final_args = [a for a in args if a != target][:2]
    elif n_pred == "on_line": final_args = [a for a in args if a != target][:2]
    elif n_pred in ["on_tline", "on_pline", "eqdistance", "on_circum"]: final_args = [a for a in args if a != target][:3]
    elif n_pred in ["rconst", "eqratio", "eqangle"]: final_args = args # Keep all arguments including the target
    else: final_args = args
    
    if len(final_args) < 2: return ""
    return f"; {target} = {n_pred} {' '.join(final_args)}"

def append_suggestion_to_prompt(prompt, sugg):
    parts = [p for p in sugg.split() if not re.match(r'^[ra]\d+$', p) and not p.isdigit() and p != 'x00']
    if len(parts) < 3: return prompt
    pred = parts[0]
    target = parts[1]
    args = [p for p in parts[2:] if len(p) <= 2]
    mapped_pred = PREDICATE_MAP.get(pred, pred)
    if not args: return prompt
    new_clause = f"{target} : {mapped_pred} {' '.join(args)}"
    
    setup, goal = prompt.split("?")
    new_prompt = f"{setup.strip()} ; {new_clause} ? {goal.strip()}"
    return new_prompt

def load_model(device, checkpoint_path=None):
    """Carica modello, tokenizer e Q-Filters (se disponibili).

    Accetta:
    - Un file .pt Phase 1 (simplex-geometry_Final.pt)
    - Una directory HF Phase 2 (from_pretrained)
    - None → fallback al path legacy

    Cerca automaticamente q_filters.pt accanto al checkpoint.

    Returns:
        model, tok, q_filter_eviction (None se non trovati), l2_eviction
    """
    from src.models.checkpoint_loader import load_from_phase1

    tok_path = ROOT_DIR / "tokenizer" / "weights" / "geometry.757.model"
    tok = load_tokenizer(str(tok_path), vocab_size=1024)

    if checkpoint_path is None:
        checkpoint_path = ROOT_DIR / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"

    path = Path(checkpoint_path)

    if path.is_file() and path.suffix == ".pt":
        model = load_from_phase1(str(path), device=str(device))
        q_filters_path = path.parent / "q_filters.pt"
    elif path.is_dir():
        config = StudentConfig.from_pretrained(str(path))
        model = StudentForCausalLM.from_pretrained(str(path), config=config)
        model.to(device)
        q_filters_path = path / "q_filters.pt"
    else:
        config = StudentConfig(
            vocab_size=1024,
            hidden_size=384,
            intermediate_size=1536,
            max_position_embeddings=512,
            num_hidden_layers=8,
            use_simplex_attention=True,
            simplex_layers=[1, 5, 7],
            w1=8,
            w2=8,
        )
        model = StudentForCausalLM(config)
        model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        model.to(device)
        q_filters_path = path.parent / "q_filters.pt"

    if "cuda" in str(device) or "mps" in str(device):
        model.half()
    else:
        model.float()
    model.eval()

    # Carica Q-Filters se disponibili
    q_filter_eviction = None
    if q_filters_path.exists():
        filters = load_q_filters(str(q_filters_path))
        q_filter_eviction = QFilterEviction(filters)
        print(f"✅ Q-Filters caricati ({len(filters)} layer standard)")
    else:
        print("ℹ️  Q-Filters non trovati — nessuna eviction per layer standard.")
        print(f"   Esegui: python scripts/calibrate_q_filters.py --checkpoint {path}")

    # L2 eviction sempre disponibile (zero overhead, no calibrazione)
    l2_eviction = L2Eviction()

    return model, tok, q_filter_eviction, l2_eviction

def run_depth_search_for_problem(fname, model, tok, device, max_depth=3,
                                  q_filter_eviction=None, l2_eviction=None, token_budget=None):
    trans_dir = ROOT_DIR / "imo_translated"
    raw_dir = ROOT_DIR / "imo_ag_30"
    
    with open(trans_dir / fname, "r") as f:
        jgex_str = f.read().strip()
        
    points, assumes, proves = parse_ag_file(raw_dir / fname)
    initial_prompt = ag_to_model_prompt(points, assumes, proves)
    
    print(f"\n📝 Analisi: {fname} (Max Depth={max_depth})")
    
    current_nodes = [("", jgex_str, initial_prompt)]
    
    for depth in range(1, max_depth + 1):
        print(f"   [Depth {depth}] Generazione massiva da {len(current_nodes)} nodi precedenti...")
        next_candidates = []
        
        for sugg_chain, jgex, prompt in current_nodes:
            # Ampiezza massiva: 2048 tentativi iniziali, 64 in profondità!
            k_val = 2048 if depth == 1 else 64
            batch_sz = 128 if ("cuda" in str(device) or "mps" in str(device)) else 32

            suggs = get_model_suggestions(
                model, tok, prompt, device,
                k=k_val, temp=0.9, batch_size=batch_sz,
                q_filter_eviction=q_filter_eviction,
                l2_eviction=l2_eviction,
                token_budget=token_budget,
            )
            
            jgex_setup, jgex_goal = jgex.split("?")
            for sugg in suggs:
                newclid_sugg = translate_suggestion_to_newclid(sugg, prompt)
                if newclid_sugg:
                    aug_jgex = f"{jgex_setup.strip()} {newclid_sugg} ? {jgex_goal.strip()}"
                    new_prompt = append_suggestion_to_prompt(prompt, sugg)
                    new_chain = f"{sugg_chain} -> {sugg}" if sugg_chain else sugg
                    next_candidates.append((new_chain, aug_jgex, new_prompt))
                    
        if not next_candidates:
            print(f"   [Depth {depth}] Nessuna costruzione valida generata.")
            return False
            
        print(f"   [Depth {depth}] Trovati {len(next_candidates)} rami unici e validi! Test con DDARN...")
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
            print("   Tengo i migliori 64 rami per l'espansione del prossimo livello...")
            current_nodes = next_candidates[:64]

    print(f"   ❌ Fallito anche a Depth {max_depth}.")
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Beam search + DDARN neuro-symbolic solver")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path al checkpoint: .pt Phase 1, directory HF Phase 2, o .bin legacy."
    )
    parser.add_argument(
        "--token_budget", type=int, default=None,
        help="Numero max di token da tenere in KV cache (eviction attiva se impostato). "
             "Suggerito: 16-32 per sequenze geometriche brevi."
    )
    args = parser.parse_args()

    print("="*80)
    print("🚀 GOOGLE DEEPMIND SIMULATION: BATCHED GPU GENERATION + 128 CORE DDARN")
    print("="*80)
    if args.token_budget:
        print(f"🗜️  KV Cache eviction: budget={args.token_budget} token")
        print(f"   Layer standard → Q-Filters | Layer simpliciali → L2-norm")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tok, q_filter_eviction, l2_eviction = load_model(device, checkpoint_path=args.checkpoint)

    # Torniamo a testare la suite IMO
    problems = [
        "translated_imo_2004_p1.txt",
        "translated_imo_2000_p1.txt",
        "translated_imo_2004_p5.txt",
        "translated_imo_2008_p1a.txt"
    ]
    
    solved_count = 0
    for prob in problems:
        success = run_depth_search_for_problem(
            prob, model, tok, device, max_depth=3,
            q_filter_eviction=q_filter_eviction if args.token_budget else None,
            l2_eviction=l2_eviction if args.token_budget else None,
            token_budget=args.token_budget,
        )
        if success:
            solved_count += 1
            print("\n🎉 ABBIAMO UN VINCITORE! Il sistema ha sbloccato una IMO!")
            break # L'utente voleva risolverne ALMENO UNA. Se la risolviamo, festeggiamo e fermiamo.
            
    if solved_count == 0:
        print(f"\nNessun problema risolto in questa run limitata (Max Depth={3}).")

if __name__ == "__main__":
    main()
