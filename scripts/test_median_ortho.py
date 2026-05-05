"""
test_median_ortho.py — Test del sistema neuro-simbolico su due problemi canonici.

Problema 1 (Mediana):
    In un triangolo ABC con M = punto medio di AB, dimostra CM = MA.
    (Richiede al modello di suggerire che C giaccia sulla circonferenza con diametro AB.)

Problema 2 (Ortocentro):
    Dato il triangolo ABC con le altezze da A e da B che si intersecano in O,
    dimostra che O giace anche sull'altezza da C.
    (Richiede la costruzione ausiliaria che chiude la concorrenza delle altezze.)

Uso:
    python scripts/test_median_ortho.py --checkpoint runs/imo_unified/final_imo_model
"""

import sys
import re
import torch
import multiprocessing as mp
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src/symbolic"))

from src.models.checkpoint_loader import load_checkpoint
from tokenizer.hf_tokenizer import load_tokenizer
from newclid.api import GeometricSolverBuilder, PythonDefault
from newclid.jgex.formulation import JGEXFormulation
from newclid.jgex.problem_builder import JGEXProblemBuilder

PROBLEMS = [
    {
        "name": "Mediana (teorema del triangolo rettangolo)",
        "jgex": "a b c = triangle a b c; m = midpoint m a b ? cong c m m a",
        "prompt": "a : ; b : ; c : ; m : 04 a b ? 01 c m m a",
        "difficulty": "base",
    },
    {
        "name": "Ortocentro (concorrenza delle altezze)",
        "jgex": "a b c = triangle a b c; d = on_tline d a b c; e = on_tline e b a c; o = on_line o a d, on_line o b e ? on_tline o c a b",
        "prompt": "a : ; b : ; c : ; d : 02 d a b c ; e : 02 e b a c ; o : 00 o a d 00 o b e ? 02 o c a b",
        "difficulty": "medio",
    },
]

PREDICATE_MAP = {
    'coll': '00', 'cong': '01', 'perp': '02', 'para': '03',
    'midpoint': '04', 'eqangle': '05', 'eqratio': '06', 'sameclock': '07',
    'sameside': '08', 'simtri': '09', 'contri': '10', 'cyclic': '11', 'circle': '12',
    'midp': '04',
}
REVERSE_MAP = {v: k for k, v in PREDICATE_MAP.items()}


def solve_worker(task):
    import threading, _thread
    jgex_str, timeout, task_id = task
    timer = threading.Timer(timeout, lambda: _thread.interrupt_main())
    timer.start()
    try:
        problem = JGEXFormulation.from_text(jgex_str)
        setup = JGEXProblemBuilder().with_problem(problem).build(
            max_attempts_to_satisfy_goals_numerically=200
        )
        solver = GeometricSolverBuilder(
            api_default=PythonDefault(use_sympy_ar=False)
        ).build(setup)
        return (solver.run(), task_id, jgex_str)
    except (KeyboardInterrupt, TimeoutError):
        return (False, task_id, jgex_str)
    except Exception:
        return (False, task_id, jgex_str)
    finally:
        timer.cancel()


@torch.inference_mode()
def get_suggestions(model, tok, prompt, device, k=512, temp=0.9, batch_size=64):
    raw_ids = tok.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([raw_ids], device=device).repeat(min(k, batch_size), 1)
    eos_id = tok.eos_token_id
    suggestions = []

    num_batches = (k + batch_size - 1) // batch_size
    for b in range(num_batches):
        bs = min(batch_size, k - b * batch_size)
        ids = torch.tensor([raw_ids], device=device).repeat(bs, 1)
        finished = torch.zeros(bs, dtype=torch.bool, device=device)

        for _ in range(32):
            out = model(ids)
            logits = out.logits if hasattr(out, "logits") else out
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_tok], dim=-1)
            finished |= (next_tok.squeeze(-1) == eos_id)
            if finished.all():
                break

        for i in range(bs):
            decoded = tok.decode(ids[i].tolist(), skip_special_tokens=True)
            gen = decoded[len(prompt):].strip()
            for num, eng in REVERSE_MAP.items():
                gen = re.sub(r'\b' + num + r'\b', eng, gen)
            first = gen.split(";")[0].strip()
            if first:
                suggestions.append(first)

    return list(set(suggestions))


def translate_to_newclid(suggestion, prompt):
    parts = [p for p in suggestion.split()
             if not re.match(r'^[ra]\d+$', p) and p != 'x00' and not p.isdigit()]
    if len(parts) < 3:
        return ""
    pred, target = parts[0], parts[1]
    args = [p for p in parts[1:] if len(p) <= 2]

    mapping = {
        "midpoint": "midpoint", "midp": "midpoint",
        "coll": "on_line",
        "perp": "on_tline",
        "para": "on_pline",
        "cong": "eqdistance",
        "circle": "on_circum",
        "cyclic": "on_circum",
    }
    if pred not in mapping:
        return ""

    n_pred = mapping[pred]
    if n_pred == "midpoint":       final_args = args[:3]
    elif n_pred == "on_line":      final_args = args[:3]
    elif n_pred in ("on_tline", "on_pline", "eqdistance"): final_args = args[:4]
    elif n_pred == "on_circum":    final_args = args[:4]
    else:
        return ""

    if len(final_args) < 3:
        return ""
    return f"; {target} = {n_pred} {' '.join(final_args)}"


def test_problem(problem, model, tok, device, k=512, max_depth=2):
    name = problem["name"]
    jgex = problem["jgex"]
    prompt = problem["prompt"]

    print(f"\n{'='*70}")
    print(f"Problema: {name}  [{problem['difficulty']}]")
    print(f"JGEX: {jgex}")
    print(f"{'='*70}")

    # Fase 1: Newclid puro (senza AI)
    print("Fase 1 — Newclid puro: ", end="", flush=True)
    ok, _, _ = solve_worker((jgex, 15, 0))
    if ok:
        print("RISOLTO (senza AI)")
        return "pure"
    print("fallito")

    # Fase 2: modello + Newclid (beam search)
    jgex_setup, jgex_goal = jgex.split("?")
    current_nodes = [(jgex, prompt)]

    for depth in range(1, max_depth + 1):
        print(f"Fase 2 depth={depth} — Generazione {k if depth == 1 else 64} suggerimenti...")
        k_this = k if depth == 1 else 64
        candidates = []

        for cur_jgex, cur_prompt in current_nodes:
            suggs = get_suggestions(model, tok, cur_prompt, device, k=k_this)
            setup_part, goal_part = cur_jgex.split("?")
            for sugg in suggs:
                nc = translate_to_newclid(sugg, cur_prompt)
                if nc:
                    aug = f"{setup_part.strip()} {nc} ? {goal_part.strip()}"
                    new_prompt = cur_prompt.rstrip() + f" {sugg} ;"
                    candidates.append((aug, new_prompt, sugg))

        if not candidates:
            print(f"  Nessuna costruzione valida generata a depth {depth}.")
            continue

        print(f"  {len(candidates)} rami da testare con Newclid...")
        tasks = [(c[0], 15, i) for i, c in enumerate(candidates)]
        with mp.Pool(max(1, mp.cpu_count())) as pool:
            try:
                from tqdm import tqdm
                it = tqdm(pool.imap_unordered(solve_worker, tasks),
                          total=len(tasks), desc=f"D{depth}")
            except ImportError:
                it = pool.imap_unordered(solve_worker, tasks)
            for success, tid, _ in it:
                if success:
                    print(f"\n  DIMOSTRATO a depth {depth}!")
                    print(f"  Costruzione: {candidates[tid][2]}")
                    pool.terminate()
                    return "ai"

        print(f"  Nessun successo a depth {depth}.")
        current_nodes = [(c[0], c[1]) for c in candidates[:64]]

    return "fail"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint fine-tunato (directory HF o .pt)")
    parser.add_argument("--k", type=int, default=512,
                        help="Suggerimenti al primo livello (default 512)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Profondità beam search (default 2)")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tok_path = ROOT_DIR / "tokenizer" / "weights" / "geometry.757.model"
    tok = load_tokenizer(str(tok_path), vocab_size=1024)

    print(f"Caricamento checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    results = {}
    for prob in PROBLEMS:
        results[prob["name"]] = test_problem(prob, model, tok, device,
                                              k=args.k, max_depth=args.depth)

    print(f"\n{'='*70}")
    print("RIEPILOGO")
    print(f"{'='*70}")
    for name, res in results.items():
        emoji = "pure" if res == "pure" else ("AI" if res == "ai" else "FAIL")
        print(f"  {name}: {emoji}")


if __name__ == "__main__":
    main()
