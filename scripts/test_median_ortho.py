"""
test_median_ortho.py — Test del sistema neuro-simbolico su due problemi canonici.

Il modello è stato addestrato sul formato:
  <problem> pt : pred args [step] ; ... ? goal </problem>
e risponde con:
  <aux> x00 new_pt : pred args [step] ; </aux> <proof> ... </proof>

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

# Formato dei problemi:
#   jgex   → stringa JGEX per Newclid (verifica simbolica)
#   prompt → stringa <problem>...</problem> che il modello ha visto in training
PROBLEMS = [
    {
        "name": "Mediana (punto medio)",
        "jgex":   "a b c = triangle a b c; m = midpoint m a b ? cong c m m a",
        "prompt": "<problem> a : ; b : ; c : ; m : midp m a b [000] ? cong c m m a </problem>",
        "next_step": 1,
        "difficulty": "base",
    },
    {
        "name": "Ortocentro (concorrenza altezze)",
        "jgex":   "a b c = triangle a b c; d = on_tline d a b c; e = on_tline e b a c; o = on_line o a d, on_line o b e ? on_tline o c a b",
        "prompt": "<problem> a : ; b : ; c : ; d : perp a d b c [000] coll d b c [001] ; e : perp b e a c [002] coll e a c [003] ; o : coll o a d [004] coll o b e [005] ? perp c o a b </problem>",
        "next_step": 6,
        "difficulty": "medio",
    },
]

# Mapping predicate → JGEX construction (per tradurre l'output del modello)
AUX_TO_JGEX = {
    "midp":      "midpoint",
    "midpoint":  "midpoint",
    "coll":      "on_line",
    "perp":      "on_tline",
    "para":      "on_pline",
    "cong":      "eqdistance",
    "circle":    "on_circum",
    "cyclic":    "on_circum",
}


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
def get_suggestions(model, tok, prompt, device, k=256, temp=0.9, batch_size=64):
    """Genera k output dal modello, restituisce (raw_text, aux_construction)."""
    raw_ids = tok.encode(prompt, add_special_tokens=True)
    eos_id = tok.eos_token_id
    results = []

    num_batches = (k + batch_size - 1) // batch_size
    for b in range(num_batches):
        bs = min(batch_size, k - b * batch_size)
        ids = torch.tensor([raw_ids], device=device).repeat(bs, 1)
        finished = torch.zeros(bs, dtype=torch.bool, device=device)

        for _ in range(64):
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
            gen = tok.decode(ids[i].tolist(), skip_special_tokens=False)
            # Estrai solo la parte generata dopo il prompt
            if "</problem>" in gen:
                after = gen.split("</problem>", 1)[1].strip()
            else:
                after = gen[len(tok.decode(raw_ids, skip_special_tokens=False)):].strip()
            results.append(after)

    return results


def parse_aux(raw_output):
    """
    Estrae le costruzioni ausiliarie dall'output del modello.
    Formato atteso: <aux> x00 pt_name : pred args [N] ; ... </aux>
    Restituisce lista di (point_name, predicate, args).
    """
    aux_match = re.search(r"<aux>(.*?)</aux>", raw_output, re.DOTALL)
    if not aux_match:
        return []

    content = aux_match.group(1).strip()
    # Rimuovi "x00" iniziale
    content = re.sub(r"^\s*x\d+\s*", "", content)

    constructions = []
    # Ogni costruzione: "pt_name : pred arg1 arg2 ... [N] ;"
    for clause in content.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        # "pt_name : pred args [N]"
        m = re.match(r"([a-z]\w*)\s*:\s*(.+?)(?:\s*\[\d+\])?$", clause, re.IGNORECASE)
        if m:
            pt_name = m.group(1)
            rest = re.sub(r"\[\d+\]", "", m.group(2)).strip()
            parts = rest.split()
            if parts:
                pred = parts[0]
                args = parts[1:]
                constructions.append((pt_name, pred, args))

    return constructions


def aux_to_newclid(pt_name, pred, args):
    """Traduce (pt_name, pred, args) in una clausola JGEX aggiuntiva."""
    if pred not in AUX_TO_JGEX:
        return ""
    jgex_pred = AUX_TO_JGEX[pred]

    # Filtra solo lettere valide come nomi di punto
    clean_args = [a for a in args if re.match(r'^[a-z]$', a, re.IGNORECASE)]

    if jgex_pred == "midpoint":
        if len(clean_args) >= 2:
            return f"; {pt_name} = midpoint {pt_name} {clean_args[0]} {clean_args[1]}"
    elif jgex_pred == "on_line":
        if len(clean_args) >= 2:
            return f"; {pt_name} = on_line {pt_name} {clean_args[0]} {clean_args[1]}"
    elif jgex_pred in ("on_tline", "on_pline"):
        if len(clean_args) >= 3:
            return f"; {pt_name} = {jgex_pred} {pt_name} {clean_args[0]} {clean_args[1]} {clean_args[2]}"
    elif jgex_pred == "eqdistance":
        if len(clean_args) >= 3:
            return f"; {pt_name} = eqdistance {pt_name} {clean_args[0]} {clean_args[1]} {clean_args[2]}"
    elif jgex_pred == "on_circum":
        if len(clean_args) >= 3:
            return f"; {pt_name} = on_circum {pt_name} {clean_args[0]} {clean_args[1]} {clean_args[2]}"
    return ""


def test_problem(problem, model, tok, device, k=256, max_depth=2, verbose=True):
    name = problem["name"]
    jgex = problem["jgex"]
    prompt = problem["prompt"]

    print(f"\n{'='*70}")
    print(f"Problema: {name}  [{problem['difficulty']}]")
    print(f"JGEX:   {jgex}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")

    # Fase 1: Newclid puro
    print("Fase 1 — Newclid puro: ", end="", flush=True)
    ok, _, _ = solve_worker((jgex, 15, 0))
    if ok:
        print("RISOLTO senza AI")
        return "pure"
    print("fallito")

    # Fase 2: modello + Newclid
    jgex_setup, jgex_goal = jgex.split("?")
    current_nodes = [(jgex, prompt)]

    for depth in range(1, max_depth + 1):
        k_this = k if depth == 1 else 32
        print(f"\nFase 2 depth={depth} — Generazione {k_this} suggerimenti...")
        candidates = []
        shown = 0

        for cur_jgex, cur_prompt in current_nodes:
            outputs = get_suggestions(model, tok, cur_prompt, device, k=k_this)

            for raw in outputs:
                # Mostra i primi output al primo depth
                if verbose and depth == 1 and shown < 8:
                    print(f"  [Output modello]: {raw[:120]!r}")
                    shown += 1

                constructions = parse_aux(raw)
                for pt_name, pred, args in constructions:
                    jgex_clause = aux_to_newclid(pt_name, pred, args)
                    if jgex_clause:
                        aug = f"{jgex_setup.strip()} {jgex_clause} ? {jgex_goal.strip()}"
                        new_prompt = cur_prompt.replace(
                            "</problem>",
                            f"<aux> x00 {pt_name} : {pred} {' '.join(args)} ; </aux></problem>"
                        )
                        candidates.append((aug, new_prompt, f"{pt_name}:{pred} {' '.join(args)}"))

        if not candidates:
            print("  Nessuna costruzione JGEX valida estratta dagli output.")
            continue

        # Deduplica
        seen = set()
        unique = []
        for c in candidates:
            if c[2] not in seen:
                seen.add(c[2])
                unique.append(c)
        candidates = unique

        print(f"  {len(candidates)} costruzioni uniche → test con Newclid...")
        for pt_name_pred, _, _ in candidates[:10]:
            print(f"    {pt_name_pred}")

        tasks = [(c[0], 15, i) for i, c in enumerate(candidates)]
        with mp.Pool(max(1, mp.cpu_count())) as pool:
            try:
                from tqdm import tqdm
                it = tqdm(pool.imap_unordered(solve_worker, tasks),
                          total=len(tasks), desc=f"Newclid D{depth}")
            except ImportError:
                it = pool.imap_unordered(solve_worker, tasks)
            for success, tid, _ in it:
                if success:
                    print(f"\n  ✅ DIMOSTRATO a depth {depth}!")
                    print(f"  Costruzione: {candidates[tid][2]}")
                    pool.terminate()
                    return "ai"

        print(f"  ❌ Nessun successo a depth {depth}.")
        current_nodes = [(c[0], c[1]) for c in candidates[:32]]

    return "fail"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
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
        results[prob["name"]] = test_problem(
            prob, model, tok, device, k=args.k, max_depth=args.depth
        )

    print(f"\n{'='*70}")
    print("RIEPILOGO")
    print(f"{'='*70}")
    for name, res in results.items():
        label = "Newclid puro" if res == "pure" else ("AI+Newclid ✅" if res == "ai" else "FAIL ❌")
        print(f"  {name}: {label}")


if __name__ == "__main__":
    main()
