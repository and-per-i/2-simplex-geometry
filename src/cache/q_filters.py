"""
Q-Filters per KV cache eviction nei layer standard (StudentAttention).

Tecnica training-free basata su SVD dello spazio delle Query (Agarwal et al. 2025).
Il vettore singolare dominante f dello spazio Query di un layer è un buon proxy
per stimare l'importanza di ogni token: score(j) = f^T K_j.

Workflow:
1. Calibrazione (offline, una sola volta):
   calibrate_q_filters(model, sequences) → dict {layer_idx: Tensor}

2. Eviction (a runtime, dentro il forward):
   eviction = QFilterEviction(filters)
   K_kept, V_kept = eviction.evict(layer_idx, K_cache, V_cache, budget)

Nota: i Q-Filters sono applicabili solo ai layer con attenzione standard
(StudentAttention, score bilineare Q·K^T). Per i layer 2-Simpliciali usare
L2Eviction (src/cache/l2_eviction.py).

Riferimento: "Q-Filters: Training-Free KV Cache Eviction" (arXiv 2503.02812)
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch import Tensor


class QFilterEviction:
    """
    Eviction per layer standard basata sui Q-Filters.

    Args:
        filters: dizionario {layer_idx: Tensor (num_heads, head_dim)}
                 prodotto da calibrate_q_filters().
    """

    def __init__(self, filters: Dict[int, Tensor]):
        self.filters = filters  # {layer_idx: (H, D)}

    def score(self, layer_idx: int, K: Tensor) -> Tensor:
        """
        Calcola lo score di importanza per ogni token nella cache.

        Score = f^T K_j aggregato sulle head.
        Token con score ALTO sono più importanti (da mantenere).

        Args:
            layer_idx: indice del layer
            K: (B, H, S, D) — key vectors nella cache

        Returns:
            scores: (B, S) — score per token
        """
        if layer_idx not in self.filters:
            # Nessun filtro disponibile → fallback su norma L2 inversa
            return -K.norm(dim=-1).mean(dim=1)  # (B, S), più alto = più importante

        f = self.filters[layer_idx]  # (H, D)
        f = f.to(K.device)

        # Proiezione: per ogni head h, score_j^h = f_h^T K_{j,h}
        # K: (B, H, S, D), f: (H, D) → (H, D, 1)
        # einsum: (B, H, S, D) × (H, D) → (B, H, S)
        scores_per_head = torch.einsum("bhsd,hd->bhs", K, f)  # (B, H, S)

        # Media sulle head → (B, S)
        return scores_per_head.mean(dim=1)

    def evict(
        self,
        layer_idx: int,
        K: Tensor,
        V: Tensor,
        budget: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Mantieni i `budget` token con score più alto.

        Args:
            layer_idx: indice del layer
            K: (B, H, S, D)
            V: (B, H, S, D)
            budget: numero massimo di token da tenere

        Returns:
            K_kept, V_kept — stessa forma ma S ≤ budget
        """
        S = K.shape[2]
        if S <= budget:
            return K, V

        scores = self.score(layer_idx, K)   # (B, S)
        mean_scores = scores.mean(dim=0)    # (S,) — media sul batch

        # Tieni i `budget` token con score più alto
        _, keep_idx = torch.topk(mean_scores, budget, largest=True)
        keep_idx, _ = keep_idx.sort()       # ordine causale

        K_kept = K[:, :, keep_idx, :]
        V_kept = V[:, :, keep_idx, :]
        return K_kept, V_kept


# ---------------------------------------------------------------------------
# Calibrazione offline
# ---------------------------------------------------------------------------

def calibrate_q_filters(
    model,
    sequences: List[Tensor],
    n_samples: int = 512,
    device: Optional[str] = None,
) -> Dict[int, Tensor]:
    """
    Esegue forward pass (no grad) su `n_samples` sequenze e calcola i Q-Filters
    tramite SVD per ogni layer standard.

    Args:
        model:     StudentForCausalLM con pesi caricati
        sequences: lista di tensori (S,) di token IDs (sequenze geometriche)
        n_samples: numero di sequenze da usare per la calibrazione
        device:    dispositivo su cui spostare il modello (None = usa il modello as-is)

    Returns:
        filters: {layer_idx: Tensor (num_heads, head_dim)}
                 uno per ogni layer standard (non simpliciale)
    """
    import torch
    from .l2_eviction import L2Eviction

    if device is not None:
        model = model.to(device)
    model.eval()

    # Identifica i layer standard (non simpliciali, non bypassati)
    simplex_layers = set(getattr(model.config, "simplex_layers", []))
    standard_layer_indices = [
        i for i, layer in enumerate(model.layers)
        if not layer.is_simplex and not layer.is_bypassed.item()
    ]

    if not standard_layer_indices:
        print("⚠️  Nessun layer standard trovato — nessun Q-Filter calcolato.")
        return {}

    # Accumulatore: per ogni layer standard, lista di Q tensori (S, D) per ogni head
    q_accumulator: Dict[int, List[Tensor]] = {i: [] for i in standard_layer_indices}

    # Hook per catturare Q durante il forward
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args, output):
            # output è (attn_out, present); present = (K, V)
            # Recuperiamo Q dall'interno del forward tramite args
            # args[0] = hidden_states dopo ln1
            hidden_states = args[0]
            with torch.no_grad():
                Q = module.q_proj(hidden_states)   # (B, S, H*D)
                B, S, _ = Q.shape
                H = module.num_heads
                D = module.head_dim
                Q = Q.view(B, S, H, D)             # (B, S, H, D)
                # Accumula: (B*S, H, D) → per ogni head (B*S, D)
                q_accumulator[layer_idx].append(Q.detach().cpu())
        return hook

    for idx in standard_layer_indices:
        h = model.layers[idx].attention.register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Forward pass su `n_samples` sequenze
    sequences = sequences[:n_samples]
    _dev = next(model.parameters()).device

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if i % 50 == 0:
                print(f"  Calibrazione Q-Filters: {i}/{len(sequences)} sequenze...")
            ids = seq.unsqueeze(0).to(_dev)    # (1, S)
            model(ids, use_cache=False)        # forward senza cache

    # Rimuovi hook
    for h in hooks:
        h.remove()

    # SVD per ogni layer e ogni head
    filters: Dict[int, Tensor] = {}
    for layer_idx in standard_layer_indices:
        if not q_accumulator[layer_idx]:
            continue

        Q_all = torch.cat(q_accumulator[layer_idx], dim=0)  # (N_total, S, H, D)
        B_tot, S_tot, H, D = Q_all.shape
        Q_flat = Q_all.permute(0, 1, 2, 3).reshape(-1, H, D)  # (N*S, H, D)

        head_filters = []
        for h in range(H):
            Q_h = Q_flat[:, h, :]              # (N*S, D)
            # SVD: primo vettore singolare destro = direzione principale
            try:
                _, _, Vh = torch.linalg.svd(Q_h.float(), full_matrices=False)
                f_h = Vh[0]                    # (D,)
            except Exception:
                # Fallback: norma media (degenera in L2 eviction inversa)
                f_h = Q_h.mean(dim=0)
                f_h = f_h / (f_h.norm() + 1e-7)
            head_filters.append(f_h)

        filters[layer_idx] = torch.stack(head_filters, dim=0)  # (H, D)
        print(f"  ✅ Layer {layer_idx}: Q-Filter calcolato (H={H}, D={D})")

    return filters


def save_q_filters(filters: Dict[int, Tensor], path: str):
    """Salva i Q-Filters su disco."""
    torch.save(filters, path)
    print(f"✅ Q-Filters salvati in {path}")


def load_q_filters(path: str) -> Dict[int, Tensor]:
    """Carica i Q-Filters da disco."""
    filters = torch.load(path, map_location="cpu")
    print(f"✅ Q-Filters caricati da {path} ({len(filters)} layer)")
    return filters
