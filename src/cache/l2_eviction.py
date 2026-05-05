"""
L2-norm eviction per layer 2-Simpliciali.

Strategia: ogni token j nella cache ha importanza determinata da
    score(j) = max(‖K_j‖₂, ‖K'_j‖₂)
dove K_j è il vettore key primario e K'_j quello secondario.

Un score BASSO indica un token potenzialmente IMPORTANTE (norma piccola →
proiezioni più "centrali" nello spazio latente, più influenti nell'attenzione
trilineare). In eviction teniamo i `budget` token con score più basso.

Questa è l'estensione naturale di Devoto et al. (2024) al caso trilineare:
    |A_ijk| ≤ (1/√d) ‖q_i‖ · ‖k_j‖ · ‖k'_k‖
La norma ‖k_j‖ è un upper bound sull'importanza di j per qualsiasi partner k.
Poiché il token j gioca sia il ruolo di chiave primaria (asse j) che secondaria
(asse k), usiamo max(‖K_j‖, ‖K'_j‖) come stima conservativa.

Riferimenti:
  - Devoto et al. (2024) "Efficient LLM Inference by Token Eviction" (EMNLP)
  - Q-Filters (2025) — per la giustificazione SVD sui layer standard
"""

from typing import Tuple

import torch
from torch import Tensor


class L2Eviction:
    """
    Eviction content-based per la KV cache dei layer 2-Simpliciali.

    Uso tipico (dentro TwoSimplicialAttention.forward):
        eviction = L2Eviction()
        K, Kp, V = eviction.evict(K_full, Kp_full, V_full, budget=8)
    """

    def score(self, K: Tensor, Kp: Tensor) -> Tensor:
        """
        Calcola il punteggio di importanza per ogni token nella cache.

        Args:
            K:  (B, S, H, D) — key primarie (già nella cache, non normalizzate)
            Kp: (B, S, H, D) — key secondarie

        Returns:
            scores: (B, S) — score per token, aggregato su head.
                    Score BASSO = token IMPORTANTE (da mantenere).
        """
        # Norma L2 per ogni token su tutte le head → media sulle head
        # K: (B, S, H, D) → norm su D → (B, S, H) → media su H → (B, S)
        k_norm  = K.norm(dim=-1).mean(dim=-1)   # (B, S)
        kp_norm = Kp.norm(dim=-1).mean(dim=-1)  # (B, S)

        # max per essere conservativi: teniamo il token se è importante
        # in ALMENO uno dei due ruoli (primario o secondario)
        return torch.maximum(k_norm, kp_norm)   # (B, S)

    def evict(
        self,
        K: Tensor,
        Kp: Tensor,
        V: Tensor,
        budget: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Seleziona i `budget` token più importanti dalla cache.

        Se la cache ha ≤ budget token non fa nulla.
        I token selezionati mantengono il loro ordine temporale originale
        (non vengono riordinati) per preservare la struttura causale.

        Args:
            K:      (B, S, H, D) — key primarie
            Kp:     (B, S, H, D) — key secondarie
            V:      (B, S, H, D) — value
            budget: numero massimo di token da tenere

        Returns:
            K_kept, Kp_kept, V_kept — tensori filtrati, stessa forma ma S ≤ budget
        """
        S = K.shape[1]
        if S <= budget:
            return K, Kp, V

        scores = self.score(K, Kp)  # (B, S)

        # Tieni i `budget` token con score più basso (= più importanti)
        # topk con largest=False → indici dei minimi
        # Usiamo il batch 0 come riferimento per la selezione degli indici.
        # In generation mode B=batch_size ma tutti i token hanno le stesse
        # posizioni, quindi un'unica selezione per batch è coerente.
        # Per semplicità usiamo la media del batch come score globale.
        mean_scores = scores.mean(dim=0)                        # (S,)
        _, keep_idx = torch.topk(mean_scores, budget, largest=False)
        keep_idx, _ = keep_idx.sort()                           # ordine causale

        K_kept  = K[:, keep_idx]
        Kp_kept = Kp[:, keep_idx]
        V_kept  = V[:, keep_idx]

        return K_kept, Kp_kept, V_kept
