"""
TwoSimplicialAttention — attenzione 2-Simpliciale con sliding window causale.

Implementa il triplo prodotto interno:
    A_ijk = (1/√d) <q_i, k_j, k'_k>  con j,k ∈ finestra causale [i-w+1, i]

Supporta:
  - Triton kernel (GPU) con fallback PyTorch automatico
  - KV cache per inferenza autoregressiva efficiente:
      cache = (K, Kp, V) shape (B, S_past, H, D)
  - L2-norm eviction (Devoto et al. 2024) per selezione content-based
    dei token da mantenere nella cache (sostituisce la selezione by-recency
    del sliding window)
  - L2 normalizzazione per-head di Q, K, Kp per stabilità numerica BF16
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoSimplicialAttention(nn.Module):
    """
    Attenzione 2-Simpliciale con sliding window causale.

    Equazioni principali:
        Q  = X W_Q,  K  = X W_K,  V  = X W_V,  K' = X W_K'
        A_ijk = (1/√d) Σ_d  q_{i,d} · k_{j,d} · k'_{k,d}
        S_ijk = softmax_{j,k}(A_ijk)
        ỹ_i   = Σ_{j,k} S_ijk · (V_j ⊙ V'_k)     con V' = V (shared)
        y_i   = W_O · ỹ_i  [+ x_i  se with_residual]

    Cache format (past_key_value):
        (K_cache, Kp_cache, V_cache)  shape (B, S_past, H, D) ciascuno
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        with_residual: bool = True,
        with_norm: bool = True,
        use_triton_kernel: bool = True,
        w1: int = 8,
        w2: int = 8,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim) if out_dim is not None else self.in_dim
        self.num_heads = int(num_heads)
        assert self.out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = self.out_dim // self.num_heads
        self.with_residual = with_residual
        self.with_norm = with_norm
        self.use_triton_kernel = bool(use_triton_kernel)
        self.w1 = w1
        self.w2 = w2

        # Proiezioni lineari (nomi allineati con i checkpoint Phase 1)
        self.W_Q       = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.W_K       = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.W_V       = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.W_K_prime = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.W_O       = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(self.out_dim) if with_norm else nn.Identity()

    # ------------------------------------------------------------------
    # Forward principale
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        l2_eviction=None,
        token_budget: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:             (B, S, in_dim) o (N, in_dim)
            attention_mask: (B, S) padding mask opzionale
            past_key_value: (K_past, Kp_past, V_past) dalla chiamata precedente
            l2_eviction:   istanza di L2Eviction (opzionale)
            token_budget:  numero max di token da tenere in cache dopo eviction

        Returns:
            out:    (B, S, out_dim)
            present:(K_full, Kp_full, V_full) — cache aggiornata
        """
        is_batched = x.dim() == 3
        if is_batched:
            B, S, _ = x.shape
        else:
            if x.dim() != 2:
                raise ValueError("input x must be (B, S, in_dim) or (N, in_dim)")
            B, S = 1, x.shape[0]

        # --- Proiezioni per i nuovi token ---
        Q_new  = self.W_Q(x).view(B, S, self.num_heads, self.head_dim)
        K_new  = self.W_K(x).view(B, S, self.num_heads, self.head_dim)
        V_new  = self.W_V(x).view(B, S, self.num_heads, self.head_dim)
        Kp_new = self.W_K_prime(x).view(B, S, self.num_heads, self.head_dim)

        # --- Aggiornamento cache ---
        if past_key_value is not None:
            K_past, Kp_past, V_past = past_key_value
            K_full  = torch.cat([K_past,  K_new],  dim=1)  # (B, S_past+S, H, D)
            Kp_full = torch.cat([Kp_past, Kp_new], dim=1)
            V_full  = torch.cat([V_past,  V_new],  dim=1)
        else:
            K_full, Kp_full, V_full = K_new, Kp_new, V_new

        # --- L2 eviction (content-based): seleziona i token più importanti ---
        if l2_eviction is not None and token_budget is not None:
            K_full, Kp_full, V_full = l2_eviction.evict(
                K_full, Kp_full, V_full, token_budget
            )

        # Cache da restituire (dopo eviction, prima della normalizzazione)
        present = (K_full, Kp_full, V_full)

        # --- L2 normalizzazione per-head (stabilità BF16) ---
        Q  = Q_new / (Q_new.norm(dim=-1, keepdim=True) + 1e-7)
        K  = K_full  / (K_full.norm(dim=-1, keepdim=True)  + 1e-7)
        Kp = Kp_full / (Kp_full.norm(dim=-1, keepdim=True) + 1e-7)
        # V non normalizzato (è il valore, non la chiave)
        Vp = V_full  # V' condiviso con V

        # --- Kernel Triton o fallback PyTorch ---
        #
        # Il kernel Triton implementa internamente la sliding window causale:
        #   for kv1_idx in range(max(0, q_start - w1), min(seq_len, q_end)):
        # Quindi va passato K_full (stessa seq_len di Q), non K_win.
        # Passare K_win (solo w1 token) causerebbe accessi OOB quando
        # seq_len > w1 perché il kernel itera fino a seq_len su K.
        #
        # Il fallback PyTorch invece riceve K_win perché fa il windowing
        # esplicitamente in _forward_pytorch.
        #
        # Il kernel Triton è usato solo in training (no past_key_value) dove
        # K_full.shape[1] == Q.shape[1] (stessa seq_len).  Per l'inferenza
        # con KV cache si usa sempre il fallback PyTorch.
        use_triton = (
            self.use_triton_kernel
            and x.is_cuda
            and past_key_value is None          # training only
            and K_full.shape[1] == S            # K_full ha la stessa seq di Q
        )

        if use_triton:
            try:
                from ..kernels.two_simplicial_autograd import TwoSimplicialAttentionFunction
                Z = TwoSimplicialAttentionFunction.apply(
                    x, Q, K, V_full, Kp, V_full,   # K, Kp, V già L2-normalizzati; V_full per valori
                    self.out_dim, self.num_heads, self.head_dim,
                    self.w1, self.w2,
                )
            except Exception as e:
                print(f"⚠️ [TwoSimplicialAttention] Triton kernel failed, fallback PyTorch: {e}")
                use_triton = False

        if not use_triton:
            # Sliding window esplicita per il fallback PyTorch
            K_win  = K[:, -self.w1:]
            Kp_win = Kp[:, -self.w2:]
            V_win  = V_full[:, -self.w1:]
            Vp_win = V_full[:, -self.w2:]
            Z = self._forward_pytorch(Q, K_win, V_win, Kp_win, Vp_win,
                                      attention_mask=attention_mask)

        # --- Output projection ---
        Z_concat = Z.reshape(x.shape[:-1] + (self.out_dim,)) if is_batched else Z.reshape(B * S, self.out_dim)
        out = self.W_O(Z_concat)
        if not is_batched:
            out = out.squeeze(0)

        if self.with_residual:
            if out.shape == x.shape:
                out = out + x
            else:
                import logging
                logging.warning(
                    f"Residual skipped in TwoSimplicialAttention: {out.shape} vs {x.shape}"
                )
        if self.with_norm:
            out = self.norm(out)

        return out, present

    # ------------------------------------------------------------------
    # PyTorch fallback (CPU / debug)
    # ------------------------------------------------------------------

    def _forward_pytorch(
        self,
        Q: torch.Tensor,
        K_win: torch.Tensor,
        V_win: torch.Tensor,
        Kp_win: torch.Tensor,
        Vp_win: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implementazione PyTorch vectorized con finestra causale.

        Args:
            Q:      (B, S_q, H, D) — query per i nuovi token
            K_win:  (B, w1, H, D)  — chiavi primarie nella finestra
            V_win:  (B, w1, H, D)  — value nella finestra
            Kp_win: (B, w2, H, D)  — chiavi secondarie
            Vp_win: (B, w2, H, D)  — value secondari (= V_win se V'=V)

        Returns:
            Z: (B, S_q, H, D)
        """
        B, S_q, H, D = Q.shape
        w1 = K_win.shape[1]
        w2 = Kp_win.shape[1]
        device = Q.device

        # Score trilineare: A[b, s, h, j, k] = sum_d Q[b,s,h,d]*K[b,j,h,d]*Kp[b,k,h,d]
        # Espandiamo le dimensioni per il broadcast:
        # Q:      (B, S_q, H, D) → (B, S_q, H, 1,  1,  D)
        # K_win:  (B, w1,  H, D) → (B, 1,   H, w1, 1,  D)
        # Kp_win: (B, w2,  H, D) → (B, 1,   H, 1,  w2, D)
        A = torch.einsum("bshd,bjhd,bkhd->bshjk", Q, K_win, Kp_win) / (D ** 0.5)
        # A: (B, S_q, H, w1, w2)

        # Maschera causale nella finestra.
        #
        # DERIVAZIONE: query al new-token index s_idx (0-based) ha posizione assoluta
        # (S_q - w1_eff + s_idx + offset), ma la formula semplificata è:
        #   maschera K_win[j] se j > s_idx + offset
        # dove offset = w1_eff - S_q (numero di token "del passato" già nella finestra).
        #
        # Questo garantisce la CONSISTENCY causale: il risultato per query s_idx
        # è lo stesso sia in prefill (S_q > 1) che in generation (S_q=1),
        # indipendentemente da quanti altri token vengono processati assieme.
        #
        # Verifica:
        #   - No past, S_q=8, w1_eff=8: offset=0 → mask j>s_idx (standard lower-tri causal) ✓
        #   - Past=7, S_q=1, w1_eff=8:  offset=7 → mask j>7 (nulla mascherato, vede tutto il past) ✓
        s_idx  = torch.arange(S_q, device=device).view(1, S_q, 1, 1)
        j_idx  = torch.arange(w1,  device=device).view(1, 1,   w1, 1)
        k_idx  = torch.arange(w2,  device=device).view(1, 1,   1,  w2)
        offset_j = w1 - S_q   # w1_eff - S_q  (w1 == K_win.shape[1] qui)
        offset_k = w2 - S_q
        if S_q > 1:
            mask_j = j_idx > (s_idx + offset_j)   # True → token futuro → maschera
            mask_k = k_idx > (s_idx + offset_k)
            mask = mask_j | mask_k                 # (1, S_q, w1, w2)
            A = A.masked_fill(mask.unsqueeze(2), float("-inf"))

        if attention_mask is not None and S_q > 1:
            A = A.masked_fill(
                attention_mask[:, :S_q, None, None, None] == 0, float("-inf")
            )

        # Softmax congiunto su (j, k).
        # Quando S_q > w1 le query iniziali non hanno chiavi causali nella finestra
        # (tutte mascherate con -inf) → softmax produce NaN.
        # Guardia: per righe interamente -inf produciamo attenzione zero (output = 0 + residual).
        A_flat = A.reshape(B, S_q, H, -1)
        all_masked = (A_flat == float('-inf')).all(dim=-1, keepdim=True)  # (B, S_q, H, 1)
        A_safe = A_flat.masked_fill(all_masked, 0.0)   # evita NaN nel softmax
        S_attn = F.softmax(A_safe, dim=-1)
        S_attn = S_attn.masked_fill(all_masked, 0.0)   # annulla righe che erano tutte mascherate
        S_attn = S_attn.reshape(B, S_q, H, w1, w2)
        S_attn = self.dropout(S_attn)

        # Aggregazione: Z[b,s,h,d] = Σ_{j,k} S[b,s,h,j,k] * V[b,j,h,d] * Vp[b,k,h,d]
        Z = torch.einsum("bshjk,bjhd,bkhd->bshd", S_attn, V_win, Vp_win)
        return Z
