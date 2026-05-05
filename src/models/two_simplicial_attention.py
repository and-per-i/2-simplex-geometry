"""
TwoSimplicialAttention — attenzione 2-Simpliciale con sliding window causale.

Implementa il triplo prodotto interno:
    A_ijk = (1/√d) <q_i, k_j, k'_k>  con j,k ∈ finestra causale [i-w+1, i]

Supporta:
  - Triton kernel (GPU) con fallback PyTorch automatico
  - KV cache per inferenza autoregressiva efficiente
  - L2-norm eviction (Devoto et al. 2024) per selezione content-based
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
            K_full  = torch.cat([K_past,  K_new],  dim=1)
            Kp_full = torch.cat([Kp_past, Kp_new], dim=1)
            V_full  = torch.cat([V_past,  V_new],  dim=1)
        else:
            K_full, Kp_full, V_full = K_new, Kp_new, V_new

        is_generating = past_key_value is not None

        # --- L2 eviction (content-based) ---
        if l2_eviction is not None and token_budget is not None:
            K_full, Kp_full, V_full = l2_eviction.evict(
                K_full, Kp_full, V_full, token_budget
            )

        present = (K_full, Kp_full, V_full)

        # --- Core computation ---
        # Niente L2 normalization: le magnitudini di Q/K/Kp fanno parte
        # delle rappresentazioni apprese. Normalizzarle distruggerebbe
        # il triplo prodotto Q·K·Kp per cui il modello è stato addestrato.

        # Triton path: solo in training (no cache), su CUDA, con K_full == S
        use_triton = (
            self.use_triton_kernel
            and x.is_cuda
            and not is_generating
            and K_full.shape[1] == S
        )

        if use_triton:
            try:
                from ..kernels.two_simplicial_autograd import TwoSimplicialAttentionFunction
                Z = TwoSimplicialAttentionFunction.apply(
                    x, Q_new, K_full, V_full, Kp_full, V_full,
                    self.out_dim, self.num_heads, self.head_dim,
                    self.w1, self.w2,
                )
            except Exception as e:
                print(f"⚠️ [TwoSimplicialAttention] Triton kernel failed, fallback PyTorch: {e}")
                use_triton = False

        if not use_triton:
            if is_generating:
                Z = self._inference_pytorch(Q_new, K_full, V_full, Kp_full,
                                            attention_mask=attention_mask)
            else:
                Z = self._training_pytorch(Q_new, K_full, V_full, Kp_full,
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
    # Training path: per-position causal sliding windows via unfold()
    # ------------------------------------------------------------------

    @staticmethod
    def _get_causal_windows(tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Crea finestre causali per-posizione con padding di zeri a sinistra.

        Per ogni posizione i, la finestra contiene le posizioni [i-w+1, i].
        Le posizioni negative sono rimpiazzate da zeri.
        """
        B, S, H, D = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
        padded = torch.cat([
            torch.zeros(B, window_size - 1, H, D, device=device, dtype=dtype),
            tensor,
        ], dim=1)
        return padded.unfold(1, window_size, 1).permute(0, 1, 4, 2, 3)

    def _training_pytorch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        Kp: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, H, D = Q.shape
        device = Q.device

        K_win  = self._get_causal_windows(K,  self.w1)   # (B, S, w1, H, D)
        Kp_win = self._get_causal_windows(Kp, self.w2)   # (B, S, w2, H, D)
        V_win  = self._get_causal_windows(V,  self.w1)   # (B, S, w1, H, D)
        Vp_win = self._get_causal_windows(V,  self.w2)   # (B, S, w2, H, D)

        A = torch.einsum('bshd,bsjhd,bskhd->bshjk', Q, K_win, Kp_win) / (D ** 0.5)

        # Maschera padding (zeri all'inizio per posizioni < w)
        j_idx = torch.arange(self.w1, device=device).view(1, 1, self.w1)
        s_idx = torch.arange(S, device=device).view(1, S, 1)
        mask1 = j_idx >= (self.w1 - 1 - s_idx)

        k_idx = torch.arange(self.w2, device=device).view(1, 1, self.w2)
        mask2 = k_idx >= (self.w2 - 1 - s_idx)

        mask = (mask1.unsqueeze(3) & mask2.unsqueeze(2)).unsqueeze(2)
        A = A.masked_fill(~mask, float('-inf'))

        if attention_mask is not None:
            A = A.masked_fill(
                attention_mask.view(B, S, 1, 1, 1) == 0, float('-inf')
            )

        S_attn = F.softmax(A.reshape(B, S, H, -1), dim=-1).reshape(B, S, H, self.w1, self.w2)
        S_attn = self.dropout(S_attn)
        Z = torch.einsum('bshjk,bsjhd,bskhd->bshd', S_attn, V_win, Vp_win)
        return Z

    # ------------------------------------------------------------------
    # Inference path: fixed window over cached past + new tokens
    # ------------------------------------------------------------------

    def _inference_pytorch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        Kp: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference con KV cache. Q ha solo i nuovi token (S_q).
        K/Kp/V contengono past + nuovi token. Si usa una finestra
        fissa sugli ultimi w1/w2 token del contesto completo.
        """
        B, S_q, H, D = Q.shape
        S_full = K.shape[1]

        K_win  = K[:,  -self.w1:]   # (B, w1, H, D)
        Kp_win = Kp[:, -self.w2:]   # (B, w2, H, D)
        V_win  = V[:,  -self.w1:]
        Vp_win = V[:,  -self.w2:]

        A = torch.einsum("bshd,bjhd,bkhd->bshjk", Q, K_win, Kp_win) / (D ** 0.5)

        # Maschera causale: i nuovi token vedono solo token ≤ sé stessi
        if S_q == 1:
            # Singolo token: vede tutta la finestra (tutto è passato)
            pass
        else:
            # Prefill: maschera causale standard
            device = Q.device
            s_idx = torch.arange(S_q, device=device).view(1, S_q, 1, 1)
            j_idx = torch.arange(self.w1, device=device).view(1, 1, self.w1, 1)
            k_idx = torch.arange(self.w2, device=device).view(1, 1, 1, self.w2)
            offset_j = self.w1 - S_q
            offset_k = self.w2 - S_q
            mask_j = j_idx > (s_idx + offset_j)
            mask_k = k_idx > (s_idx + offset_k)
            mask = mask_j | mask_k
            A = A.masked_fill(mask.unsqueeze(2), float("-inf"))

        if attention_mask is not None:
            A = A.masked_fill(
                attention_mask[:, :S_q, None, None, None] == 0, float("-inf")
            )

        A_flat = A.reshape(B, S_q, H, -1)
        all_masked = (A_flat == float('-inf')).all(dim=-1, keepdim=True)
        A_safe = A_flat.masked_fill(all_masked, 0.0)
        S_attn = F.softmax(A_safe, dim=-1)
        S_attn = S_attn.masked_fill(all_masked, 0.0)
        S_attn = S_attn.reshape(B, S_q, H, self.w1, self.w2)
        S_attn = self.dropout(S_attn)
        Z = torch.einsum("bshjk,bjhd,bkhd->bshd", S_attn, V_win, Vp_win)
        return Z
