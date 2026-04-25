"""
FASE 2 & 3: Student Model con Progressive Pruning
==================================================

Modello Student che supporta:
1. Attenzione 2-Simpliciale (usa K e K' per layer designati)
2. Bypass dinamico dei layer durante training (Progressive Pruning)
3. Residual connections che permettono bypass senza rompersi

Architecture:
- 12 layers iniziali (ridotti progressivamente a 6)
- 3 layer simpliciali (indici 3, 7, 11)
- Hidden dim: 384 (vs 1024 del Teacher)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class SimplexAttention(nn.Module):
    """
    Attenzione 2-Simpliciale: usa K e K' per creare due spazi di rappresentazione.
    
    Formula:
        Attn(Q, K, V) = softmax(QK^T / √d) V  (standard)
        Attn_simplex = α * Attn(Q, K, V) + (1-α) * Attn(Q, K', V)
    
    Dove K' è la matrice "clone" con identità perturbata.
    """
    
    def __init__(self, dim_hidden: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.head_dim = dim_hidden // num_heads
        assert self.head_dim * num_heads == dim_hidden, "dim_hidden deve essere divisibile per num_heads"
        
        # Matrici standard
        self.W_Q = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.W_K = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.W_V = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.W_O = nn.Linear(dim_hidden, dim_hidden, bias=False)
        
        # Matrice K' per attenzione simpliciale (sarà inizializzata dalla Fase 1)
        self.W_K_prime = nn.Linear(dim_hidden, dim_hidden, bias=False)
        
        # Coefficiente di mixing (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, use_simplex: bool = True) -> torch.Tensor:
        """
        Forward pass con attenzione simpliciale.
        
        Args:
            x: Input tensor (batch, seq_len, dim_hidden)
            use_simplex: Se True, usa K + K' (Simplex). Se False, solo K (standard)
        
        Returns:
            Output tensor con stessa shape dell'input
        """
        B, T, C = x.shape
        
        # Proiezioni Q, K, V
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attenzione standard
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        
        # Causal Mask (NON SBIRCIARE IL FUTURO!)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out_standard = attn @ V  # (B, nh, T, hd)
        
        if use_simplex:
            # Proiezione K' (clone perturbato)
            K_prime = self.W_K_prime(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attenzione con K'
            scores_prime = (Q @ K_prime.transpose(-2, -1)) / self.scale
            scores_prime = scores_prime.masked_fill(causal_mask, float('-inf'))
            attn_prime = torch.softmax(scores_prime, dim=-1)
            attn_prime = self.dropout(attn_prime)
            out_prime = attn_prime @ V
            
            # Mixing simpliciale
            out = self.alpha * out_standard + (1 - self.alpha) * out_prime
        else:
            out = out_standard
        
        # Reassemble heads e proiezione output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)


class MLP(nn.Module):
    """Feed-Forward Network standard."""
    
    def __init__(self, dim_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim_hidden, 4 * dim_hidden)
        self.fc2 = nn.Linear(4 * dim_hidden, dim_hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Blocco Transformer con supporto per:
    1. Bypass dinamico (Progressive Pruning)
    2. Attenzione Simpliciale opzionale
    """
    
    def __init__(self, dim_hidden: int, num_heads: int, is_simplicial: bool = False, dropout: float = 0.1):
        super().__init__()
        self.is_simplicial = is_simplicial
        
        # Flag per Progressive Pruning (Fase 3)
        self.is_bypassed = False
        
        # Attention (Simpliciale o Standard)
        if is_simplicial:
            self.attention = SimplexAttention(dim_hidden, num_heads, dropout)
        else:
            # Usiamo SimplexAttention ma con use_simplex=False per uniformità
            self.attention = SimplexAttention(dim_hidden, num_heads, dropout)
        
        self.mlp = MLP(dim_hidden, dropout)
        self.ln1 = nn.LayerNorm(dim_hidden)
        self.ln2 = nn.LayerNorm(dim_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward con residual connections e bypass support.
        
        Se is_bypassed=True: ritorna x senza modifiche (Fase 3: Progressive Pruning)
        Altrimenti: normale operazione residuale
        """
        if self.is_bypassed:
            # BYPASS ATTIVO: il gradiente si ferma, il layer è "spento"
            return x
        
        # Residual connections standard
        use_simplex = self.is_simplicial
        x = x + self.attention(self.ln1(x), use_simplex=use_simplex)
        x = x + self.mlp(self.ln2(x))
        return x


class StudentModelProgressive(nn.Module):
    """
    MODELLO STUDENT COMPLETO per Knowledge Distillation con Progressive Pruning.
    
    Features:
    - 12 layers iniziali (→ 6 dopo pruning)
    - 3 layer simpliciali (posizioni 3, 7, 11 in 0-indexed)
    - Bypass dinamico per Progressive Pruning
    - Compatibile con KD Loss
    """
    
    def __init__(self, vocab_size: int = 1024, dim_hidden: int = 384, 
                 num_layers: int = 12, num_heads: int = 8,
                 simplicial_layers: list = [3, 7, 11],
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.simplicial_layers = simplicial_layers
        
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, dim_hidden)
        self.position_embedding = nn.Embedding(max_seq_len, dim_hidden)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim_hidden=dim_hidden,
                num_heads=num_heads,
                is_simplicial=(i in simplicial_layers),
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(dim_hidden)
        self.lm_head = nn.Linear(dim_hidden, vocab_size, bias=False)
        
        # Weight tying (opzionale: lega embedding e lm_head)
        # self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inizializzazione pesi standard."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass completo.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            labels: Target tokens per calcolo loss (opzionale)
        
        Returns:
            Se labels=None: solo logits (batch, seq_len, vocab_size)
            Se labels forniti: dict con 'logits' e 'loss'
        """
        B, T = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, dim_hidden)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer layers (alcuni potrebbero essere bypassed)
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calcolo loss se labels forniti (Shift per autoregressive!)
        loss = None
        if labels is not None:
            # Shift dei logits e labels: predici il token T+1 dato il token T
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss} if loss is not None else logits
    
    def get_active_layers(self) -> list:
        """Ritorna lista di indici dei layer attivi (non bypassed)."""
        return [i for i, layer in enumerate(self.layers) if not layer.is_bypassed]
    
    def get_bypassed_layers(self) -> list:
        """Ritorna lista di indici dei layer bypassed."""
        return [i for i, layer in enumerate(self.layers) if layer.is_bypassed]
    
    def count_parameters(self) -> dict:
        """Conta parametri totali e trainable."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Conta parametri per layer tipo
        attention_params = sum(
            p.numel() for layer in self.layers 
            for p in layer.attention.parameters()
        )
        mlp_params = sum(
            p.numel() for layer in self.layers 
            for p in layer.mlp.parameters()
        )
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'attention': attention_params,
            'mlp': mlp_params,
            'embedding': self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        }
