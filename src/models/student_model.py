"""
StudentForCausalLM — Modello Student custom compatibile con HuggingFace Trainer.

Architettura: Decoder-only Transformer (Causal LM) con:
- Token embeddings + Positional embeddings (learnable)
- N layer Transformer decoder con causal self-attention
- LayerNorm finale
- LM head (linear projection → vocab_size logits)

Supporta KV cache per inferenza efficiente:
- Layer standard (StudentAttention): cache (K, V) in formato HuggingFace
- Layer simpliciali (TwoSimplicialAttention): cache (K, Kp, V) con L2-norm eviction opzionale
- past_key_values segue la convenzione HF: lista di tuple, una per layer

Contratto HuggingFace rispettato:
- Eredita da PreTrainedModel
- Restituisce CausalLMOutputWithPast (ha .logits, .loss, .past_key_values)
- Implementa _init_weights per inizializzazione standard HF
- Compatible con .save_pretrained() / .from_pretrained()
- Compatible con .generate() tramite prepare_inputs_for_generation()
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .student_config import StudentConfig
from .two_simplicial_attention import TwoSimplicialAttention


# ---------------------------------------------------------------------------
# Blocchi costitutivi
# ---------------------------------------------------------------------------


class StudentEmbeddings(nn.Module):
    """Token embeddings + positional embeddings learnable."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_length: int = 0,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            # Posizioni incrementali considerando i token già nella cache
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
            # In generation mode con cache, teniamo solo le posizioni per i nuovi token
            if past_length > 0:
                position_ids = position_ids[:, -S:]
        else:
            position_ids = torch.arange(
                past_length, past_length + S, dtype=torch.long, device=device
            ).unsqueeze(0).expand(B, S)

        token_emb = self.token_embedding(input_ids)           # (B, S, H)
        pos_emb = self.position_embedding(position_ids)       # (B, S, H)

        return self.dropout(token_emb + pos_emb)


class StudentAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention standard con supporto KV cache.

    In training (past_key_value=None): elabora l'intera sequenza in parallelo.
    In inferenza (past_key_value fornito): elabora solo il nuovo token e aggiorna la cache.

    Cache format: past_key_value = (K, V)
      K: (B, num_heads, S_past, head_dim)
      V: (B, num_heads, S_past, head_dim)
    """

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = hidden_states.shape

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # (B, S, H) → (B, num_heads, S, head_dim)
            return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(hidden_states))   # (B, H, S, D)
        K_new = split_heads(self.k_proj(hidden_states))
        V_new = split_heads(self.v_proj(hidden_states))

        # Concatena con la cache esistente
        if past_key_value is not None:
            K_past, V_past = past_key_value
            K = torch.cat([K_past, K_new], dim=2)   # (B, H, S_past+S, D)
            V = torch.cat([V_past, V_new], dim=2)
        else:
            K, V = K_new, V_new

        # Cache aggiornata da restituire
        present = (K, V)

        S_full = K.shape[2]   # lunghezza totale sequenza (past + new)

        # Costruzione della maschera attn per SDPA.
        #
        # NOTA CRITICA: PyTorch SDPA con is_causal=True costruisce una maschera
        # lower-triangolare basata sulla sola lunghezza di Q.
        # Se Q ha 1 token ma KV ha 8, la maschera (1×8) risultante è:
        #   [0, -inf, -inf, -inf, -inf, -inf, -inf, -inf]
        # → il nuovo token vede SOLO il KV alla posizione 0, non i 7 token in cache!
        #
        # Soluzione: is_causal=True solo in prefill puro (nessun past, S==S_full).
        # In generation mode (past_key_value exists, S=1): is_causal=False senza maschera
        # → il nuovo token vede correttamente tutti i token in cache.
        sdpa_mask = None
        is_causal = (past_key_value is None)  # True solo in prefill senza cache

        if attention_mask is not None and S > 1:
            # Training / prefill mode: combina padding mask + maschera causale esplicita
            # attention_mask shape: (B, S_full) — copre l'intera sequenza incluso il past
            pad_mask = attention_mask[:, None, None, :S_full].to(dtype=torch.bool)
            sdpa_mask = torch.zeros(B, 1, S, S_full, dtype=Q.dtype, device=Q.device)
            sdpa_mask = sdpa_mask.masked_fill(~pad_mask, float("-inf"))
            causal = torch.triu(
                torch.full((S, S_full), float("-inf"), device=Q.device, dtype=Q.dtype),
                diagonal=S_full - S + 1,
            )
            sdpa_mask = sdpa_mask + causal
            is_causal = False

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # (B, num_heads, S, head_dim) → (B, S, H)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        return self.out_proj(attn_out), present


class StudentFeedForward(nn.Module):
    """FFN con GELU activation (standard nei transformer moderni)."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class StudentTransformerLayer(nn.Module):
    """
    Un singolo layer Transformer con Pre-LayerNorm (più stabile del Post-LN).

    Pre-LN schema:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    Supporta KV cache: riceve e restituisce past_key_value.
    Layer bypassati restituiscono (hidden_states, None).
    """

    def __init__(self, config: StudentConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        # Per-layer decision: simplex_layers (list) takes precedence over global flag
        _simplex_layers = getattr(config, "simplex_layers", [])
        is_simplex = (layer_idx in _simplex_layers) if _simplex_layers else config.use_simplex_attention
        self.is_simplex = is_simplex
        if is_simplex:
            self.attention = TwoSimplicialAttention(
                in_dim=config.hidden_size,
                out_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.dropout_prob,
                use_triton_kernel=getattr(config, "use_triton", True),
                with_residual=False,  # Handled by StudentTransformerLayer
                with_norm=False,      # Handled by StudentTransformerLayer
                w1=config.w1,
                w2=config.w2,
            )
        else:
            self.attention = StudentAttention(config)
        self.mlp = StudentFeedForward(config)
        self.resid_dropout = nn.Dropout(config.dropout_prob)
        # Use a buffer for is_bypassed to ensure synchronization across DDP ranks
        self.register_buffer("is_bypassed", torch.tensor(False, dtype=torch.bool))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        l2_eviction=None,
        token_budget: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[tuple]]:
        # Layer bypassato → identità, nessuna cache
        if self.is_bypassed:
            return hidden_states, None

        # Self-attention con residual
        residual = hidden_states
        hidden_states_ln = self.ln1(hidden_states)

        if self.is_simplex:
            # TwoSimplicialAttention: cache = (K, Kp, V), supporta L2 eviction
            attn_out, present = self.attention(
                hidden_states_ln,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                l2_eviction=l2_eviction,
                token_budget=token_budget,
            )
        else:
            # StudentAttention standard: cache = (K, V)
            attn_out, present = self.attention(
                hidden_states_ln,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
            )

        hidden_states = residual + self.resid_dropout(attn_out)

        # FFN con residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(hidden_states)

        return hidden_states, present


# ---------------------------------------------------------------------------
# Modello principale
# ---------------------------------------------------------------------------


class StudentModel(PreTrainedModel):
    """
    Backbone del modello Student (senza LM head).
    Esposto separatamente per permettere feature-based KD sugli hidden states.
    """

    config_class = StudentConfig

    def __init__(self, config: StudentConfig):
        super().__init__(config)
        self.embeddings = StudentEmbeddings(config)
        self.layers = nn.ModuleList(
            [StudentTransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_ln = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.gradient_checkpointing = False
        self.post_init()  # chiama _init_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        hidden_states = self.embeddings(input_ids, attention_mask=attention_mask)

        all_hidden_states = []
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            all_hidden_states.append(hidden_states)

        hidden_states = self.final_ln(hidden_states)
        return hidden_states, tuple(all_hidden_states)


class StudentForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Modello Student completo per Causal Language Modeling.

    Contratto HuggingFace:
    - Eredita PreTrainedModel (→ save/load, device management, .generate())
    - Restituisce CausalLMOutputWithPast (→ .logits, .loss, .past_key_values)
    - lm_head ha out_features == config.vocab_size (critico per KD con teacher)

    KV Cache:
    - past_key_values: lista di tuple per layer, formato dipende dal tipo di layer
        Standard:    (K, V)       → shape (B, H, S_past, D)
        Simpliciale: (K, Kp, V)  → shape (B, S_past, H, D)
        Bypassed:    None
    - use_cache=True abilita il caching (default in inferenza)
    - In training use_cache=False per evitare overhead di memoria
    """

    config_class = StudentConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: StudentConfig):
        super().__init__(config)
        # Backbone components directly at top level to match checkpoint
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.emb_dropout = nn.Dropout(config.dropout_prob)

        self.layers = nn.ModuleList(
            [StudentTransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)

        # LM head: hidden_size → vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs):
        """Disabilitato per stabilità salvataggio in questa versione."""
        pass

    def _init_weights(self, module: nn.Module):
        """Inizializzazione standard HuggingFace (necessaria per post_init)."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[tuple]]] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        token_type_ids: Optional[torch.LongTensor] = None,
        # Eviction opzionale per layer simpliciali
        l2_eviction=None,
        token_budget: Optional[int] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple]:

        B, S = input_ids.shape
        device = input_ids.device

        # Lunghezza del contesto già in cache (0 se prima chiamata)
        past_length = 0
        if past_key_values is not None:
            # Ricava la lunghezza del past dal primo layer non None
            for pkv in past_key_values:
                if pkv is not None:
                    # Standard (len==2): K shape (B, H, S_past, D) → dim 2
                    # Simpliciale (len==3): K shape (B, S_past, H, D) → dim 1
                    if len(pkv) == 2:
                        past_length = pkv[0].shape[2]   # (B, H, S_past, D)
                    else:
                        past_length = pkv[0].shape[1]   # (B, S_past, H, D)
                    break

        # Position IDs che tengono conto della cache esistente
        if attention_mask is not None:
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
            if past_length > 0:
                position_ids = position_ids[:, -S:]
        else:
            position_ids = torch.arange(
                past_length, past_length + S, dtype=torch.long, device=device
            ).unsqueeze(0).expand(B, S)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        hidden_states = self.emb_dropout(token_emb + pos_emb)

        all_hidden_states = () if output_hidden_states else None
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            pkv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=pkv,
                l2_eviction=l2_eviction if layer.is_simplex else None,
                token_budget=token_budget if layer.is_simplex else None,
            )

            if use_cache:
                new_past_key_values.append(present)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,) + (all_hidden_states if output_hidden_states else ())
            return (loss,) + out if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            past_key_values=tuple(new_past_key_values) if use_cache else None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Richiesto da GenerationMixin per l'autoregressive decoding con KV cache.

        Con past_key_values attivo passa solo l'ultimo token come input_ids,
        riducendo il costo di ogni step da O(S) a O(1) per gli embedding.
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # solo il nuovo token

        # Propaga l2_eviction e token_budget se passati come kwargs a model.generate()
        # HuggingFace genera li mette in model_kwargs e li passa qui ad ogni step.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "l2_eviction": kwargs.get("l2_eviction"),
            "token_budget": kwargs.get("token_budget"),
        }
