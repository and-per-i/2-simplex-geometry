"""
Utility to load student model checkpoints (Phase 1 or Phase 2 format).

Phase 1 (Simplex Distillery) stores weights with a nested ``model.*`` prefix
and wraps 2-Simplex attention as ``attention.simplex_attn.*``.

Phase 2 uses a flat layout (``layers.{i}.attention.*`` directly).

Use ``load_checkpoint()`` for the general case — it auto-detects the format.
Use ``load_from_phase1()`` explicitly for Phase 1 ``.pt`` files.

Key remapping rules (Phase 1 → Phase 2)
----------------------------------------
  model.embeddings.token_embeddings.*  →  token_embedding.*
  model.embeddings.position_embeddings.*  →  position_embedding.*
  model.final_ln.*  →  ln_f.*
  model.layers.{i}.attention.simplex_attn.*  →  layers.{i}.attention.*
  model.layers.{i}.*  →  layers.{i}.*   (standard layers, strip prefix only)
  lm_head.*  →  lm_head.*               (unchanged)
"""

import re
from pathlib import Path
from typing import List, Optional

import torch

from .student_config import StudentConfig
from .student_model import StudentForCausalLM

# Default Phase 1 architecture (simplex-geometry 8L after Phase 4 extraction)
_PHASE1_DEFAULTS = dict(
    vocab_size=1024,
    hidden_size=384,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=1024,
    dropout_prob=0.1,
    use_simplex_attention=True,
    simplex_layers=[1, 5, 7],
    use_triton=True,
    w1=8,
    w2=8,
)


def remap_phase1_keys(state_dict: dict) -> dict:
    """Remap Phase 1 state_dict keys to Phase 2's flat key structure."""
    remapped = {}
    for k, v in state_dict.items():
        # 1. Strip top-level 'model.' wrapper added by StudentModel sub-module
        k = re.sub(r"^model\.", "", k)

        # 2. Embedding layer names differ
        k = k.replace("embeddings.token_embeddings.", "token_embedding.")
        k = k.replace("embeddings.position_embeddings.", "position_embedding.")

        # 3. Final LayerNorm name
        k = k.replace("final_ln.", "ln_f.")

        # 4. Simplicial attention: Phase 1 wraps TwoSimplicialAttention as
        #    `attention.simplex_attn.*`; Phase 2 uses `attention.*` directly.
        k = re.sub(
            r"(layers\.\d+\.attention)\.simplex_attn\.",
            r"\1.",
            k,
        )

        remapped[k] = v
    return remapped


def load_from_phase1(
    checkpoint_path: str,
    simplex_layers: Optional[List[int]] = None,
    device: str = "cpu",
    strict: bool = True,
) -> StudentForCausalLM:
    """Load a Phase 1 ``simplex-geometry_Final.pt`` into Phase 2's model.

    Args:
        checkpoint_path: Path to the ``.pt`` file (or a directory containing
                         ``pytorch_model.bin`` / ``model.safetensors`` +
                         ``config.json``).
        simplex_layers:  Override for which layer indices are simplicial.
                         Defaults to ``[1, 5, 7]`` (Phase 1 8L layout).
        device:          Target device for the loaded model.
        strict:          If True, ``load_state_dict`` will raise on key
                         mismatches.  Set False to debug partial loads.

    Returns:
        A ``StudentForCausalLM`` with Phase 1 weights, ready for inference
        or further fine-tuning.
    """
    path = Path(checkpoint_path)

    # --- Resolve config -------------------------------------------------------
    config_path = path / "config.json" if path.is_dir() else None
    if config_path and config_path.exists():
        config = StudentConfig.from_pretrained(str(path))
        # Ensure Phase 2-specific fields are present even if absent in old config
        if not hasattr(config, "simplex_layers") or not config.simplex_layers:
            config.simplex_layers = simplex_layers if simplex_layers is not None else [1, 5, 7]
        elif simplex_layers is not None:
            config.simplex_layers = simplex_layers
        config.use_simplex_attention = True
    else:
        # No config.json — use Phase 1 defaults
        kwargs = dict(_PHASE1_DEFAULTS)
        if simplex_layers is not None:
            kwargs["simplex_layers"] = simplex_layers
        config = StudentConfig(**kwargs)

    # --- Resolve state dict ---------------------------------------------------
    if path.is_dir():
        bin_path = path / "pytorch_model.bin"
        sf_path = path / "model.safetensors"
        if sf_path.exists():
            from safetensors.torch import load_file
            raw_sd = load_file(str(sf_path), device=device)
        elif bin_path.exists():
            raw_sd = torch.load(str(bin_path), map_location=device)
        else:
            raise FileNotFoundError(
                f"No pytorch_model.bin or model.safetensors found in {path}"
            )
    else:
        raw_sd = torch.load(str(path), map_location=device)
        # .pt files may store the dict under a 'model' key
        if isinstance(raw_sd, dict) and "model" in raw_sd and isinstance(raw_sd["model"], dict):
            raw_sd = raw_sd["model"]

    # --- Remap keys -----------------------------------------------------------
    remapped_sd = remap_phase1_keys(raw_sd)

    # --- Build model and load weights -----------------------------------------
    model = StudentForCausalLM(config)
    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)

    # Filter out is_bypassed buffers (all-False tensors added by progressive
    # pruning; they carry no learned info and are safe to ignore).
    real_missing = [k for k in missing if "is_bypassed" not in k and not k.endswith(".alpha")]
    real_unexpected = [k for k in unexpected if "is_bypassed" not in k]

    if real_missing:
        msg = f"Missing keys after Phase 1 → Phase 2 remapping:\n  " + "\n  ".join(real_missing)
        if strict:
            raise RuntimeError(msg)
        print(f"⚠️  [checkpoint_loader] {msg}")

    if real_unexpected:
        msg = f"Unexpected keys after Phase 1 → Phase 2 remapping:\n  " + "\n  ".join(real_unexpected)
        if strict:
            raise RuntimeError(msg)
        print(f"⚠️  [checkpoint_loader] {msg}")

    model = model.to(device)
    print(
        f"✅ Phase 1 checkpoint loaded: {sum(p.numel() for p in model.parameters()):,} params, "
        f"simplex_layers={config.simplex_layers}"
    )
    return model


# ---------------------------------------------------------------------------
# Auto-detect config from state dict
# ---------------------------------------------------------------------------

def _infer_config_from_state_dict(sd: dict) -> StudentConfig:
    """Build a StudentConfig by inspecting the state_dict shapes and keys."""
    layer_ids = sorted(set(
        int(m.group(1))
        for k in sd
        if (m := re.match(r"layers\.(\d+)\.", k))
    ))
    num_layers = max(layer_ids) + 1 if layer_ids else 6

    simplex_layers = [
        i for i in layer_ids
        if any(f"layers.{i}.attention.W_Q" in k for k in sd)
    ]

    emb = sd.get("token_embedding.weight")
    vocab_size = emb.shape[0] if emb is not None else 1024
    hidden_size = emb.shape[1] if emb is not None else 384

    fc1 = next((sd[k] for k in sd if "mlp.fc1.weight" in k), None)
    intermediate_size = fc1.shape[0] if fc1 is not None else 1536

    pos_emb = sd.get("position_embedding.weight")
    max_pos = pos_emb.shape[0] if pos_emb is not None else 1024

    # Infer num_heads from q_proj or W_Q shape — not directly available;
    # default to 8 (matches all known checkpoints).
    return StudentConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_pos,
        use_simplex_attention=bool(simplex_layers),
        simplex_layers=simplex_layers,
        use_triton=True,
        w1=8,
        w2=8,
    )


# ---------------------------------------------------------------------------
# Generic loader (auto-detects Phase 1 vs Phase 2 format)
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> StudentForCausalLM:
    """Load any student checkpoint — Phase 1 or Phase 2 format.

    Automatically detects the format from the state_dict keys:
    - If keys start with ``model.`` → Phase 1 format (apply remapping).
    - Otherwise → Phase 2 flat format (load directly).

    In both cases:
    - ``is_bypassed`` buffers are silently ignored.
    - Legacy ``alpha`` learnable scalars (present in old Phase 2 checkpoints)
      are stripped before loading.

    The config is reconstructed from the checkpoint's shapes when no
    ``config.json`` is found alongside the file.
    """
    path = Path(checkpoint_path)

    # --- Load raw state dict --------------------------------------------------
    if path.is_dir():
        sf_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if sf_path.exists():
            from safetensors.torch import load_file
            raw_sd = load_file(str(sf_path), device=device)
        elif bin_path.exists():
            raw_sd = torch.load(str(bin_path), map_location=device)
        else:
            raise FileNotFoundError(f"No weights found in {path}")
    else:
        raw_sd = torch.load(str(path), map_location=device, weights_only=True)
        if isinstance(raw_sd, dict) and "model" in raw_sd and isinstance(raw_sd["model"], dict):
            raw_sd = raw_sd["model"]

    # --- Detect format --------------------------------------------------------
    is_phase1 = any(k.startswith("model.") for k in raw_sd)
    sd = remap_phase1_keys(raw_sd) if is_phase1 else dict(raw_sd)

    # --- Strip obsolete keys --------------------------------------------------
    # 'alpha' was a learnable scalar present in old Phase 2 checkpoints; it was
    # removed from TwoSimplicialAttention. Drop it silently.
    sd = {k: v for k, v in sd.items() if not k.endswith(".alpha")}

    # --- Resolve config -------------------------------------------------------
    config_json = path / "config.json" if path.is_dir() else None
    if config_json and config_json.exists():
        config = StudentConfig.from_pretrained(str(path))
        if not getattr(config, "simplex_layers", None):
            config.simplex_layers = [
                i for i in range(config.num_hidden_layers)
                if any(f"layers.{i}.attention.W_Q" in k for k in sd)
            ]
    else:
        config = _infer_config_from_state_dict(sd)

    # --- Build model and load -------------------------------------------------
    model = StudentForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    real_missing = [k for k in missing if "is_bypassed" not in k and not k.endswith(".alpha")]
    real_unexpected = [k for k in unexpected if "is_bypassed" not in k]

    if real_missing:
        msg = "Missing keys:\n  " + "\n  ".join(real_missing)
        if strict:
            raise RuntimeError(msg)
        print(f"⚠️  [checkpoint_loader] {msg}")

    if real_unexpected:
        msg = "Unexpected keys:\n  " + "\n  ".join(real_unexpected)
        if strict:
            raise RuntimeError(msg)
        print(f"⚠️  [checkpoint_loader] {msg}")

    model = model.to(device)
    fmt = "Phase 1" if is_phase1 else "Phase 2"
    print(
        f"✅ [{fmt}] checkpoint loaded: {sum(p.numel() for p in model.parameters()):,} params, "
        f"{config.num_hidden_layers}L, simplex_layers={config.simplex_layers}"
    )
    return model
