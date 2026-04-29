"""
Triton forward kernel for 2-simplicial attention (Forward pass).
Layout: [batch, seq_len, head_dim, num_heads]  (i.e. [b, s, k, h])
"""
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _check_triton():
    return triton is not None and tl is not None


if _check_triton():
    # ---------------------------------------------------------------------------
    # Forward kernel
    # ---------------------------------------------------------------------------
    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32, "num_stages": 1},
                num_warps=4,
            )
        ],
        key=["HEAD_DIM"],
    )
    @triton.jit
    def two_simplicial_attn_fwd_kernel(
        Q_ptr, K1_ptr, K2_ptr, V1_ptr, V2_ptr, O_ptr, M_ptr,
        bs, seq_len, num_heads, head_dim,
        w1: tl.constexpr, w2: tl.constexpr,
        q_stride_b, q_stride_s, q_stride_k, q_stride_h,
        k1_stride_b, k1_stride_s, k1_stride_k, k1_stride_h,
        k2_stride_b, k2_stride_s, k2_stride_k, k2_stride_h,
        v1_stride_b, v1_stride_s, v1_stride_k, v1_stride_h,
        v2_stride_b, v2_stride_s, v2_stride_k, v2_stride_h,
        out_stride_b, out_stride_s, out_stride_k, out_stride_h,
        m_stride_b, m_stride_k, m_stride_s,
        BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr, INPUT_PRECISION: tl.constexpr,
        SM_SCALE: tl.constexpr,
        K2_BIAS: tl.constexpr, V2_BIAS: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        data_dtype = tl.bfloat16
        compute_dtype = tl.float32
        gemm_dtype = tl.bfloat16

        q_start = tl.program_id(0) * BLOCK_SIZE_Q
        q_end = q_start + BLOCK_SIZE_Q
        bk = tl.program_id(1)
        offs_b = bk // num_heads
        offs_k = bk % num_heads

        qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k

        Q_ptr += qkv_offs_bk
        K1_ptr += qkv_offs_bk
        K2_ptr += qkv_offs_bk
        V1_ptr += qkv_offs_bk
        V2_ptr += qkv_offs_bk
        O_ptr += qkv_offs_bk
        M_ptr += offs_b * m_stride_b + offs_k * m_stride_k

        m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype) - float("inf")
        l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype)
        acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=compute_dtype)

        q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
        qkv_offs_h = tl.arange(0, HEAD_DIM)
        q_mask_s = q_offs_s < seq_len
        qkv_mask_h = qkv_offs_h < head_dim
        q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
        q_mask = q_mask_s[:, None] & qkv_mask_h[None, :]

        q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(compute_dtype)  # [BLOCK_SIZE_Q, HEAD_DIM]
        softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

        for kv1_idx in tl.range(tl.maximum(0, q_start - w1), tl.minimum(seq_len, q_end)):
            k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
            k1_tile = (tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(compute_dtype))[None, :]  # [1, HEAD_DIM]
            qk1 = q_tile * k1_tile  # [BLOCK_SIZE_Q, HEAD_DIM]
            qk1 = qk1.to(gemm_dtype)

            v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
            v1_tile = (tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(compute_dtype))[None, :]  # [1, HEAD_DIM]

            for kv2_idx in tl.range(
                tl.maximum(0, q_start - w2),
                tl.minimum(seq_len, q_end),
                BLOCK_SIZE_KV,
                num_stages=num_stages,
            ):
                kv2_offs_s = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
                kv2_mask_s = kv2_offs_s < seq_len
                k2t_mask = kv2_mask_s[None, :] & qkv_mask_h[:, None]
                v2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]
                k2_offs = kv2_offs_s[None, :] * k2_stride_s + qkv_offs_h[:, None] * k2_stride_h
                v2_offs = kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h
                k2t_tile = tl.load(K2_ptr + k2_offs, mask=k2t_mask).to(compute_dtype)  # [HEAD_DIM, BLOCK_SIZE_KV]
                v2_tile = tl.load(V2_ptr + v2_offs, mask=v2_mask).to(compute_dtype)    # [BLOCK_SIZE_KV, HEAD_DIM]
                k2t_tile += K2_BIAS
                v2_tile += V2_BIAS
                k2t_tile = k2t_tile.to(gemm_dtype)
                v2_tile = v2_tile.to(compute_dtype)

                qk = tl.dot(
                    qk1 * softmax_scale,
                    k2t_tile,
                    input_precision=INPUT_PRECISION,
                    out_dtype=tl.float32,
                )  # [BLOCK_SIZE_Q, BLOCK_SIZE_KV]

                qk_mask = q_mask_s[:, None] & kv2_mask_s[None, :]
                # Mask for q_idx - w1 < kv1_idx <= q_idx
                # and q_idx - w2 < kv2_offs_s <= q_idx
                kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (
                    kv1_idx <= q_offs_s[:, None]
                )
                kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (
                    kv2_offs_s[None, :] <= q_offs_s[:, None]
                )
                qk_mask &= kv1_local_mask & kv2_local_mask
                qk += tl.where(qk_mask, 0, -1.0e38)

                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.math.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]

                v12_tile = v1_tile * v2_tile  # [BLOCK_SIZE_KV, HEAD_DIM]
                acc += tl.dot(
                    p.to(gemm_dtype),
                    v12_tile.to(gemm_dtype),
                    input_precision=INPUT_PRECISION,
                    out_dtype=tl.float32,
                )

                m_i = m_ij

        # Normalize AFTER both loops — dividing inside kv1 gives wrong results
        acc = acc / l_i[:, None]
        acc = tl.where(q_mask, acc, 0.0)
        acc = acc.to(data_dtype)
        out_offs = q_offs_s[:, None] * out_stride_s + qkv_offs_h[None, :] * out_stride_h
        tl.store(O_ptr + out_offs, acc, mask=q_mask)

        m = m_i + tl.log(l_i)
        m_offs = q_offs_s * m_stride_s
        m_mask = q_offs_s < seq_len
        tl.store(M_ptr + m_offs, m, mask=m_mask)

    # ---------------------------------------------------------------------------
    # Internal launcher
    # ---------------------------------------------------------------------------
    def _forward_kernel_call(Q, K1, K2, V1, V2, O, M, bs, seq_len, num_heads, head_dim, w1, w2, strides):
        grid = (triton.cdiv(seq_len, 64), bs * num_heads)
        
        two_simplicial_attn_fwd_kernel[grid](
            Q, K1, K2, V1, V2, O, M,
            bs, seq_len, num_heads, head_dim, w1, w2,
            **strides,
            HEAD_DIM=head_dim,
            INPUT_PRECISION="tf32",
            SM_SCALE=1.0 / (head_dim ** 0.5),
            K2_BIAS=0.0, V2_BIAS=0.0,
        )


def forward(x, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1=8, w2=8):
    if not _check_triton():
        raise RuntimeError("Triton is not available in this environment.")

    # Validation
    for t in [Q, K, V, Kp, Vp]:
        if not t.is_cuda:
            raise ValueError("All input tensors must be on CUDA for Triton kernels.")
        if t.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            raise ValueError(f"Unsupported dtype {t.dtype} for Triton kernel.")

    # Handle both (N, H, D) and (B, S, H, D) inputs

    if Q.dim() == 3:
        # N, H, D
        N = Q.size(0)
        bs = 1
        seq_len = N
        Q_r  = Q.view(bs, seq_len, num_heads, head_dim).contiguous()
        K_r  = K.view(bs, seq_len, num_heads, head_dim).contiguous()
        V_r  = V.view(bs, seq_len, num_heads, head_dim).contiguous()
        Kp_r = Kp.view(bs, seq_len, num_heads, head_dim).contiguous()
        Vp_r = Vp.view(bs, seq_len, num_heads, head_dim).contiguous()
    else:
        # B, S, H, D
        bs, seq_len, _, _ = Q.shape
        Q_r, K_r, V_r, Kp_r, Vp_r = Q.contiguous(), K.contiguous(), V.contiguous(), Kp.contiguous(), Vp.contiguous()

    O = torch.empty((bs, seq_len, num_heads, head_dim), device=x.device, dtype=x.dtype)
    M = torch.empty((bs, num_heads, seq_len),    device=x.device, dtype=torch.float32)

    q_s = Q_r.stride()
    k1_s = K_r.stride()
    k2_s = Kp_r.stride()
    v1_s = V_r.stride()
    v2_s = Vp_r.stride()
    o_s = O.stride()
    m_s = M.stride()

    strides = {
        "q_stride_b": q_s[0], "q_stride_s": q_s[1], "q_stride_k": q_s[2], "q_stride_h": q_s[3],
        "k1_stride_b": k1_s[0], "k1_stride_s": k1_s[1], "k1_stride_k": k1_s[2], "k1_stride_h": k1_s[3],
        "k2_stride_b": k2_s[0], "k2_stride_s": k2_s[1], "k2_stride_k": k2_s[2], "k2_stride_h": k2_s[3],
        "v1_stride_b": v1_s[0], "v1_stride_s": v1_s[1], "v1_stride_k": v1_s[2], "v1_stride_h": v1_s[3],
        "v2_stride_b": v2_s[0], "v2_stride_s": v2_s[1], "v2_stride_k": v2_s[2], "v2_stride_h": v2_s[3],
        "out_stride_b": o_s[0], "out_stride_s": o_s[1], "out_stride_k": o_s[2], "out_stride_h": o_s[3],
        "m_stride_b": m_s[0], "m_stride_k": m_s[1], "m_stride_s": m_s[2],
    }

    _forward_kernel_call(Q_r, K_r, Kp_r, V_r, Vp_r, O, M, bs, seq_len, num_heads, head_dim, w1=w1, w2=w2, strides=strides)
    
    if Q.dim() == 3:
        return O.view(seq_len, -1), M.view(num_heads, seq_len)
    return O, M