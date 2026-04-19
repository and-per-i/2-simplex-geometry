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
    def _forward_kernel_call(Q, K1, K2, V1, V2, O, M, bs, seq_len, num_heads, head_dim, w1, w2, *args, **kwargs):
        grid = (triton.cdiv(seq_len, 64), bs * num_heads)
        
        # If strides are passed as positional arguments (from tests)
        if len(args) > 0:
            two_simplicial_attn_fwd_kernel[grid](
                Q, K1, K2, V1, V2, O, M,
                bs, seq_len, num_heads, head_dim, w1, w2,
                *args,
                HEAD_DIM=head_dim,
                INPUT_PRECISION="tf32",
                SM_SCALE=1.0 / (head_dim ** 0.5),
                K2_BIAS=0.0, V2_BIAS=0.0,
            )
        else:
            # Called from forward() with a dictionary of strides
            actual_strides = kwargs.pop('strides', {})
            two_simplicial_attn_fwd_kernel[grid](
                Q, K1, K2, V1, V2, O, M,
                bs, seq_len, num_heads, head_dim, w1, w2,
                **actual_strides,
                **kwargs,
                HEAD_DIM=head_dim,
                INPUT_PRECISION="tf32",
                SM_SCALE=1.0 / (head_dim ** 0.5),
                K2_BIAS=0.0, V2_BIAS=0.0,
            )


def forward(x, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1=8, w2=8):
    if not _check_triton():
        raise RuntimeError("Triton is not available in this environment.")

    N = x.size(0)
    H = num_heads
    D = head_dim

    Q_r  = Q.view(1, N, H, D).contiguous()
    K_r  = K.view(1, N, H, D).contiguous()
    V_r  = V.view(1, N, H, D).contiguous()
    Kp_r = Kp.view(1, N, H, D).contiguous()
    Vp_r = Vp.view(1, N, H, D).contiguous()

    O = torch.empty((1, N, H, D), device=x.device, dtype=x.dtype)
    M = torch.empty((1, H, N),    device=x.device, dtype=x.dtype)

    q_stride_b, q_stride_s, q_stride_k, q_stride_h = Q_r.stride()
    strides = {
        "q_stride_b": q_stride_b, "q_stride_s": q_stride_s, "q_stride_k": q_stride_k, "q_stride_h": q_stride_h,
        "k1_stride_b": q_stride_b, "k1_stride_s": q_stride_s, "k1_stride_k": q_stride_k, "k1_stride_h": q_stride_h,
        "k2_stride_b": q_stride_b, "k2_stride_s": q_stride_s, "k2_stride_k": q_stride_k, "k2_stride_h": q_stride_h,
        "v1_stride_b": q_stride_b, "v1_stride_s": q_stride_s, "v1_stride_k": q_stride_k, "v1_stride_h": q_stride_h,
        "v2_stride_b": q_stride_b, "v2_stride_s": q_stride_s, "v2_stride_k": q_stride_k, "v2_stride_h": q_stride_h,
        "out_stride_b": q_stride_b, "out_stride_s": q_stride_s, "out_stride_k": q_stride_k, "out_stride_h": q_stride_h,
        "m_stride_b": 0, "m_stride_k": M.stride()[1], "m_stride_s": M.stride()[2],
    }

    _forward_kernel_call(Q_r, K_r, Kp_r, V_r, Vp_r, O, M, 1, N, H, D, w1=w1, w2=w2, strides=strides)
    return O.view(N, -1), M.view(N, H)