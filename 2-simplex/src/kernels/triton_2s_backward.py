"""
Triton backward kernels for 2-simplicial attention.
Layout: [batch, seq_len, head_dim, num_heads]
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
    # KV1 Kernel: Computes dK1, dV1, and partial dQ
    # ---------------------------------------------------------------------------
    @triton.jit
    def two_simplicial_attn_bwd_kv1_kernel(
        Q_ptr, K1_ptr, K2_ptr, V1_ptr, V2_ptr, dO_ptr, M_ptr, D_ptr,
        dQ_ptr, dK1_ptr, dV1_ptr,
        bs, seq_len, num_heads, head_dim,
        w1, w2,
        q_stride_b, q_stride_s, q_stride_k, q_stride_h,
        k1_stride_b, k1_stride_s, k1_stride_k, k1_stride_h,
        k2_stride_b, k2_stride_s, k2_stride_k, k2_stride_h,
        v1_stride_b, v1_stride_s, v1_stride_k, v1_stride_h,
        v2_stride_b, v2_stride_s, v2_stride_k, v2_stride_h,
        dO_stride_b, dO_stride_s, dO_stride_k, dO_stride_h,
        m_stride_b, m_stride_k, m_stride_s,
        d_stride_b, d_stride_k, d_stride_s,
        dq_stride_b, dq_stride_s, dq_stride_k, dq_stride_h,
        dk1_stride_b, dk1_stride_s, dk1_stride_k, dk1_stride_h,
        dv1_stride_b, dv1_stride_s, dv1_stride_k, dv1_stride_h,
        BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr, SM_SCALE: tl.constexpr,
        K2_BIAS: tl.constexpr, V2_BIAS: tl.constexpr,
        COMPUTE_DQ: tl.constexpr, num_stages: tl.constexpr,
        is_flipped: tl.constexpr,
    ):
        data_dtype = tl.bfloat16
        compute_dtype = tl.float32
        gemm_dtype = tl.bfloat16

        kv1_start = tl.program_id(0) * BLOCK_SIZE_KV
        kv1_end = kv1_start + BLOCK_SIZE_KV
        bk = tl.program_id(1)
        offs_b = bk // num_heads
        offs_k = bk % num_heads

        qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
        Q_ptr += qkv_offs_bk
        K1_ptr += qkv_offs_bk
        K2_ptr += qkv_offs_bk
        V1_ptr += qkv_offs_bk
        V2_ptr += qkv_offs_bk

        dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
        M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
        D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
        dK1_ptr += offs_b * dk1_stride_b + offs_k * dk1_stride_k
        dV1_ptr += offs_b * dv1_stride_b + offs_k * dv1_stride_k
        if COMPUTE_DQ:
            dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k

        softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
        qkv_offs_h = tl.arange(0, HEAD_DIM)
        qkv_mask_h = qkv_offs_h < head_dim

        kv1_offs_s = kv1_start + tl.arange(0, BLOCK_SIZE_KV)
        k1_offs = kv1_offs_s[:, None] * k1_stride_s + qkv_offs_h[None, :] * k1_stride_h
        kv1_mask_s = kv1_offs_s < seq_len
        kv1_mask = kv1_mask_s[:, None] & qkv_mask_h[None, :]
        k1_tile = tl.load(K1_ptr + k1_offs, mask=kv1_mask).to(compute_dtype)
        v1_offs = kv1_offs_s[:, None] * v1_stride_s + qkv_offs_h[None, :] * v1_stride_h
        v1_tile = tl.load(V1_ptr + v1_offs, mask=kv1_mask).to(compute_dtype)

        if is_flipped:
            k1_tile += K2_BIAS
            v1_tile += V2_BIAS

        dv1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
        dk1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)

        for kv2_idx in tl.range(tl.maximum(0, kv1_start - w2), tl.minimum(seq_len, kv1_end + w1)):
            k2_offs = kv2_idx * k2_stride_s + qkv_offs_h * k2_stride_h
            k2_tile = (tl.load(K2_ptr + k2_offs, mask=qkv_mask_h).to(compute_dtype))[None, :]
            v2_offs = kv2_idx * v2_stride_s + qkv_offs_h * v2_stride_h
            v2_tile = (tl.load(V2_ptr + v2_offs, mask=qkv_mask_h).to(compute_dtype))[None, :]
            if not is_flipped:
                k2_tile += K2_BIAS
                v2_tile += V2_BIAS

            k1k2 = k1_tile * k2_tile
            v1v2 = v1_tile * v2_tile
            k1k2 = k1k2.to(gemm_dtype)
            v1v2 = v1v2.to(gemm_dtype)

            q_start = tl.maximum(kv1_start, kv2_idx)
            q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2))

            for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q):
                q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)
                q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None] * q_stride_h
                q_mask_s = q_offs_s < seq_len
                qt_mask = q_mask_s[None, :] & qkv_mask_h[:, None]
                qt_tile = tl.load(Q_ptr + q_offs, mask=qt_mask).to(gemm_dtype)

                m_offs = q_offs_s * m_stride_s
                m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype)[None, :]
                d_offs = q_offs_s * d_stride_s
                d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype)[None, :]
                dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
                dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask_s[:, None] & qkv_mask_h[None, :]).to(compute_dtype)

                if COMPUTE_DQ:
                    dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)

                qkkT = tl.dot(k1k2, qt_tile * softmax_scale, out_dtype=tl.float32)
                kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (kv1_offs_s[:, None] <= q_offs_s[None, :])
                kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx <= q_offs_s)
                local_mask = (kv1_local_mask & kv2_local_mask[None, :])
                qkkT = tl.where(local_mask, qkkT, -1.0e38)

                pT = tl.exp(qkkT - m_tile)
                pT = tl.where(local_mask, pT, 0.0)
                dOv2 = dO_tile * v2_tile
                dv1 += tl.dot(pT.to(gemm_dtype), dOv2.to(gemm_dtype), out_dtype=tl.float32)

                dpT = tl.dot(v1v2, tl.trans(dO_tile.to(gemm_dtype)), out_dtype=tl.float32)
                dsT = pT * (dpT - d_tile)
                dsT = tl.where(local_mask, dsT, 0.0)
                dsT = dsT.to(gemm_dtype)

                dk1 += (tl.dot(dsT, tl.trans(qt_tile), out_dtype=tl.float32) * k2_tile.to(tl.float32) * softmax_scale)

                if COMPUTE_DQ:
                    dq += (tl.dot(tl.trans(dsT), k1k2, out_dtype=tl.float32) * softmax_scale)
                    dq_offs = q_offs_s[:, None] * dq_stride_s + qkv_offs_h[None, :] * dq_stride_h
                    tl.atomic_add(dQ_ptr + dq_offs, dq, mask=q_mask_s[:, None] & qkv_mask_h[None, :])

        dv1_offs = kv1_offs_s[:, None] * dv1_stride_s + qkv_offs_h[None, :] * dv1_stride_h
        dk1_offs = kv1_offs_s[:, None] * dk1_stride_s + qkv_offs_h[None, :] * dk1_stride_h
        tl.store(dV1_ptr + dv1_offs, dv1.to(data_dtype), mask=kv1_mask)
        tl.store(dK1_ptr + dk1_offs, dk1.to(data_dtype), mask=kv1_mask)

    # ---------------------------------------------------------------------------
    # KV2Q Kernel: Computes dK2, dV2, and final dQ
    # ---------------------------------------------------------------------------
    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_KV2": 64, "num_stages": 1}, num_warps=4)],
        key=["HEAD_DIM"],
    )
    @triton.jit
    def two_simplicial_attn_bwd_kv2q_kernel(
        Q_ptr, K1_ptr, K2_ptr, V1_ptr, V2_ptr, dO_ptr, M_ptr, D_ptr,
        dQ_ptr, dK2_ptr, dV2_ptr,
        bs, seq_len, num_heads, head_dim,
        w1, w2,
        q_stride_b, q_stride_s, q_stride_k, q_stride_h,
        k1_stride_b, k1_stride_s, k1_stride_k, k1_stride_h,
        k2_stride_b, k2_stride_s, k2_stride_k, k2_stride_h,
        v1_stride_b, v1_stride_s, v1_stride_k, v1_stride_h,
        v2_stride_b, v2_stride_s, v2_stride_k, v2_stride_h,
        dO_stride_b, dO_stride_s, dO_stride_k, dO_stride_h,
        m_stride_b, m_stride_k, m_stride_s,
        d_stride_b, d_stride_k, d_stride_s,
        dq_stride_b, dq_stride_s, dq_stride_k, dq_stride_h,
        dk2_stride_b, dk2_stride_s, dk2_stride_k, dk2_stride_h,
        dv2_stride_b, dv2_stride_s, dv2_stride_k, dv2_stride_h,
        BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV2: tl.constexpr,
        HEAD_DIM: tl.constexpr, SM_SCALE: tl.constexpr,
        K2_BIAS: tl.constexpr, V2_BIAS: tl.constexpr,
        num_stages: tl.constexpr, IS_SECOND_PASS: tl.constexpr,
    ):
        assert BLOCK_SIZE_KV2 == BLOCK_SIZE_Q + w2
        data_dtype = tl.bfloat16
        compute_dtype = tl.float32
        gemm_dtype = tl.bfloat16

        q_start = tl.program_id(0) * BLOCK_SIZE_KV2
        if IS_SECOND_PASS:
            q_start += BLOCK_SIZE_Q
        q_end = q_start + BLOCK_SIZE_Q
        kv2_start = q_start - w2

        bk = tl.program_id(1)
        offs_b = bk // num_heads
        offs_k = bk % num_heads

        qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
        Q_ptr += qkv_offs_bk
        K1_ptr += qkv_offs_bk
        K2_ptr += qkv_offs_bk
        V1_ptr += qkv_offs_bk
        V2_ptr += qkv_offs_bk

        dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
        M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
        D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
        dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k
        dK2_ptr += offs_b * dk2_stride_b + offs_k * dk2_stride_k
        dV2_ptr += offs_b * dv2_stride_b + offs_k * dv2_stride_k

        softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
        qkv_offs_h = tl.arange(0, HEAD_DIM)
        qkv_mask_h = qkv_offs_h < head_dim

        q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
        kv2_offs_s = kv2_start + tl.arange(0, BLOCK_SIZE_KV2)
        q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
        kv2_offs = kv2_offs_s[:, None] * k2_stride_s + qkv_offs_h[None, :] * k2_stride_h
        m_offs = q_offs_s * m_stride_s
        d_offs = q_offs_s * d_stride_s
        dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
        q_mask_s = q_offs_s < seq_len
        q_mask = q_mask_s[:, None] & qkv_mask_h[None, :]
        kv2_mask_s = (0 <= kv2_offs_s) & (kv2_offs_s < seq_len)
        kv2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]

        q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(compute_dtype)
        k2_tile = tl.load(K2_ptr + kv2_offs, mask=kv2_mask).to(gemm_dtype)
        v2_tile = tl.load(V2_ptr + kv2_offs, mask=kv2_mask).to(gemm_dtype)
        m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype)
        d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype)
        dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask).to(gemm_dtype)

        k2_tile += K2_BIAS
        v2_tile += V2_BIAS
        k2_tile = k2_tile.to(gemm_dtype)
        v2_tile = v2_tile.to(gemm_dtype)

        dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)
        dk2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)
        dv2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)

        kv1_start = tl.maximum(0, q_start - w1)
        kv1_end = tl.minimum(seq_len, q_end)
        for kv1_idx in tl.range(kv1_start, kv1_end, num_stages=num_stages):
            k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
            v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
            k1_tile = tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(compute_dtype)
            v1_tile = tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(compute_dtype)

            qk1_s = q_tile * (k1_tile[None, :] * softmax_scale)
            qk1_s = qk1_s.to(gemm_dtype)
            qkkT = tl.dot(k2_tile, qk1_s.T, out_dtype=tl.float32)

            qkT_mask = kv2_mask_s[:, None] & q_mask_s[None, :]
            kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_idx) & (kv1_idx <= q_offs_s[None, :])
            kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (kv2_offs_s[:, None] <= q_offs_s[None, :])
            local_mask = (kv1_local_mask & kv2_local_mask)
            qkT_mask &= local_mask

            pT = tl.exp(qkkT - m_tile[None, :])
            pT = tl.where(qkT_mask, pT, 0.0)

            dOv1 = (dO_tile * v1_tile[None, :]).to(gemm_dtype)
            dv2 += tl.dot(pT.to(gemm_dtype), dOv1, out_dtype=tl.float32)

            dpT = tl.dot(v2_tile, dOv1.T, out_dtype=tl.float32)
            dsT = pT * (dpT - d_tile[None, :])
            dsT = tl.where(qkT_mask, dsT, 0.0).to(gemm_dtype)

            dk2 += tl.dot(dsT, qk1_s, out_dtype=tl.float32)
            k1k2 = (k1_tile[None, :] * k2_tile).to(gemm_dtype)
            dq += tl.dot(dsT.T, k1k2)

        if IS_SECOND_PASS:
            prev_dk2 = tl.load(dK2_ptr + kv2_offs, kv2_mask)
            prev_dv2 = tl.load(dV2_ptr + kv2_offs, kv2_mask)
            dk2 += prev_dk2
            dv2 += prev_dv2

        dq *= softmax_scale
        tl.store(dK2_ptr + kv2_offs, dk2.to(data_dtype), kv2_mask)
        tl.store(dV2_ptr + kv2_offs, dv2.to(data_dtype), kv2_mask)
        tl.store(dQ_ptr + q_offs, dq.to(data_dtype), q_mask)

def backward(grad_output, tri_feats, edge_index, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1=8, w2=8):
    """Backward pass entry point."""
    if not _check_triton():
        raise RuntimeError("Triton not available.")

    N = Q.size(0)
    H = num_heads
    D = head_dim

    # Reshape all to [1, N, D, H]
    Q_r = Q.view(1, N, D, H).contiguous()
    K_r = K.view(1, N, D, H).contiguous()
    V_r = V.view(1, N, D, H).contiguous()
    Kp_r = Kp.view(1, N, D, H).contiguous()
    Vp_r = Vp.view(1, N, D, H).contiguous()
    dO_r = grad_output.view(1, N, D, H).contiguous()

    # Re-fetch M (saved from forward) and compute D
    # For now, we assume M is passed in. But in actual autograd, we'd have it.
    # We also need a rowsum(dO * O) helper.
    # Let's assume we have them or compute them here for the MVP.
    # O is required to compute D.
    from . import triton_2s_forward
    O_r, M_r = triton_2s_forward.forward(tri_feats, edge_index, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1, w2)
    O_r = O_r.view(1, N, D, H)
    M_r = M_r.view(1, H, N)

    D_row = torch.sum(dO_r * O_r, dim=2).contiguous()  # [1, H, N] — rowsum(dO * O)

    dQ = torch.zeros_like(Q_r)
    dK1 = torch.zeros_like(K_r)
    dV1 = torch.zeros_like(V_r)
    dK2 = torch.zeros_like(Kp_r)
    dV2 = torch.zeros_like(Vp_r)

    # Launch KV1
    grid_kv1 = (triton.cdiv(N, 32), 1 * H)
    strides = {
        "q_stride_b": Q_r.stride(0), "q_stride_s": Q_r.stride(1), "q_stride_k": Q_r.stride(2), "q_stride_h": Q_r.stride(3),
        "k1_stride_b": K_r.stride(0), "k1_stride_s": K_r.stride(1), "k1_stride_k": K_r.stride(2), "k1_stride_h": K_r.stride(3),
        "k2_stride_b": Kp_r.stride(0), "k2_stride_s": Kp_r.stride(1), "k2_stride_k": Kp_r.stride(2), "k2_stride_h": Kp_r.stride(3),
        "v1_stride_b": V_r.stride(0), "v1_stride_s": V_r.stride(1), "v1_stride_k": V_r.stride(2), "v1_stride_h": V_r.stride(3),
        "v2_stride_b": Vp_r.stride(0), "v2_stride_s": Vp_r.stride(1), "v2_stride_k": Vp_r.stride(2), "v2_stride_h": Vp_r.stride(3),
        "dO_stride_b": dO_r.stride(0), "dO_stride_s": dO_r.stride(1), "dO_stride_k": dO_r.stride(2), "dO_stride_h": dO_r.stride(3),
        "m_stride_b": 0, "m_stride_k": M_r.stride(1), "m_stride_s": M_r.stride(2),
        "d_stride_b": 0, "d_stride_k": D_row.stride(1), "d_stride_s": D_row.stride(2),
        "dq_stride_b": dQ.stride(0), "dq_stride_s": dQ.stride(1), "dq_stride_k": dQ.stride(2), "dq_stride_h": dQ.stride(3),
        "dk1_stride_b": dK1.stride(0), "dk1_stride_s": dK1.stride(1), "dk1_stride_k": dK1.stride(2), "dk1_stride_h": dK1.stride(3),
        "dv1_stride_b": dV1.stride(0), "dv1_stride_s": dV1.stride(1), "dv1_stride_k": dV1.stride(2), "dv1_stride_h": dV1.stride(3),
    }

    two_simplicial_attn_bwd_kv1_kernel[grid_kv1](
        Q_r, K_r, Kp_r, V_r, Vp_r, dO_r, M_r, D_row,
        dQ, dK1, dV1,
        1, N, H, head_dim,
        w1, w2,
        **strides,
        BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=32, HEAD_DIM=head_dim, SM_SCALE=1.0 / (head_dim ** 0.5),
        K2_BIAS=0.0, V2_BIAS=0.0, COMPUTE_DQ=True, num_stages=1, is_flipped=False
    )

    # Launch KV2Q (Pass 1 & 2)
    grid_kv2q = (triton.cdiv(N, 64), 1 * H)
    strides_kv2 = strides.copy()
    strides_kv2["dk2_stride_b"] = dK2.stride(0); strides_kv2["dk2_stride_s"] = dK2.stride(1); strides_kv2["dk2_stride_k"] = dK2.stride(2); strides_kv2["dk2_stride_h"] = dK2.stride(3)
    strides_kv2["dv2_stride_b"] = dV2.stride(0); strides_kv2["dv2_stride_s"] = dV2.stride(1); strides_kv2["dv2_stride_k"] = dV2.stride(2); strides_kv2["dv2_stride_h"] = dV2.stride(3)

    two_simplicial_attn_bwd_kv2q_kernel[grid_kv2q](
        Q_r, K_r, Kp_r, V_r, Vp_r, dO_r, M_r, D_row,
        dQ, dK2, dV2,
        1, N, H, head_dim,
        w1, w2,
        **strides_kv2,
        HEAD_DIM=head_dim, SM_SCALE=1.0 / (head_dim ** 0.5), K2_BIAS=0.0, V2_BIAS=0.0, num_stages=1, IS_SECOND_PASS=False
    )
    two_simplicial_attn_bwd_kv2q_kernel[grid_kv2q](
        Q_r, K_r, Kp_r, V_r, Vp_r, dO_r, M_r, D_row,
        dQ, dK2, dV2,
        1, N, H, head_dim,
        w1, w2,
        **strides_kv2,
        HEAD_DIM=head_dim, SM_SCALE=1.0 / (head_dim ** 0.5), K2_BIAS=0.0, V2_BIAS=0.0, num_stages=1, IS_SECOND_PASS=True
    )

    return dQ.view(N, H, head_dim), dK1.view(N, H, head_dim), dV1.view(N, H, head_dim), dK2.view(N, H, head_dim), dV2.view(N, H, head_dim)