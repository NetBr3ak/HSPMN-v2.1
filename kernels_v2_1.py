"""
Triton kernels for HSPMN v2.1.
Implements Sparse-Query Dense-Key (SQDK) Attention.
Optimized for NVIDIA H100/Blackwell architectures.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _sqdk_fwd_kernel(
    Q, K, V, Out,
    Q_indices,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im,
    sm_scale,
    Z, H, N_CTX, N_ACTIVE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Forward pass kernel for Sparse-Query Dense-Key Attention.
    
    Args:
        Q, K, V: Query, Key, Value tensors.
        Out: Output tensor.
        Q_indices: Indices of the active queries.
        stride_*: Strides for all tensors.
        sm_scale: Softmax scaling factor (1/sqrt(head_dim)).
        Z: Batch size.
        H: Number of heads.
        N_CTX: Context length (sequence length).
        N_ACTIVE: Number of active queries.
        BLOCK_M: Block size for queries (M dimension).
        BLOCK_N: Block size for keys/values (N dimension).
        HEAD_DIM: Dimension of each head.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Pointers to Q, K, V
    Q_ptr = Q + off_z * stride_qb + off_h * stride_qh
    K_ptr = K + off_z * stride_kb + off_h * stride_kh
    V_ptr = V + off_z * stride_vb + off_h * stride_vh
    Out_ptr = Out + off_z * stride_ob + off_h * stride_oh
    Idx_ptr = Q_indices + off_z * stride_ib
    
    # Block offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load Q indices for this block
    # Q_indices shape: [Batch, Active_Tokens]
    # We need to know the real position of the query to do causal masking
    q_real_pos = tl.load(Idx_ptr + offs_m * stride_im, mask=offs_m < N_ACTIVE, other=-1)
    
    # Load Q
    # Q shape: [Batch, Active_Tokens, Heads, Dim]
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_ACTIVE, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    
    # Loop over K, V blocks
    # We attend to ALL keys/values (Dense Key)
    # But we must respect causal mask: key_pos <= q_real_pos
    
    # Optimization: We can skip blocks that are entirely beyond the max q_real_pos in this Q block
    max_q_pos = tl.max(q_real_pos, 0)
    
    # We iterate through all K/V blocks up to max_q_pos
    # Since keys are dense, their positions are just 0, 1, 2...
    
    for start_n in range(0, max_q_pos + 1, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K
        k_ptrs = K_ptr + ( (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        
        # Compute attention scores: QK^T
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        # Causal Masking
        # key_pos = start_n + offs_n
        # mask: key_pos <= q_real_pos
        key_pos = start_n + offs_n[None, :]
        mask = key_pos <= q_real_pos[:, None]
        
        qk = tl.where(mask, qk, float('-inf'))
        
        # Softmax and Update
        m_ij = tl.max(qk, 1)
        
        # Fix for all-masked rows: replace -inf with 0 temporarily to avoid NaN in exp
        m_ij_safe = tl.where(m_ij == float('-inf'), 0.0, m_ij)
        p = tl.exp(qk - m_ij_safe[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Avoid NaN in alpha/beta when m_i_new is -inf
        m_i_new_safe = tl.where(m_i_new == float('-inf'), 0.0, m_i_new)
        
        alpha = tl.exp(m_i - m_i_new_safe)
        beta = tl.exp(m_ij - m_i_new_safe)
        
        l_i_new = alpha * l_i + beta * l_ij
        
        # Load V
        v_ptrs = V_ptr + ( (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        # Update accumulator
        p = p.to(tl.float16)
        v = v.to(tl.float16)
        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p, v)
        
        l_i = l_i_new
        m_i = m_i_new

    # Finalize
    acc = acc / l_i[:, None]
    
    # Store Output
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_ACTIVE)

@torch.compiler.disable
def sparse_query_dense_key_attention(q, k, v, q_indices, block_m=64, block_n=64):
    """
    Executes Sparse-Query Dense-Key Attention.

    Args:
        q: [Batch, N_Active, Heads, Dim]
        k: [Batch, Seq_Len, Heads, Dim]
        v: [Batch, Seq_Len, Heads, Dim]
        q_indices: [Batch, N_Active] - The real positions of the query tokens. MUST BE SORTED.
        block_m: Block size for queries.
        block_n: Block size for keys/values.
    """
    # Check constraints
    Lq, Lk = q.shape[1], k.shape[1]
    assert Lk == v.shape[1]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    
    HEAD_DIM = q.shape[-1]
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # Output
    o = torch.empty_like(q)
    
    grid = (triton.cdiv(q.shape[1], block_m), q.shape[0] * q.shape[2])
    
    _sqdk_fwd_kernel[grid](
        q, k, v, o,
        q_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q_indices.stride(0), q_indices.stride(1),
        sm_scale,
        q.shape[0], q.shape[2], k.shape[1], q.shape[1],
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HEAD_DIM,
        num_warps=4, num_stages=3
    )
    return o
