"""
HSPMN v2 Triton Kernels.

This module implements high-performance custom CUDA kernels using OpenAI Triton.
The primary kernel is 'Sparse-Query Dense-Key (SQDK) Attention', designed to
efficiently handle the hybrid sparsity pattern of HSPMN.

Key Features:
- **Ragged/Sparse Queries**: Handles variable numbers of active queries per batch item.
- **Dense Keys/Values**: Efficiently attends to full context without padding overhead.
- **Grouped Query Attention (GQA)**: Supports MQA/GQA for reduced memory bandwidth.
- **Online Softmax**: Numerically stable single-pass softmax (FlashAttention style).
- **Causal Masking**: Efficient on-chip masking for autoregressive modeling.

Author: Szymon Jędryczko
Date: December 2025
"""

import torch
import triton
import triton.language as tl

__all__ = ["sparse_query_dense_key_attention"]

@triton.jit
def _sparse_query_dense_key_fwd_kernel(
    Q, K, V, Out,
    PosQ, PosK,  # Position IDs for Causal Masking
    cu_seqlens_q, cu_seqlens_k,  # Cumulative sequence lengths (Ragged offsets)
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_GROUPS: tl.constexpr,
):
    """
    Triton kernel for Forward Pass of Sparse-Query Dense-Key Attention.
    
    Parallelization Strategy:
    - Grid Z: Batch dimension (Batch ID)
    - Grid Y: Head dimension (Head ID)
    - Grid X: Block M dimension (Chunk of Queries)
    """
    # -----------------------------------------------------------
    # 1. Grid & Indexing Setup
    # -----------------------------------------------------------
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_m_idx = tl.program_id(2)

    # Retrieve start/end indices for the current batch element
    # cu_seqlens_q/k are [Batch+1] tensors containing cumulative counts
    q_start = tl.load(cu_seqlens_q + batch_id)
    q_end = tl.load(cu_seqlens_q + batch_id + 1)
    k_start = tl.load(cu_seqlens_k + batch_id)
    k_end = tl.load(cu_seqlens_k + batch_id + 1)

    q_len = q_end - q_start
    k_len = k_end - k_start

    # Determine the global range of queries this block is responsible for
    start_m = block_m_idx * BLOCK_M
    
    # Early exit if this block is outside the valid query range for this batch
    if start_m >= q_len:
        return

    # -----------------------------------------------------------
    # 2. Load Queries (Sparse)
    # -----------------------------------------------------------
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Mask to prevent out-of-bounds access for the last block
    mask_m = offs_m < q_len
    
    # Pointer arithmetic for Q
    # Q shape: [Total_Active_Tokens, Num_Heads, Head_Dim]
    Q_ptr = Q + (q_start + offs_m[:, None]) * stride_qm + \
            head_id * stride_qh + \
            offs_d[None, :] * stride_qd
            
    q = tl.load(Q_ptr, mask=mask_m[:, None], other=0.0)
    
    # Load Position IDs for Causal Masking if enabled
    if IS_CAUSAL:
        PosQ_ptr = PosQ + (q_start + offs_m)
        pos_q = tl.load(PosQ_ptr, mask=mask_m, other=0) # [BLOCK_M]

    # -----------------------------------------------------------
    # 3. Initialize Accumulators (Online Softmax)
    # -----------------------------------------------------------
    # m_i: Running max for numerical stability
    # l_i: Running sum of exponentials (denominator)
    # acc: Running weighted sum of values (numerator)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Pre-scale queries by 1/sqrt(d)
    # Cast back to input dtype to ensure consistent precision in dot product
    q = (q * sm_scale).to(Q.dtype.element_ty)

    # Handle Grouped Query Attention (GQA)
    # Map current Query Head ID to the corresponding Key/Value Head ID
    kv_head_id = head_id // KV_GROUPS

    # -----------------------------------------------------------
    # 4. Main Loop over Keys/Values (Dense)
    # -----------------------------------------------------------
    # Iterate over the dense context in chunks of BLOCK_N
    for start_n in range(0, k_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_len
        
        # -- Load Keys --
        K_ptr = K + (k_start + offs_n[None, :]) * stride_kn + \
                kv_head_id * stride_kh + \
                offs_d[:, None] * stride_kd
        k_block = tl.load(K_ptr, mask=mask_n[None, :], other=0.0)
        
        # -- Compute Attention Scores (QK^T) --
        qk = tl.dot(q, k_block)
        
        # -- Apply Masks --
        # 1. Sequence Length Mask (for padding/boundaries)
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        
        # 2. Causal Mask (Autoregressive)
        if IS_CAUSAL:
            PosK_ptr = PosK + (k_start + offs_n)
            pos_k = tl.load(PosK_ptr, mask=mask_n, other=1000000) # [BLOCK_N]
            
            # Valid if Query Position >= Key Position
            causal_mask = pos_q[:, None] >= pos_k[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # -- Online Softmax Update --
        # Update running max
        m_i_new = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_i_new)
        
        # Compute rescaling factors
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        
        # Update running denominator
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        # -- Load Values --
        V_ptr = V + (k_start + offs_n[:, None]) * stride_vn + \
                kv_head_id * stride_vh + \
                offs_d[None, :] * stride_vd
        v_block = tl.load(V_ptr, mask=mask_n[:, None], other=0.0)
        
        # Update accumulator
        # Note: beta is cast to v_block.dtype to match precision
        acc = acc * alpha[:, None] + tl.dot(beta.to(v_block.dtype), v_block)
        
        # Update max for next iteration
        m_i = m_i_new

    # -----------------------------------------------------------
    # 5. Finalize and Store
    # -----------------------------------------------------------
    # Normalize by the denominator
    acc = acc / l_i[:, None]
    
    # Store result
    Out_ptr = Out + (q_start + offs_m[:, None]) * stride_om + \
              head_id * stride_oh + \
              offs_d[None, :] * stride_od
              
    tl.store(Out_ptr, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

def sparse_query_dense_key_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor, 
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, 
    max_seqlen_k: int, 
    sm_scale: float = 1.0,
    is_causal: bool = False,
    pos_ids_q: torch.Tensor = None,
    pos_ids_k: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes attention between sparse queries and dense keys/values.
    
    Args:
        q: [Total_Active, H_q, D] - Packed active query tokens.
        k: [Total_Dense, H_k, D] - Packed dense key tokens.
        v: [Total_Dense, H_k, D] - Packed dense value tokens.
        cu_seqlens_q: [Batch+1] - Cumulative sequence lengths for Q.
        cu_seqlens_k: [Batch+1] - Cumulative sequence lengths for K/V.
        max_seqlen_q: Maximum sequence length in Q (for grid sizing).
        max_seqlen_k: Maximum sequence length in K (unused in grid, but good for validation).
        sm_scale: Softmax scaling factor (usually 1/sqrt(d)).
        is_causal: Whether to apply causal masking.
        pos_ids_q: [Total_Active] - Position IDs for queries (required if causal).
        pos_ids_k: [Total_Dense] - Position IDs for keys (required if causal).
        
    Returns:
        output: [Total_Active, H_q, D] - Attention output.
    """
    # Input Validation
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Head dimensions must match"
    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128}, "Triton kernel supports head dims 16, 32, 64, 128"
    
    num_heads_q = q.shape[1]
    num_heads_k = k.shape[1]
    assert num_heads_q % num_heads_k == 0, "Query heads must be divisible by KV heads (GQA)"
    KV_GROUPS = num_heads_q // num_heads_k
    
    if is_causal:
        assert pos_ids_q is not None and pos_ids_k is not None, "Position IDs required for causal attention"
    
    num_batches = cu_seqlens_q.shape[0] - 1
    
    # Allocate Output
    output = torch.empty_like(q)
    
    # Kernel Configuration
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Launch Grid
    # (Batch, Heads, Blocks_M)
    grid = (num_batches, num_heads_q, triton.cdiv(max_seqlen_q, BLOCK_M))
    
    _sparse_query_dense_key_fwd_kernel[grid](
        q, k, v, output,
        pos_ids_q if pos_ids_q is not None else q, # Dummy ptr if not causal
        pos_ids_k if pos_ids_k is not None else k, # Dummy ptr if not causal
        cu_seqlens_q, cu_seqlens_k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        sm_scale,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        KV_GROUPS=KV_GROUPS
    )
    
    return output
