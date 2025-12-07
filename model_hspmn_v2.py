"""
HSPMN v2: Hierarchical Shallow Predictive Matter Networks.

This module implements the HSPMN architecture, a hybrid model that combines
a lightweight "Reflexive Stream" (Dense MLP) with a sparse, topology-aware "Contextual Stream" (Sparse Attention).
It utilizes a Target-Sparsity Regularized Router to dynamically select tokens for deep processing,
optimizing computational efficiency without sacrificing representational power.

Key Components:
- SparsityRouter: Determines token importance with sparsity regularization.
- HSPMNBlock: The core building block combining Reflexive and Contextual streams.
- Native SDPA: Optimized variable-length attention for modern GPUs.

Author: Szymon Jędryczko
Date: December 2025
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["HSPMNv2Block", "SparsityRouter", "HSPMNv2Config"]

# -----------------------------------------------------------------------------
# Optional Dependencies
# -----------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from triton_kernels import sparse_query_dense_key_attention
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton kernel not found. Falling back to slower Python implementation.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class HSPMNv2Config:
    """Configuration for HSPMN v2 Block."""
    dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    qkv_bias: bool = False
    target_sparsity: float = 0.2
    mlp_ratio: int = 4
    rotary_base: int = 10000

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by num_heads {self.num_heads}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}")
        
        # Triton Kernel Safety Check
        head_dim = self.dim // self.num_heads
        if TRITON_AVAILABLE and head_dim not in [16, 32, 64, 128]:
             logger.warning(f"Head dim {head_dim} is not a power of 2 (16, 32, 64, 128). Triton kernel might fail or be suboptimal.")

# -----------------------------------------------------------------------------
# Helper Modules
# -----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                        position_ids_q: torch.Tensor, position_ids_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to query and key tensors."""
    cos_q = cos[position_ids_q].unsqueeze(1)
    sin_q = sin[position_ids_q].unsqueeze(1)
    cos_k = cos[position_ids_k].unsqueeze(1)
    sin_k = sin[position_ids_k].unsqueeze(1)
    
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

# -----------------------------------------------------------------------------
# Core Modules
# -----------------------------------------------------------------------------
class SparsityRouter(nn.Module):
    """
    Predicts the 'importance' of each token to decide its processing path.
    Uses Target-Sparsity Regularization to prevent router collapse.
    """
    def __init__(self, dim: int, target_sparsity: float = 0.2):
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        self.target_sparsity = target_sparsity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [Batch, Seq, Dim]
        Returns:
            logits: Raw output from the gate.
            probs: Sigmoid probabilities.
            load_balancing_loss: Auxiliary loss for sparsity regularization.
        """
        logits = self.gate(x)
        probs = torch.sigmoid(logits)
        
        # Aux Loss: Load Balancing (Target-Sparsity Regularization)
        # Enforces average activation to match target_sparsity.
        mean_activation = probs.mean()
        load_balancing_loss = (mean_activation - self.target_sparsity) ** 2
        
        return logits, probs, load_balancing_loss

class HSPMNv2Block(nn.Module):
    """
    A hybrid processing block implementing the HSPMN v2 architecture.
    
    Structure:
    1. Router: Decides which tokens need attention (Target-Sparsity Regularized).
    2. Reflexive Stream: A lightweight MLP applied to ALL tokens.
    3. Contextual Stream: SQDK Attention applied ONLY to selected tokens.
    """
    def __init__(self, config: HSPMNv2Config):
        super().__init__()
        self.config = config
        self.head_dim = config.dim // config.num_heads
        
        self.router = SparsityRouter(config.dim, config.target_sparsity)
        
        # Contextual Stream: Attention Projections (SQDK)
        self.W_q = nn.Linear(config.dim, config.dim, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.dim, config.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.dim, config.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.W_o = nn.Linear(config.dim, config.dim, bias=config.qkv_bias)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=config.rotary_base)
        
        # Reflexive Stream (Dense MLP + Depthwise Conv) - The "blue collar" worker; never sleeps.
        # Includes a lightweight Depthwise Conv1d to prevent "Bag-of-Words" collapse for shallow tokens.
        mlp_hidden_dim = config.mlp_ratio * config.dim
        self.shallow_mixer = nn.Conv1d(config.dim, config.dim, kernel_size=3, padding=0, groups=config.dim)
        self.shallow_mlp = nn.Sequential(
            nn.Linear(config.dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, config.dim)
        )

    @torch.compile(mode="reduce-overhead")
    def _fused_router_forward(self, x: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused router logic using torch.compile to minimize Python overhead."""
        logits, probs, aux_loss = self.router(x)
        
        if training:
            # Gumbel-Sigmoid with Straight-Through Estimator
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            tau = 1.0 
            y_soft = torch.sigmoid((logits + g) / tau)
            y_hard = (y_soft > 0.5).float()
            mask_ste = (y_hard - y_soft.detach() + y_soft).to(x.dtype)
            active_mask = y_hard.squeeze(-1).bool()
        else:
            active_mask = (probs > 0.5).squeeze(-1).bool()
            mask_ste = active_mask.float().unsqueeze(-1).to(x.dtype)
            
        return active_mask, mask_ste, aux_loss

    def _native_varlen_attention_fallback(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int, max_seqlen_k: int,
        is_causal: bool, pos_ids_q: torch.Tensor, pos_ids_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized fallback for variable-length attention when Triton/FlashAttn are unavailable.
        Note: This is significantly slower due to Python loop overhead over batch dimension.
        """
        cu_seqlens_q_cpu = cu_seqlens_q.cpu()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu()
        batch_size = len(cu_seqlens_q_cpu) - 1
        output = torch.zeros_like(q)
        
        scale = 1.0 / math.sqrt(self.head_dim)

        for i in range(batch_size):
            start_q, end_q = cu_seqlens_q_cpu[i].item(), cu_seqlens_q_cpu[i+1].item()
            start_k, end_k = cu_seqlens_k_cpu[i].item(), cu_seqlens_k_cpu[i+1].item()
            
            if start_q >= end_q:
                continue
                
            qi = q[start_q:end_q].unsqueeze(0).transpose(1, 2) # [1, H, Sq, D]
            ki = k[start_k:end_k].unsqueeze(0).transpose(1, 2) # [1, H, Sk, D]
            vi = v[start_k:end_k].unsqueeze(0).transpose(1, 2)
            
            # Handle GQA: Repeat KV heads
            n_heads_q = qi.shape[1]
            n_heads_k = ki.shape[1]
            if n_heads_q != n_heads_k:
                 n_rep = n_heads_q // n_heads_k
                 ki = ki.repeat_interleave(n_rep, dim=1)
                 vi = vi.repeat_interleave(n_rep, dim=1)

            attn_mask = None
            is_causal_arg = is_causal
            
            if is_causal:
                 p_q = pos_ids_q[start_q:end_q]
                 p_k = pos_ids_k[start_k:end_k]
                 mask = p_q.unsqueeze(1) >= p_k.unsqueeze(0)
                 attn_mask = torch.zeros(mask.shape, device=q.device, dtype=q.dtype).masked_fill(~mask, float('-inf'))
                 is_causal_arg = False

            out_i = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=attn_mask, scale=scale, is_causal=is_causal_arg)
            output[start_q:end_q] = out_i.transpose(1, 2).squeeze(0)
            
        return output

    def forward(self, x: torch.Tensor, pos_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the HSPMN block."""
        B, S, D = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(S, device=x.device).expand(B, S)
        
        # 1. Routing Decision (Fused)
        active_mask, mask_ste, aux_loss = self._fused_router_forward(x, self.training)
        
        # 2. Reflexive Stream (Always executed)
        # Apply Depthwise Conv1d for local mixing (B, D, S) -> (B, D, S)
        # Use Causal Padding (Left Padding) to prevent data leakage from future tokens
        x_transposed = x.transpose(1, 2)
        x_padded = F.pad(x_transposed, (2, 0)) # Pad left by kernel_size - 1
        x_conv = self.shallow_mixer(x_padded).transpose(1, 2)
        shallow_out = self.shallow_mlp(x_conv) 
        
        # 3. Contextual Stream (SQDK Execution)
        x_flat = x.view(-1, D)
        mask_flat = active_mask.view(-1)
        pos_ids_flat = pos_ids.reshape(-1)
        
        if not mask_flat.any():
            return x + shallow_out, aux_loss

        # Gather active tokens
        x_active = x_flat[mask_flat]
        pos_ids_active = pos_ids_flat[mask_flat]
        
        # Projections
        q_active = self.W_q(x_active).view(-1, self.config.num_heads, self.head_dim)
        k_dense = self.W_k(x).view(-1, self.config.num_kv_heads, self.head_dim) 
        v_dense = self.W_v(x).view(-1, self.config.num_kv_heads, self.head_dim)
        
        # RoPE
        cos, sin = self.rotary_emb(v_dense, seq_len=S)
        pos_ids_dense = pos_ids.reshape(-1)
        q_active, k_dense = apply_rotary_pos_emb(q_active, k_dense, cos, sin, pos_ids_active, pos_ids_dense)
        
        # Prepare Ragged Metadata
        active_counts = active_mask.sum(dim=1).int()
        cu_seqlens_q = torch.zeros(B + 1, device=x.device, dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(active_counts, dim=0)
        cu_seqlens_k = torch.arange(0, (B + 1) * S, step=S, device=x.device, dtype=torch.int32)
        
        # Attention Execution
        if TRITON_AVAILABLE and x.is_cuda:
            attn_out_active = sparse_query_dense_key_attention(
                q_active, k_dense, v_dense,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=S, max_seqlen_k=S,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                is_causal=True,
                pos_ids_q=pos_ids_active,
                pos_ids_k=pos_ids_dense
            )
        else:
            attn_out_active = self._native_varlen_attention_fallback(
                q_active, k_dense, v_dense,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=S, max_seqlen_k=S,
                is_causal=True,
                pos_ids_q=pos_ids_active,
                pos_ids_k=pos_ids_dense
            )
        
        # Output Projection & Scatter
        attn_out_active = self.W_o(attn_out_active.view(-1, D))
        
        deep_out_flat = torch.zeros_like(x_flat)
        active_indices = torch.nonzero(mask_flat).squeeze()
        
        # Apply STE Mask for gradient flow
        mask_ste_active = mask_ste.view(-1, 1)[mask_flat]
        attn_out_active = attn_out_active * mask_ste_active
        
        deep_out_flat.index_copy_(0, active_indices, attn_out_active)
            
        return x + shallow_out + deep_out_flat.view(B, S, D), aux_loss

def smoke_test():
    """Runs a basic smoke test to verify model functionality."""
    print("="*60)
    print("HSPMN v2 Model Smoke Test")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    try:
        config = HSPMNv2Config(dim=64, num_heads=4, num_kv_heads=2)
        logger.info(f"Initializing HSPMNv2Block with {config}")
        model = HSPMNv2Block(config).to(device)
        
        batch, seq = 2, 16
        logger.info(f"Creating input tensor [Batch={batch}, Seq={seq}, Dim={config.dim}]")
        x = torch.randn(batch, seq, config.dim).to(device)
        
        logger.info("Running forward pass...")
        out, aux = model(x)
        
        print("-" * 40)
        print(f"✅ Output shape: {out.shape}")
        print(f"✅ Aux loss:     {aux.item():.6f}")
        print(f"✅ Output stats: Mean={out.mean().item():.4f}, Std={out.std().item():.4f}")
        
        logits, probs, _ = model.router(x)
        print(f"🔍 Router probs: Mean={probs.mean().item():.4f}, Min={probs.min().item():.4f}, Max={probs.max().item():.4f}")
        print("-" * 40)
        logger.info("Smoke test passed successfully.")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    smoke_test()
