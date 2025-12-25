"""HSPMN v2.1: Hierarchical Shallow Predictive Matter Networks.

Optimized for NVIDIA Blackwell using FlexAttention and block sparsity.
Author: Szymon Jędryczko, Dec 2025
"""

import math
from typing import Tuple, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from utils_v2_1 import HSPMNConfig

try:
    from kernels_v2_1 import sparse_query_dense_key_attention
    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False

__all__ = ["HSPMNBlock", "TopKRouter", "HSPMNConfig"]


class RouterOutput(NamedTuple):
    """Output from the TopKRouter."""
    mask: torch.Tensor
    indices: torch.Tensor
    logits: torch.Tensor
    aux_loss: torch.Tensor


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Inverse frequency Rotary Embedding optimized for torch.compile."""
    def __init__(self, dim: int, max_len: int = 131072, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cached_cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("cached_sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        # Dynamic cache resizing if needed
        if seq_len > self.cached_cos.shape[2]:
            self._build_cache(seq_len)
        cos = self.cached_cos[:, :, :seq_len, :]
        sin = self.cached_sin[:, :, :seq_len, :]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings (RoPE)."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# Cache compiled flex_attention for performance
_compiled_flex_attention = None

def get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex_attention


class TopKRouter(nn.Module):
    """Differentiable Top-K Router with Gumbel-Softmax and Entropy Minimization."""
    def __init__(self, dim: int, target_sparsity: float = 0.2, sparsity_coef: float = 0.1, entropy_coef: float = 0.01):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=True)
        # Register as buffer to allow dynamic updates (annealing) without recompilation
        self.register_buffer('target_sparsity', torch.tensor(target_sparsity))
        self.sparsity_coef = sparsity_coef
        self.entropy_coef = entropy_coef
        self.log_temp = nn.Parameter(torch.zeros(1))
        
        nn.init.xavier_uniform_(self.gate.weight, gain=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        B, S, _ = x.shape
        logits = self.gate(x).squeeze(-1)
        temp = torch.exp(self.log_temp).clamp(0.1, 2.0)
        
        if self.training:
            U = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(U + 1e-10) + 1e-10)
            noisy_logits = (logits + gumbel_noise) / temp
        else:
            noisy_logits = logits / temp

        probs = torch.sigmoid(logits)
        sparsity_loss = (probs.mean() - self.target_sparsity).pow(2)
        dist_entropy = -(probs * (probs + 1e-10).log() + (1.0 - probs) * (1.0 - probs + 1e-10).log())
        aux_loss = (self.sparsity_coef * sparsity_loss) + (self.entropy_coef * dist_entropy.mean())
        
        k = max(1, int(S * self.target_sparsity.item()))
        _, indices = torch.topk(noisy_logits, k, dim=1, sorted=False)
        
        # Sort indices for better memory locality in attention
        indices, _ = torch.sort(indices, dim=-1)
        indices = indices.contiguous()
        
        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask.scatter_(1, indices, True)

        return RouterOutput(mask, indices, logits, aux_loss)


class ReflexiveStream(nn.Module):
    """Lightweight stream: RMSNorm -> Conv1d -> SwiGLU MLP."""
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        for w in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(w.weight, gain=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return res + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HSPMNBlock(nn.Module):
    """Main HSPMN Block combining Reflexive and Contextual Streams."""
    def __init__(self, config: HSPMNConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.router = TopKRouter(config.dim, config.sparsity_k, config.router_sparsity_coef, config.router_entropy_coef)
        self.norm = nn.RMSNorm(config.dim)
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        self.reflexive = ReflexiveStream(config.dim, config.mlp_ratio)
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.dim)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=scale)

    def _attention_triton(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, router_out: RouterOutput, B: int, S: int, D: int) -> torch.Tensor:
        """Executes Sparse-Query Dense-Key (SQDK) Attention using Triton kernels."""
        # q: [B, H, S, D] -> [B, S, H, D]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Gather selected queries
        # indices: [B, K] -> expand to [B, K, H, D]
        indices_expanded = router_out.indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        q_selected = torch.gather(q_t, 1, indices_expanded).contiguous()
        
        # Run SQDK Attention
        # q_selected: [B, K, H, D]
        # k_t, v_t: [B, S, H, D]
        # indices: [B, K]
        attn_out = sparse_query_dense_key_attention(q_selected, k_t, v_t, router_out.indices)
        
        # Scatter back
        out = torch.zeros_like(q_t)
        out.scatter_(1, indices_expanded, attn_out)
        
        return self.o_proj(out.view(B, S, D))

    def _attention_flex(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, router_out: RouterOutput, B: int, S: int, D: int) -> torch.Tensor:
        """Executes attention using PyTorch FlexAttention."""
        def block_mask_fn(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & router_out.mask[b, q_idx]

        block_mask = create_block_mask(
            block_mask_fn, B=B, H=self.num_heads, Q_LEN=S, KV_LEN=S,
            device=q.device, BLOCK_SIZE=self.config.block_size
        )

        flex_attn_fn = get_compiled_flex_attention()
        out = flex_attn_fn(
            q, k, v, block_mask=block_mask, enable_gqa=(self.num_kv_heads != self.num_heads),
            scale=1.0 / math.sqrt(self.head_dim)
        )
        
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

    def _attention(self, x: torch.Tensor, router_out: RouterOutput) -> torch.Tensor:
        B, S, D = x.shape
        x_norm = self.norm(x)

        q = self.q_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use Triton Kernel if available (SQDK) - Inference Only (no backward pass yet)
        if HAS_TRITON_KERNELS and router_out.indices is not None and not self.training:
            return self._attention_triton(q, k, v, router_out, B, S, D)

        # Fallback to FlexAttention
        return self._attention_flex(q, k, v, router_out, B, S, D)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_out = self.router(x)
        return self.reflexive(x) + self._attention(x, router_out), router_out.aux_loss
