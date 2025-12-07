"""
Unit tests for HSPMN v2 Logic.

This suite verifies the architectural integrity of the HSPMN model, focusing on:
1.  **Entropy Routing**: Ensuring probability distributions are valid.
2.  **Sparse Topology**: Verifying that token gathering and scattering preserves data.
3.  **Native SDPA**: Checking that the variable-length attention mechanism runs without errors.
4.  **End-to-End Flow**: Confirming the full block processes data correctly.

Author: Szymon Jędryczko
Date: December 2025
"""

import logging
import unittest
import torch
from model_hspmn_v2 import HSPMNv2Block, HSPMNv2Config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class TestHSPMNv2Architecture(unittest.TestCase):
    """
    Comprehensive test suite for HSPMN components.
    """
    
    def setUp(self):
        """Set up a consistent testing environment."""
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.dim = 64
        self.num_heads = 4
        self.num_kv_heads = 2 # GQA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use Config object
        self.config = HSPMNv2Config(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        self.model = HSPMNv2Block(self.config).to(self.device)
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)

    def test_router_probabilities(self):
        """
        Test: Sparsity Router output.
        Expectation: Output shape [B, S, 1] and values in range [0, 1].
        """
        logger.info("Testing Router Probabilities...")
        logits, probs, aux_loss = self.model.router(self.x)
        
        logger.info(f"Probs Mean: {probs.mean().item():.4f} | Aux Loss: {aux_loss.item():.4f}")

        self.assertEqual(probs.shape, (self.batch_size, self.seq_len, 1), 
                         "Router output shape mismatch.")
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0), 
                        "Router probabilities must be between 0 and 1.")
        self.assertTrue(aux_loss.numel() == 1, "Aux loss should be a scalar.")

    def test_topology_preservation(self):
        """
        Test: Gather/Scatter logic (Sparse Execution).
        Expectation: Tokens selected for the deep path should be returned to their 
        exact original positions after processing.
        """
        logger.info("Testing Topology Preservation (Gather/Scatter)...")
        # Create a recognizable pattern
        x_cpu = torch.arange(self.seq_len * self.dim, dtype=torch.float32).view(1, self.seq_len, self.dim)
        x = x_cpu.to(self.device)
        
        # Force a specific mask: Select every 2nd token
        mask = torch.zeros(1, self.seq_len, dtype=torch.bool, device=self.device)
        mask[:, ::2] = True
        
        # Simulate the gather/scatter process manually to verify logic
        x_flat = x.view(-1, self.dim)
        mask_flat = mask.view(-1)
        
        # Gather
        x_active = x_flat[mask_flat]
        logger.info(f"Gathered {x_active.shape[0]} tokens out of {x_flat.shape[0]}")
        
        # Verify gathered content
        expected_active = x_cpu[0, ::2, :].to(self.device)
        self.assertTrue(torch.equal(x_active, expected_active), "Gathered tokens do not match expected values.")
        
        # Scatter (Identity operation for test)
        deep_out_flat = torch.zeros_like(x_flat)
        active_indices = torch.nonzero(mask_flat).squeeze()
        deep_out_flat.index_copy_(0, active_indices, x_active)
        deep_out = deep_out_flat.view(1, self.seq_len, self.dim)
        
        # Verify scattered content (Active positions should match original, others zero)
        expected_out = torch.zeros_like(x)
        expected_out[:, ::2, :] = x[:, ::2, :]
        
        self.assertTrue(torch.equal(deep_out, expected_out), "Scattered tokens do not match expected topology.")

    def test_native_sdpa_execution(self):
        """
        Test: Native Variable Length Attention (SQDK).
        Expectation: Function runs and returns tensor of correct shape.
        """
        logger.info("Testing Native SDPA Execution...")
        # Mock inputs for SDPA
        total_tokens_q = 6 # Sparse
        total_tokens_k = 10 # Dense
        head_dim = self.dim // self.num_heads
        q = torch.randn(total_tokens_q, self.num_heads, head_dim, device=self.device)
        
        # Note: native_varlen_attention expects matching heads if not using flash_attn logic internally for GQA
        # But for this unit test of the function itself, we'll use matching heads to be safe, 
        # as the GQA expansion happens inside the Attention module, not necessarily the raw kernel wrapper.
        k = torch.randn(total_tokens_k, self.num_heads, head_dim, device=self.device)
        v = torch.randn(total_tokens_k, self.num_heads, head_dim, device=self.device)
        
        # 2 sequences. 
        # Seq 1: 5 tokens total, 3 active.
        # Seq 2: 5 tokens total, 3 active.
        cu_seqlens_q = torch.tensor([0, 3, 6], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=self.device)
        
        # Mock pos_ids
        # Seq 1: 5 tokens. Active: 0, 2, 4.
        # Seq 2: 5 tokens. Active: 1, 3, 4.
        pos_ids_q = torch.tensor([0, 2, 4, 1, 3, 4], dtype=torch.long, device=self.device)
        pos_ids_k = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long, device=self.device)

        try:
            # Use the internal fallback method for testing logic correctness
            out = self.model._native_varlen_attention_fallback(
                q, k, v, 
                cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q=5, max_seqlen_k=5, 
                is_causal=True,
                pos_ids_q=pos_ids_q,
                pos_ids_k=pos_ids_k
            )
            logger.info(f"SDPA Output Shape: {out.shape}")
            self.assertEqual(out.shape, q.shape, "SDPA output shape mismatch.")
        except Exception as e:
            self.fail(f"Native SDPA raised an exception: {e}")

    def test_full_forward_pass(self):
        """
        Test: End-to-end model forward pass.
        Expectation: Output shape matches input shape [B, S, D].
        """
        logger.info("Testing Full Forward Pass...")
        try:
            out, aux_loss = self.model(self.x)
            logger.info(f"Model Output Shape: {out.shape} | Aux Loss: {aux_loss.item():.4f}")
            self.assertEqual(out.shape, self.x.shape, "Model output shape mismatch.")
            self.assertTrue(aux_loss.numel() == 1, "Aux loss should be a scalar.")
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
