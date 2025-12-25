
import unittest
import torch
import torch.nn.functional as F
from kernels_v2_1 import sparse_query_dense_key_attention

class TestTritonKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_sqdk_attention_correctness(self):
        """
        Verifies that the Sparse-Query Dense-Key (SQDK) Triton kernel produces
        results consistent with a standard PyTorch implementation (masked).
        """
        B, H, S, D = 2, 4, 128, 64
        K_active = 32  # Number of active queries
        
        # Inputs
        q = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=self.device, dtype=torch.float16)
        
        # Select random indices for active queries
        indices = torch.stack([
            torch.randperm(S, device=self.device)[:K_active] for _ in range(B)
        ]) # [B, K_active]
        indices = torch.sort(indices, dim=1)[0] # Sort for stability/determinism if needed
        
        # --- Reference Implementation (PyTorch) ---
        # Gather active queries
        # q: [B, S, H, D]
        # indices: [B, K] -> [B, K, 1, 1] -> expand to [B, K, H, D]
        indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        q_selected = torch.gather(q, 1, indices_expanded) # [B, K, H, D]
        
        # Standard Attention: Softmax(Q_sel @ K.T / sqrt(D)) @ V
        # Q_sel: [B, K, H, D] -> [B, H, K, D]
        # K: [B, S, H, D] -> [B, H, S, D]
        q_ref = q_selected.permute(0, 2, 1, 3)
        k_ref = k.permute(0, 2, 1, 3)
        v_ref = v.permute(0, 2, 1, 3)
        
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale # [B, H, K, S]
        
        # --- Causal Masking ---
        # indices: [B, K_active] contains the real positions of queries
        # We mask keys where key_pos > q_real_pos
        
        # q_pos: [B, 1, K, 1]
        q_pos = indices.unsqueeze(1).unsqueeze(-1)
        # k_pos: [1, 1, 1, S]
        k_pos = torch.arange(S, device=self.device).reshape(1, 1, 1, S)
        
        mask = k_pos <= q_pos # [B, 1, K, S]
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out_ref = torch.matmul(attn_weights, v_ref) # [B, H, K, D]
        out_ref = out_ref.permute(0, 2, 1, 3) # [B, K, H, D]
        
        # --- Triton Implementation ---
        # The kernel expects inputs in specific layout. 
        # Based on hspmn_v2_1.py usage:
        # q_selected: [B, K, H, D]
        # k, v: [B, S, H, D]
        # indices: [B, K]
        
        # Note: kernels_v2_1.py likely expects contiguous tensors
        q_selected_c = q_selected.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        indices_c = indices.contiguous()
        
        # Call the wrapper function from kernels_v2_1
        # Assuming the signature matches what's in hspmn_v2_1.py
        # sparse_query_dense_key_attention(q, k, v, q_indices)
        out_triton = sparse_query_dense_key_attention(q_selected_c, k_c, v_c, indices_c)
        
        # --- Comparison ---
        # Tolerances for float16
        atol = 1e-2
        rtol = 1e-2
        
        # Check shape
        self.assertEqual(out_triton.shape, out_ref.shape)
        
        # Check values
        # Note: Triton kernels might have different accumulation order, so slight diffs expected
        max_diff = (out_triton - out_ref).abs().max().item()
        print(f"Max difference between Triton and PyTorch: {max_diff}")
        
        self.assertTrue(torch.allclose(out_triton, out_ref, atol=atol, rtol=rtol),
                        f"Triton output mismatch. Max diff: {max_diff}")

if __name__ == '__main__':
    unittest.main()
