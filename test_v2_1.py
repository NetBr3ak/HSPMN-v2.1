"""HSPMN v2.1 Unit Test Suite.
Verifies architectural integrity and gradient flow.
Author: Szymon Jędryczko
"""

import unittest
import torch
import torch.nn as nn
from hspmn_v2_1 import HSPMNBlock, TopKRouter
from utils_v2_1 import HSPMNConfig, setup_logging, seed_everything

logger = setup_logging()

class TestHSPMNv2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Running on {cls.device.type.upper()}" + 
                    (f": {torch.cuda.get_device_name(0)}" if cls.device.type == 'cuda' else ""))

    def setUp(self):
        seed_everything(42)
        self.config = HSPMNConfig(dim=64, num_heads=4, num_kv_heads=2, sparsity_k=0.5, block_size=64)

    def test_router_entropy_loss(self):
        router = TopKRouter(self.config.dim, target_sparsity=0.5).to(self.device)
        x = torch.randn(2, 128, self.config.dim, device=self.device)
        out = router(x)
        
        self.assertIsNotNone(out.aux_loss)
        self.assertTrue(out.aux_loss > 0)
        self.assertEqual(out.mask.shape, (2, 128))
        
        out.aux_loss.backward()
        self.assertIsNotNone(router.gate.weight.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "FlexAttention requires CUDA")
    def test_sparse_gradient_isolation(self):
        model = HSPMNBlock(self.config).to(self.device)
        x = torch.randn(2, 128, self.config.dim, device=self.device, requires_grad=True)
        
        class MockRouter(nn.Module):
            def __init__(self, dim, device):
                super().__init__()
                self.gate = nn.Linear(dim, 1)
                self.aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            def forward(self, x):
                B, S, _ = x.shape
                mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
                mask[:, :S//2] = True
                # Create dummy indices for the first half
                indices = torch.arange(S//2, device=x.device).unsqueeze(0).expand(B, -1)
                
                from model_hspmn_v2 import RouterOutput
                # mask, indices, logits, aux_loss
                return RouterOutput(mask, indices, torch.zeros(B, S, device=x.device), self.aux_loss)

        model.router = MockRouter(self.config.dim, self.device).to(self.device)
        out, _ = model(x)
        out.sum().backward()
        
        self.assertIsNotNone(model.q_proj.weight.grad)
        self.assertFalse(torch.isnan(model.q_proj.weight.grad).any())

    def test_reflexive_stream_correctness(self):
        model = HSPMNBlock(self.config).to(self.device)
        x = torch.randn(2, 64, self.config.dim, device=self.device)
        out = model.reflexive(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_gqa_shapes(self):
        cfg = HSPMNConfig(dim=64, num_heads=4, num_kv_heads=1)
        model = HSPMNBlock(cfg).to(self.device)
        self.assertEqual(model.num_kv_heads, 1)

    def test_forward_pass_integration(self):
        model = HSPMNBlock(self.config).to(self.device)
        x = torch.randn(1, 128, self.config.dim, device=self.device)
        out, aux = model(x)
        self.assertEqual(out.shape, (1, 128, self.config.dim))
        self.assertTrue(torch.is_tensor(aux))

if __name__ == '__main__':
    unittest.main()
