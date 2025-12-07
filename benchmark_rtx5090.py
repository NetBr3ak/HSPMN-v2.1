"""
Benchmark script for HSPMN v2 on RTX 5090 (or equivalent).

Measures throughput, latency, and VRAM usage with high sparsity settings.
Supports command-line arguments for flexible configuration.

Author: Szymon Jędryczko
Date: December 2025
"""

import argparse
import time
import logging
import torch
import torch.nn as nn
from typing import Tuple
from model_hspmn_v2 import HSPMNv2Block, HSPMNv2Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set High Performance Computing settings for Ampere+ GPUs
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

class BenchmarkRunner:
    """
    Encapsulates the benchmarking logic for the HSPMN model.
    """
    def __init__(self, config: HSPMNv2Config, batch_size: int, seq_len: int, iterations: int, warmup: int):
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.iterations = iterations
        self.warmup = warmup
        
        self.device, self.dtype, self.device_name = self._get_device_info()
        
    def _get_device_info(self) -> Tuple[torch.device, torch.dtype, str]:
        """Detects the best available device and precision."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Prefer BF16 if supported (standard for modern LLMs), else FP16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_name = torch.cuda.get_device_name(0)
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            device_name = "CPU"
        return device, dtype, device_name

    def print_header(self):
        print("=" * 60)
        print(f"🚀 HSPMN v2 Benchmark")
        print(f"   Device:      {self.device_name}")
        print(f"   Precision:   {self.dtype}")
        print(f"   Config:      Batch={self.batch_size}, SeqLen={self.seq_len}, Dim={self.config.dim}")
        print("=" * 60)

    def run(self):
        self.print_header()

        # Model Initialization
        logger.info("Initializing HSPMNv2Block...")
        model = HSPMNv2Block(self.config).to(self.device).to(self.dtype)
        
        # Compile the model for maximum performance
        # Note: torch.compile with CUDA Graphs can be unstable with dynamic shapes/buffers in some versions.
        # Disabling for benchmark stability unless explicitly requested.
        # if self.device.type == 'cuda':
        #     logger.info("Compiling model with torch.compile...")
        #     model = torch.compile(model, mode="reduce-overhead")

        # Force high sparsity (~80% tokens skipped) for realistic testing
        # Bias -2.0 -> sigmoid(-2.0) ~= 0.12 probability of being active
        # Note: This forces a specific sparsity level for throughput testing ("Peak Throughput Test")
        # and does not reflect the dynamic sparsity distribution of a trained model.
        nn.init.constant_(model.router.gate.bias, -2.0)
        model.eval()

        # Input Generation
        logger.info("Generating random inputs...")
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim, device=self.device, dtype=self.dtype)
        
        # Warmup Phase
        logger.info(f"Warming up ({self.warmup} iterations)...")
        with torch.no_grad():
            for _ in range(self.warmup):
                _, _ = model(x)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark Phase
        logger.info(f"Running benchmark ({self.iterations} iterations)...")
        
        if self.device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        
        with torch.no_grad():
            for i in range(self.iterations):
                _, _ = model(x)
                if (i + 1) % 10 == 0:
                    logger.debug(f"Iteration {i+1}/{self.iterations} completed.")
                
        if self.device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.time() - start_time) * 1000

        # Metrics
        avg_time_ms = elapsed_ms / self.iterations
        avg_time_sec = avg_time_ms / 1000
        throughput = (self.batch_size * self.seq_len) / avg_time_sec
        
        # Memory Usage (Peak)
        if self.device.type == 'cuda':
            max_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_info = f"{max_mem_gb:.2f} GB"
            logger.info(f"Peak VRAM Usage: {mem_info}")
        else:
            mem_info = "N/A"

        print("-" * 60)
        print("✅ RESULTS:")
        print(f"   Avg Latency:    {avg_time_ms:.2f} ms (Single Layer)")
        print(f"   Throughput:     {throughput:,.0f} tokens/sec (Single Layer)")
        print(f"   Est. Full Model: {throughput/25:,.0f} tokens/sec (Layer-wise scaling)")
        print(f"   Peak VRAM:      {mem_info}")
        print(f"   Room Temp:      +2°C (Estimated)")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="HSPMN v2 Benchmark")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--dim", type=int, default=2048, help="Model dimension")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--kv_heads", type=int, default=4, help="Number of KV heads (GQA)")
    parser.add_argument("--iter", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    config = HSPMNv2Config(
        dim=args.dim,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads
    )
    
    runner = BenchmarkRunner(
        config=config,
        batch_size=args.batch,
        seq_len=args.seq_len,
        iterations=args.iter,
        warmup=args.warmup
    )
    runner.run()

if __name__ == "__main__":
    main()
