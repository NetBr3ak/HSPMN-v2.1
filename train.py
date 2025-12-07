"""
Training script for HSPMN v2.

Implements a professional training loop with:
- Gradient Clipping
- Learning Rate Scheduling (Cosine Decay)
- Mixed Precision (BF16)
- Logging and Checkpointing
- Synthetic Data Loading
- Progress Bars (tqdm)

Author: Szymon Jędryczko
Date: December 2025
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from model_hspmn_v2 import HSPMNv2Block, HSPMNv2Config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Set High Performance Computing settings
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

@dataclass
class TrainingConfig:
    """Configuration for the training run."""
    batch_size: int
    seq_len: int
    dim: int
    heads: int
    steps: int
    lr: float
    log_interval: int
    save_interval: int

class SyntheticTextDataset(IterableDataset):
    """
    Generates infinite synthetic data for training throughput testing.
    """
    def __init__(self, seq_len: int, dim: int):
        self.seq_len = seq_len
        self.dim = dim

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            # Simulate embeddings directly for this demo
            x = torch.randn(self.seq_len, self.dim)
            # Target: simple regression or next-token prediction simulation
            y = torch.randn(self.seq_len, self.dim) 
            yield x, y

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, step: int, path: str):
    """Saves a training checkpoint safely."""
    logger.info(f"Saving checkpoint to {path}...")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def print_header(config: TrainingConfig, device: torch.device, dtype: torch.dtype):
    print("=" * 60)
    print(f"🚀 HSPMN v2 Training Loop")
    print(f"   Device:      {device}")
    print(f"   Precision:   {dtype}")
    print(f"   Config:      Batch={config.batch_size}, SeqLen={config.seq_len}, Steps={config.steps}")
    print("=" * 60)

def train(config: TrainingConfig):
    # 1. Setup Device & Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print_header(config, device, dtype)

    # 2. Initialize Model
    model_config = HSPMNv2Config(dim=config.dim, num_heads=config.heads)
    model = HSPMNv2Block(model_config)
    model.to(device).to(dtype)
    
    # Initialize router bias for sparsity start
    nn.init.constant_(model.router.gate.bias, -1.0) 
    
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # 3. Setup Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    criterion = nn.MSELoss() # Placeholder loss for synthetic data

    # 4. Data Loader
    dataset = SyntheticTextDataset(config.seq_len, config.dim)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=2, pin_memory=True)
    data_iter = iter(dataloader)

    # 5. Training Loop
    model.train()
    logger.info("Starting training loop...")
    
    start_time = time.time()
    warmup_steps = int(0.1 * config.steps) # 10% warmup
    
    try:
        for step in range(1, config.steps + 1):
            # Anneal target sparsity from 0.8 to 0.2 during warmup
            if step <= warmup_steps:
                current_target = 0.8 - (0.6 * (step / warmup_steps))
                model.router.target_sparsity = current_target
            else:
                model.router.target_sparsity = 0.2

            x, y = next(data_iter)
            x, y = x.to(device).to(dtype), y.to(device).to(dtype)
            
            optimizer.zero_grad()
            
            # Forward Pass
            output, aux_loss = model(x)
            
            # Loss Calculation
            main_loss = criterion(output, y)
            total_loss = main_loss + 0.1 * aux_loss
            
            # Backward Pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            if step % config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (config.batch_size * config.seq_len * config.log_interval) / elapsed
                start_time = time.time()
                
                logger.info(
                    f"Step {step}/{config.steps} | "
                    f"Loss: {total_loss.item():.4f} (Aux: {aux_loss.item():.4f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Throughput: {tokens_per_sec:,.0f} tok/s"
                )
                
            # Checkpointing
            if step % config.save_interval == 0:
                save_checkpoint(model, optimizer, step, f"checkpoint_step_{step}.pt")
                
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    
    print("-" * 60)
    print("✅ RESULTS:")
    print(f"   Status:      Completed {step} steps")
    print(f"   Final Loss:  {total_loss.item():.4f}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="HSPMN v2 Training Script")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint interval")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        batch_size=args.batch,
        seq_len=args.seq_len,
        dim=args.dim,
        heads=args.heads,
        steps=args.steps,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )
    
    train(config)

if __name__ == "__main__":
    main()
