"""HSPMN v2.1 Training Script.
Optimized for RTX 5090 with PyTorch 2.5+ AMP, gradient accumulation, and W&B.
Author: Szymon Jędryczko
"""

import argparse
import time
import math
import logging
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from hspmn_v2_1 import HSPMNBlock
from utils_v2_1 import HSPMNConfig, setup_env, get_device, setup_logging, seed_everything

setup_env()
logger = setup_logging()


@dataclass
class TrainingConfig:
    batch_size: int = 32
    seq_len: int = 1024
    dim: int = 512
    heads: int = 8
    steps: int = 1000
    lr: float = 3e-4
    log_interval: int = 10
    save_interval: int = 500
    grad_accum: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    wandb_project: Optional[str] = None
    run_name: str = "hspmn-v2-blackwell"


class SyntheticDataset(IterableDataset):
    """Infinite stream of synthetic data."""
    def __init__(self, seq_len: int, dim: int):
        self.seq_len = seq_len
        self.dim = dim

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield torch.randn(self.seq_len, self.dim), torch.randn(self.seq_len, self.dim)


def get_cosine_schedule(optimizer, warmup_steps, training_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_device()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self._init_wandb()
        self._init_model()
        self._init_optimizer()
        self._init_dataloader()
        
    def _init_wandb(self):
        if self.config.wandb_project and HAS_WANDB:
            wandb.init(project=self.config.wandb_project, name=self.config.run_name, config=self.config.__dict__)

    def _init_model(self):
        logger.info("=" * 60)
        logger.info(f"🚀 HSPMN v2.1 Training | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | {self.dtype}")
        logger.info(f"   Config: B={self.config.batch_size}, S={self.config.seq_len}, D={self.config.dim}, Steps={self.config.steps}")
        logger.info("=" * 60)

        model_config = HSPMNConfig(dim=self.config.dim, num_heads=self.config.heads, max_seq_len=self.config.seq_len)
        self.model = HSPMNBlock(model_config).to(self.device, dtype=self.dtype)
        logger.info(f"Model Params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
        logger.info("Compiling model...")
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
        self.raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _init_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay,
                                     betas=(0.9, 0.95), fused=True)
        self.scheduler = get_cosine_schedule(self.optimizer, self.config.warmup_steps, self.config.steps)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))
        self.criterion = nn.MSELoss()

    def _init_dataloader(self):
        dataset = SyntheticDataset(self.config.seq_len, self.config.dim)
        num_workers = min(8, torch.get_num_threads() // 2)
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=num_workers,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
        self.data_iter = iter(self.dataloader)

    def _anneal_sparsity(self, step: int):
        if step <= self.config.warmup_steps:
            progress = step / self.config.warmup_steps
            current_sparsity = 0.8 - (0.6 * progress)
            self.raw_model.router.target_sparsity.fill_(current_sparsity)
        else:
            self.raw_model.router.target_sparsity.fill_(0.2)

    def train(self):
        self.model.train()
        torch.cuda.empty_cache()
        start_time = time.time()
        total_loss, total_aux, tokens_processed = 0.0, 0.0, 0
        best_loss = float('inf')

        try:
            pbar = tqdm(range(1, self.config.steps + 1), desc="Training", unit="step")
            
            for step in pbar:
                self._anneal_sparsity(step)
                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                
                for _ in range(self.config.grad_accum):
                    try:
                        x, y = next(self.data_iter)
                    except StopIteration:
                        self.data_iter = iter(self.dataloader)
                        x, y = next(self.data_iter)
                    
                    x, y = x.to(self.device, non_blocking=True, dtype=self.dtype), y.to(self.device, non_blocking=True, dtype=self.dtype)

                    with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                        out, aux_loss = self.model(x)
                        loss = (self.criterion(out, y) + aux_loss) / self.config.grad_accum
                    
                    self.scaler.scale(loss).backward()
                    accum_loss += loss.item()
                    total_loss += loss.item() * self.config.grad_accum - aux_loss.item()
                    total_aux += aux_loss.item()
                    tokens_processed += self.config.batch_size * self.config.seq_len

                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                if step % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss, avg_aux = total_loss / self.config.log_interval, total_aux / self.config.log_interval
                    throughput = tokens_processed / elapsed
                    vram_gb = torch.cuda.memory_allocated() / 1e9

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "aux": f"{avg_aux:.4f}",
                                      "tok/s": f"{throughput:,.0f}", "vram": f"{vram_gb:.1f}GB",
                                      "sp": f"{self.raw_model.router.target_sparsity.item():.2f}"})

                    if self.config.wandb_project and HAS_WANDB:
                        wandb.log({"loss": avg_loss, "aux_loss": avg_aux, "lr": self.scheduler.get_last_lr()[0],
                                   "throughput": throughput, "vram_gb": vram_gb,
                                   "sparsity_target": self.raw_model.router.target_sparsity.item()}, step=step)

                    start_time, total_loss, total_aux, tokens_processed = time.time(), 0.0, 0.0, 0

                if step % self.config.save_interval == 0 or step == self.config.steps:
                    state = {'step': step, 'model': self.model.state_dict(), 'optim': self.optimizer.state_dict(), 'config': self.config.__dict__}
                    torch.save(state, f"checkpoint_step_{step}.pt")
                    if accum_loss < best_loss:
                        best_loss = accum_loss
                        torch.save(state, "best_model.pt")

        except KeyboardInterrupt:
            logger.warning("Graceful shutdown...")
        finally:
            if self.config.wandb_project and HAS_WANDB:
                wandb.finish()
            logger.info("✅ Training Complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb", type=str, default=None)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        batch_size=args.batch,
        seq_len=args.seq_len,
        dim=args.dim,
        heads=args.heads,
        steps=args.steps,
        lr=args.lr,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        wandb_project=args.wandb
    )
    
    seed_everything()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
