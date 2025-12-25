import torch
import logging
import random
import numpy as np
from dataclasses import dataclass

def setup_env():
    """Sets up the environment for training/inference."""
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

def get_device() -> torch.device:
    """Returns the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def setup_logging(name: str = __name__) -> logging.Logger:
    """Configures and returns a logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(name)

def seed_everything(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class HSPMNConfig:
    """Configuration for HSPMN v2.1 model."""
    dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    sparsity_k: float = 0.2
    mlp_ratio: int = 4
    max_seq_len: int = 16384
    rope_base: int = 10000
    block_size: int = 128
    router_temp_init: float = 1.0
    router_sparsity_coef: float = 0.1  # Lambda_1: Coefficient for sparsity loss
    router_entropy_coef: float = 0.01  # Lambda_2: Coefficient for entropy regularization

    def __post_init__(self):
        assert self.dim % self.num_heads == 0, "Dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "Heads must be divisible by KV heads (GQA)"
        self.head_dim = self.dim // self.num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
