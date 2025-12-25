# HSPMN v2.1: Bio-Inspired Adaptive Computation 🚀

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-orange)
![Status](https://img.shields.io/badge/status-production_ready-green)
![Performance](https://img.shields.io/badge/throughput-1.41M_tok/s-brightgreen)

**HSPMN v2.1** is a next-generation LLM architecture optimized for **NVIDIA Blackwell (RTX 5090)**. It achieves **extreme throughput** on a single GPU by introducing bio-inspired adaptive computation patterns and hardware-native block sparsity.

> *"The brain doesn't use the full weight of the neocortex to process a simple 'hello'. Why should our models?"*

---

## 🎯 Key Features

- ✅ **Hybrid Execution Strategy**: FlexAttention (Training) + Triton SQDK Kernels (Inference).
- ✅ **Hardware-Native Sparsity**: Custom Triton kernels optimized for H100/Blackwell (`num_warps=4`).
- ✅ **262k Context Window**: Verified on RTX 5090 (11.94 GB VRAM usage).
- ✅ **High Throughput**: 1.41M tokens/sec (Production Scale, BF16).
- ✅ **Entropy Minimization**: Router learns crisp binary decisions.
- ✅ **Zero Graph Breaks**: Fully compatible with `torch.compile`.

---

## 🚀 Performance Verified (RTX 5090)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Throughput** | **1,406,304 tok/s** | Batch=64, Seq=4096, Dim=2048 (Triton Kernel) |
| **Max Context** | **262,144 tokens** | Batch=1, Dim=2048 (11.94 GB VRAM) |
| **Latency** | **186 ms** | End-to-end forward pass (Batch=64, 4k seq) |
| **Training Speed** | **~980k tok/s** | Real training with gradients (FlexAttention) |

---

## 📦 Installation

Prerequisites: NVIDIA Driver 550+, CUDA 12.4+, Python 3.10+

```bash
# Clone repository
git clone https://github.com/your-username/HSPMN-v2.git
cd HSPMN-v2

# Install dependencies (Strictly pinned for stability)
pip install -r requirements.txt
```

---

## 🎓 Quick Start

### 1. Run Benchmarks
Verify your hardware capability immediately:
```bash
# Run both throughput and stress tests
python benchmark_v2_1.py --mode all
```

### 2. Simple Inference
```python
import torch
from hspmn_v2_1 import HSPMNBlock, HSPMNConfig

# Configure for speed
config = HSPMNConfig(dim=2048, num_heads=16, sparsity_k=0.2)
model = HSPMNBlock(config).cuda().bfloat16()

# CRITICAL: Model uses compiled flex_attention internally
# Compile the full model for maximum performance
model = torch.compile(model, mode="reduce-overhead")

# Run (first call will be slow due to compilation)
x = torch.randn(1, 4096, 2048).cuda().bfloat16()
output, aux_loss = model(x)
print(output.shape)
```

### 3. High-Performance Training
```bash
python train_v2_1.py \
    --batch 32 \
    --seq_len 4096 \
    --dim 2048 \
    --steps 1000 \
    --grad_accum 4 \
    --wandb "hspmn-experiment-1"
```

### 4. Testing & Verification
Ensure kernel correctness and model integrity:
```bash
# Verify Triton kernels against PyTorch reference
python test_kernels_v2_1.py

# Verify saved checkpoints
python verify_models.py
```

---

## 🧠 Architecture Highlights

1.  **Reflexive Stream (System 1):**
    *   Runs on *all* tokens.
    *   Components: RMSNorm -> Depthwise Conv1d -> SwiGLU MLP.
    *   Role: Syntax, grammar, shallow processing.

2.  **Contextual Stream (System 2):**
    *   Runs on *sparse* tokens (Top-K Router).
    *   **Inference**: Uses custom **Triton SQDK Kernel** for max speed.
    *   **Training**: Uses **FlexAttention** for autograd support.
    *   Role: Logic, reasoning, long-range dependencies.

3.  **Router:**
    *   Learned Top-K selection with Gumbel-Softmax.
    *   **Entropy Minimization** ensures the router makes confident (0 or 1) decisions.

---

## 💡 Core Concept & Applications

**HSPMN v2.1** addresses the quadratic bottleneck of traditional Transformers by decoupling **memory capacity** from **compute cost**. While standard models process every token with equal intensity, HSPMN uses a **Dual-System Architecture**:

1.  **Reflexive Stream (System 1):** Handles syntax and local patterns for *all* tokens (Linear complexity).
2.  **Contextual Stream (System 2):** Activates heavy attention *only* for semantically dense tokens (Sparse complexity).

### Real-World Use Cases
*   **Private Long-Document Analysis:** Process 500+ page legal/medical contracts locally on a single GPU without data leaving the premise.
*   **Repository-Level Coding Agents:** Ingest entire codebases (200k+ tokens) into context for "whole-project" awareness with low typing latency.
*   **Real-Time Log Filtering:** Efficiently scan terabytes of server logs, where the Router automatically learns to ignore repetitive noise and attend only to anomalies.

---

## 📂 Project Structure

```
HSPMN-v2/
├── hspmn_v2_1.py           # Core architecture (Clean, Type-hinted)
├── kernels_v2_1.py         # Custom Triton SQDK kernels
├── utils_v2_1.py           # Configuration and helper functions
├── train_v2_1.py           # Production-grade training script
├── benchmark_v2_1.py       # Unified benchmarking tool
├── test_kernels_v2_1.py    # Unit tests for Triton kernels
├── verify_models.py        # Checkpoint verification script
├── requirements.txt        # Minimal dependencies
└── README.md               # Documentation
```

---

**Author**: Szymon Jędryczko
**License**: MIT
