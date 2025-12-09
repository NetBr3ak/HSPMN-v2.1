# HSPMN v2: Bio-Inspired Adaptive Computation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-orange)
![Status](https://img.shields.io/badge/status-research_preview-yellow)

**HSPMN v2** is an experimental LLM architecture that challenges the way standard Transformers work. Instead of spending the exact same amount of compute on every single word, it tries to mimic biological efficiency.

> *"The brain doesn't use the full weight of the neocortex to process a simple 'hello'. Why should our models?"*

---

## 🧠 The Big Idea

Current LLMs suffer from **Computational Isotropy**: the word "the" costs as much to process as a complex scientific concept. That's a waste of energy.

**HSPMN v2** fixes this by splitting the workload into two paths, similar to Kahneman's "System 1" and "System 2" thinking:

1.  **Reflexive Stream (Fast Path):** A lightweight MLP + Conv1d layer. It handles simple syntax and grammar. It runs on *every* token to keep the sentence flowing smoothly.
2.  **Contextual Stream (Slow Path):** A heavy Attention mechanism. It activates *only* for difficult or important tokens that need deep context.

### Why is this cool?
*   **No "Context Fracture":** In many sparse models, if you skip a token, it disappears. Here, we use **SQDK (Sparse-Query, Dense-Key)** attention. Even if a token takes the "Fast Path", it remains visible to future tokens. Nothing gets lost.
*   **Smart Budgeting:** The model learns to be lazy. We force it to route only a small percentage (e.g., 20%) of tokens to the heavy Attention layer.
*   **Hardware Ready:** Designed with next-gen GDDR7 memory in mind, using Grouped Query Attention (GQA) to keep memory usage low.

---

## 🏗️ How It Works (Simplified)

Instead of a complex diagram, think of the data flow like this:

1.  **Input:** Token comes in.
2.  **Router:** A tiny neural network asks: *"Is this token surprising?"*
3.  **Decision:**
    *   **No (Simple token):** Go to the **Reflexive Stream**. Apply simple mixing (Conv1d) so we don't lose track of position. **Cost: Low.**
    *   **Yes (Complex token):** Go to the **Contextual Stream**. Perform full Attention. **Cost: High.**
4.  **Output:** Merge the results.

This ensures we save massive amounts of compute (FLOPs) without breaking the logical chain of the sentence.

---

## 🚀 Performance

We tested this on an **NVIDIA RTX 5090**. Here is the reality of the current implementation:

| Metric | Dense Baseline | HSPMN v2 (Ours) | Impact |
| :--- | :--- | :--- | :--- |
| **Computation (FLOPs)** | 100% | **~18%** | **5.4x less work** |
| **Throughput** | ~182k tok/sec | ~115k tok/sec* | *Python Overhead |

**Wait, why is it slower?**
Currently, the theoretical speedup (5.4x) is masked by the overhead of Python code managing the routing logic. The math is solid, but the engine needs tuning. We are working on custom **OpenAI Triton kernels** to unlock the full speed potential.

---

## 🛠️ Quick Start

### Prerequisites
You'll need Python 3.10+ and a GPU with CUDA support.

### Installation
```bash
# Clone the repo
git clone https://github.com/your-org/HSPMN-v2.git
cd HSPMN-v2

# Set up environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Run a Benchmark

Want to see the router in action?

```bash
python benchmark_rtx5090.py --iter 100 --warmup 10
```

## 📄 Citation

If you find this useful for your research, please cite:

```bibtex
@article{jedryczko2025hspmn,
  title={HSPMN v2: Adaptive Computation via Context-Aware Target-Sparsity Regularized Gating},
  author={Jędryczko, Szymon},
  year={2025}
}
```

## ⚠️ Known Limitations

*   **Memory Bound:** Even though we do fewer calculations, we still need to load all the Keys and Values from memory. On fast GPUs, memory speed is often the bottleneck, not calculation speed.
*   **Router Training:** Teaching the router is tricky. If not careful, it might "collapse" and send everything to one path. We use special regularization to prevent this.
*   **Semantic Isolation:** Tokens on the "Fast Path" don't get to look around (no self-attention). If a token stays shallow for too many layers, it might lose context.

---

*Research Preview - Code provided as-is for educational and research purposes.*
