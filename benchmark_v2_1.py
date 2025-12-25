"""HSPMN v2.1 Benchmarking Suite for RTX 5090.
Usage: python benchmark_rtx5090.py --mode [throughput|stress|all]
"""

import argparse
import statistics
import time
import torch
import gc
from hspmn_v2_1 import HSPMNBlock
from utils_v2_1 import HSPMNConfig, setup_env, get_device

setup_env()

def print_header(title):
    print(f"\n{'=' * 60}\n🚀 HSPMN v2.1: {title}\n   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n{'=' * 60}")

def run_throughput_test(batch: int, seq_len: int, dim: int, iterations: int):
    device = get_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print_header("Throughput Benchmark")
    print(f"Config: Batch={batch}, SeqLen={seq_len}, Dim={dim}")

    config = HSPMNConfig(dim=dim, num_heads=16, max_seq_len=seq_len)
    model = HSPMNBlock(config).to(device, dtype=dtype).eval()
    print("Compiling model (reduce-overhead mode)...")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    print("Warmup (15 iter)...")
    with torch.no_grad():
        for _ in range(15):
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Running {iterations} iterations...")
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.time()
                _ = model(x)
                end = time.time()
                times.append((end - start) * 1000)
    
    avg_ms = statistics.mean(times)
    p95_ms = sorted(times)[int(0.95 * len(times))]
    throughput = (batch * seq_len) / (avg_ms / 1000)
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    
    print(f"{'-' * 60}\n✅ RESULTS:\n   Avg Latency:    {avg_ms:.2f} ms\n   P95 Latency:    {p95_ms:.2f} ms\n   Throughput:     {throughput:,.0f} tok/s\n   Peak VRAM:      {vram:.2f} GB\n{'=' * 60}")

def run_stress_test(max_seq: int):
    device = get_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dim = 2048
    print_header("Max Context Stress Test")
    if torch.cuda.is_available():
        print(f"Finding max seq len for {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Finding max seq len for CPU")

    seq_lengths = [8192, 16384, 32768, 65536, 131072]
    if max_seq > 131072:
        seq_lengths.append(max_seq)

    for S in seq_lengths:
        try:
            print(f"Testing SeqLen={S:<8} ... ", end="", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            config = HSPMNConfig(dim=dim, num_heads=16, max_seq_len=S)
            model = HSPMNBlock(config).to(device, dtype=dtype).eval()
            x = torch.zeros(1, S, dim, device=device, dtype=dtype)
            
            # Compile to match production inference memory profile
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

            with torch.no_grad():
                _ = model(x)
            
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"✅ PASS | VRAM: {mem:.2f} GB")
            del model, x
            
        except torch.cuda.OutOfMemoryError:
            print("❌ OOM Limit Reached")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            break
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["throughput", "stress", "all"], default="all")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()

    if args.mode in ["throughput", "all"]:
        run_throughput_test(args.batch, args.seq_len, args.dim, args.iter)
    
    if args.mode in ["stress", "all"]:
        run_stress_test(max_seq=262144)

if __name__ == "__main__":
    main()
