#!/usr/bin/env python3
"""
Benchmark script for ProteinSearch-GPU.

Run with: python benchmark.py
"""

import torch
import time
import json
import argparse
from pathlib import Path

try:
    import protein_search_gpu as psg
except ImportError:
    print("protein_search_gpu not installed. Run: pip install -e .")
    exit(1)


def benchmark_l2_normalize(sizes, dtype=torch.float32, iters=50):
    results = []
    for num_vecs, dim in sizes:
        torch.cuda.empty_cache()
        x = torch.randn(num_vecs, dim, device='cuda', dtype=dtype)
        
        for _ in range(10):
            _ = psg.l2_normalize(x)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = psg.l2_normalize(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        results.append({
            'op': 'l2_normalize',
            'num_vectors': num_vecs,
            'dim': dim,
            'dtype': str(dtype),
            'time_ms': elapsed / iters * 1000,
            'throughput': num_vecs * iters / elapsed,
        })
        print(f"L2 Norm: {num_vecs}x{dim} - {elapsed/iters*1000:.2f} ms")
    return results


def benchmark_cosine_similarity(sizes, dtype=torch.float32, iters=20):
    results = []
    for num_q, db_size, dim in sizes:
        torch.cuda.empty_cache()
        q = torch.randn(num_q, dim, device='cuda', dtype=dtype)
        db = torch.randn(db_size, dim, device='cuda', dtype=dtype)
        
        for _ in range(5):
            _ = psg.cosine_similarity(q, db)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = psg.cosine_similarity(q, db)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        results.append({
            'op': 'cosine_similarity',
            'num_queries': num_q,
            'db_size': db_size,
            'dim': dim,
            'dtype': str(dtype),
            'time_ms': elapsed / iters * 1000,
            'throughput': num_q * iters / elapsed,
        })
        print(f"CosSim: {num_q}x{db_size}x{dim} - {elapsed/iters*1000:.2f} ms")
    return results


def benchmark_topk(sizes, k=32, iters=50):
    # The custom top-k kernel supports k <= 32.
    results = []
    for num_q, db_size in sizes:
        torch.cuda.empty_cache()
        sims = torch.randn(num_q, db_size, device='cuda', dtype=torch.float32)
        
        for _ in range(10):
            _ = psg.topk(sims, k)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = psg.topk(sims, k)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        results.append({
            'op': 'topk',
            'num_queries': num_q,
            'db_size': db_size,
            'k': k,
            'time_ms': elapsed / iters * 1000,
            'throughput': num_q * iters / elapsed,
        })
        print(f"TopK: {num_q}x{db_size} k={k} - {elapsed/iters*1000:.2f} ms")
    return results


def benchmark_gather(sizes, k=100, iters=50):
    results = []
    db_size = 100000
    for num_q, dim in sizes:
        torch.cuda.empty_cache()
        db = torch.randn(db_size, dim, device='cuda', dtype=torch.float32)
        indices = torch.randint(0, db_size, (num_q, k), device='cuda', dtype=torch.int64)
        
        for _ in range(10):
            _ = psg.gather(db, indices)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = psg.gather(db, indices)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        results.append({
            'op': 'gather',
            'num_queries': num_q,
            'k': k,
            'dim': dim,
            'time_ms': elapsed / iters * 1000,
            'throughput': num_q * iters / elapsed,
        })
        print(f"Gather: {num_q}x{k}x{dim} - {elapsed/iters*1000:.2f} ms")
    return results


def benchmark_pytorch_baselines(sizes):
    """Compare against PyTorch native ops."""
    results = []
    for num_q, db_size, dim in sizes:
        q = torch.randn(num_q, dim, device='cuda', dtype=torch.float32)
        db = torch.randn(db_size, dim, device='cuda', dtype=torch.float32)
        
        q_norm = torch.nn.functional.normalize(q, dim=-1)
        db_norm = torch.nn.functional.normalize(db, dim=-1)
        
        for _ in range(5):
            _ = torch.matmul(q_norm, db_norm.t())
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch.matmul(q_norm, db_norm.t())
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        results.append({
            'op': 'pytorch_cosine',
            'num_queries': num_q,
            'db_size': db_size,
            'dim': dim,
            'time_ms': elapsed / 20 * 1000,
            'throughput': num_q * 20 / elapsed,
        })
        print(f"PyTorch CosSim: {num_q}x{db_size}x{dim} - {elapsed/20*1000:.2f} ms")
    return results


def run_full_benchmark():
    print("=" * 60)
    print("ProteinSearch-GPU Benchmark")
    print("=" * 60)
    
    all_results = []
    
    print("\n--- L2 Normalization ---")
    norm_sizes = [
        (1000, 1280), (10000, 1280), (100000, 1280),
        (1000, 2560), (10000, 2560), (100000, 2560),
    ]
    all_results.extend(benchmark_l2_normalize(norm_sizes))
    
    print("\n--- Cosine Similarity ---")
    cos_sizes = [
        (1, 10000, 1280), (1, 100000, 1280), (1, 560000, 1280),
        (8, 10000, 1280), (64, 10000, 1280), (512, 10000, 1280),
    ]
    all_results.extend(benchmark_cosine_similarity(cos_sizes))
    
    print("\n--- Top-K Selection ---")
    topk_sizes = [
        (1, 10000), (1, 100000), (1, 560000),
        (8, 10000), (64, 10000), (512, 10000),
    ]
    all_results.extend(benchmark_topk(topk_sizes))
    
    print("\n--- Candidate Gathering ---")
    gather_sizes = [
        (1, 1280), (8, 1280), (64, 1280), (512, 1280),
    ]
    all_results.extend(benchmark_gather(gather_sizes))
    
    print("\n--- PyTorch Baseline ---")
    all_results.extend(benchmark_pytorch_baselines(cos_sizes))
    
    output_path = Path("benchmarks/results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    if args.full:
        run_full_benchmark()
    elif args.quick:
        print("Running quick benchmark...")
        sizes = [(1, 10000, 1280), (8, 10000, 1280), (64, 10000, 1280)]
        benchmark_cosine_similarity(sizes)
        benchmark_topk([(1, 10000), (8, 10000), (64, 10000)])
    else:
        run_full_benchmark()


if __name__ == "__main__":
    main()