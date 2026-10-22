import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


def benchmark_l2_normalize(
    num_vectors: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    input_tensor = torch.randn(num_vectors, dim, device='cuda', dtype=dtype)
    
    import protein_search_gpu as psg
    
    for _ in range(num_warmup):
        _ = psg.l2_normalize(input_tensor)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = psg.l2_normalize(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_vectors': num_vectors,
        'dim': dim,
        'dtype': str(dtype),
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_vectors_per_sec': num_vectors * num_iters / elapsed,
        'memory_gb': input_tensor.numel() * input_tensor.element_size() / 1e9
    }


def benchmark_cosine_similarity(
    num_queries: int,
    db_size: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 10,
    num_iters: int = 50
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    queries = torch.randn(num_queries, dim, device='cuda', dtype=dtype)
    database = torch.randn(db_size, dim, device='cuda', dtype=dtype)
    
    import protein_search_gpu as psg
    
    for _ in range(num_warmup):
        _ = psg.cosine_similarity(queries, database)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = psg.cosine_similarity(queries, database)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_queries': num_queries,
        'db_size': db_size,
        'dim': dim,
        'dtype': str(dtype),
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_queries_per_sec': num_queries * num_iters / elapsed,
        'memory_gb': (queries.numel() + database.numel()) * queries.element_size() / 1e9
    }


def benchmark_topk(
    num_queries: int,
    db_size: int,
    k: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    similarities = torch.randn(num_queries, db_size, device='cuda', dtype=torch.float32)
    
    import protein_search_gpu as psg
    
    for _ in range(num_warmup):
        _ = psg.topk(similarities, k)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = psg.topk(similarities, k)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_queries': num_queries,
        'db_size': db_size,
        'k': k,
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_queries_per_sec': num_queries * num_iters / elapsed
    }


def benchmark_gather(
    num_queries: int,
    k: int,
    dim: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    database = torch.randn(100000, dim, device='cuda', dtype=torch.float32)
    indices = torch.randint(0, 100000, (num_queries, k), device='cuda', dtype=torch.int64)
    
    import protein_search_gpu as psg
    
    for _ in range(num_warmup):
        _ = psg.gather(database, indices)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = psg.gather(database, indices)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_queries': num_queries,
        'k': k,
        'dim': dim,
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_queries_per_sec': num_queries * num_iters / elapsed
    }


def benchmark_smith_waterman(
    num_queries: int,
    num_db: int,
    max_len: int = 512,
    num_warmup: int = 5,
    num_iters: int = 20
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    query_seqs = torch.randint(0, 24, (num_queries, max_len), device='cuda', dtype=torch.int8)
    db_seqs = torch.randint(0, 24, (num_db, max_len), device='cuda', dtype=torch.int8)
    query_lens = torch.full((num_queries,), max_len, device='cuda', dtype=torch.int64)
    db_lens = torch.full((num_db,), max_len, device='cuda', dtype=torch.int64)
    
    import protein_search_gpu as psg
    
    for _ in range(num_warmup):
        _ = psg.smith_waterman(query_seqs, db_seqs, query_lens, db_lens, 11, 1)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = psg.smith_waterman(query_seqs, db_seqs, query_lens, db_lens, 11, 1)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_queries': num_queries,
        'num_db': num_db,
        'max_len': max_len,
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_pairs_per_sec': num_queries * num_db * num_iters / elapsed
    }


def benchmark_pytorch_cosine(
    num_queries: int,
    db_size: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 10,
    num_iters: int = 50
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    queries = torch.randn(num_queries, dim, device='cuda', dtype=dtype)
    database = torch.randn(db_size, dim, device='cuda', dtype=dtype)
    
    queries_norm = torch.nn.functional.normalize(queries, dim=-1)
    database_norm = torch.nn.functional.normalize(database, dim=-1)
    
    for _ in range(num_warmup):
        _ = torch.matmul(queries_norm, database_norm.t())
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = torch.matmul(queries_norm, database_norm.t())
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_queries': num_queries,
        'db_size': db_size,
        'dim': dim,
        'dtype': str(dtype),
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_queries_per_sec': num_queries * num_iters / elapsed
    }


def benchmark_pytorch_topk(
    num_queries: int,
    db_size: int,
    k: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    
    similarities = torch.randn(num_queries, db_size, device='cuda', dtype=torch.float32)
    
    for _ in range(num_warmup):
        _ = torch.topk(similarities, k, dim=1)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = torch.topk(similarities, k, dim=1)
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    per_iter = elapsed / num_iters * 1000
    
    return {
        'num_queries': num_queries,
        'db_size': db_size,
        'k': k,
        'total_time_ms': elapsed * 1000,
        'per_iter_ms': per_iter,
        'throughput_queries_per_sec': num_queries * num_iters / elapsed
    }


def run_full_benchmark(
    db_sizes: List[int] = [10000, 100000, 500000],
    batch_sizes: List[int] = [1, 8, 32, 64, 128, 256, 512],
    dim: int = 1280,
    k: int = 10,
    output_path: str = "benchmark_results.json"
):
    results = {}
    
    for db_size in db_sizes:
        print(f"\n=== Database size: {db_size} ===")
        results[db_size] = {}
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # No try/except here: a missing extension or a kernel error should
            # fail loudly instead of producing an empty result row.
            cos_result = benchmark_cosine_similarity(batch_size, db_size, dim)
            print(f"    CUDA cosine: {cos_result['per_iter_ms']:.2f} ms")
            
            pytorch_cos_result = benchmark_pytorch_cosine(batch_size, db_size, dim)
            print(f"    PyTorch cosine: {pytorch_cos_result['per_iter_ms']:.2f} ms")
            
            topk_result = benchmark_topk(batch_size, db_size, k)
            print(f"    CUDA topk: {topk_result['per_iter_ms']:.2f} ms")
            
            pytorch_topk_result = benchmark_pytorch_topk(batch_size, db_size, k)
            print(f"    PyTorch topk: {pytorch_topk_result['per_iter_ms']:.2f} ms")
            
            results[db_size][batch_size] = {
                'cuda_cosine': cos_result,
                'pytorch_cosine': pytorch_cos_result,
                'cuda_topk': topk_result,
                'pytorch_topk': pytorch_topk_result
            }
            
            torch.cuda.empty_cache()
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


def profile_memory():
    import torch.cuda.memory as memory
    
    db_size = 100000
    dim = 1280
    batch_size = 64
    
    queries = torch.randn(batch_size, dim, device='cuda')
    database = torch.randn(db_size, dim, device='cuda')
    
    torch.cuda.reset_peak_memory_stats()
    
    import protein_search_gpu as psg
    _ = psg.cosine_similarity(queries, database)
    torch.cuda.synchronize()
    
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory: {peak:.2f} GB")
    
    return peak


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark ProteinSearch-GPU")
    parser.add_argument("--db-sizes", nargs="+", type=int, default=[10000, 100000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--k", type=int, default=10, help="top-k (custom kernel supports k <= 32)")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--memory-only", action="store_true")
    
    args = parser.parse_args()
    
    if args.memory_only:
        profile_memory()
    else:
        run_full_benchmark(
            db_sizes=args.db_sizes,
            batch_sizes=args.batch_sizes,
            dim=args.dim,
            k=args.k,
            output_path=args.output
        )