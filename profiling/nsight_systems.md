# NVIDIA Nsight Systems Profiling Guide

## Overview

Nsight Systems provides system-wide profiling to visualize:
- CUDA kernel timeline
- CPU-GPU synchronization
- Memory transfers (H2D, D2H, D2D)
- Stream overlap and concurrency
- Kernel launch overhead

## Basic Usage

### CLI Profiling

```bash
# Basic profile
nsys profile --stats=true python benchmark.py

# With CUDA trace
nsys profile --trace=cuda,nvtx --stats=true python benchmark.py

# With output file
nsys profile -o profile_report --trace=cuda,nvtx python benchmark.py
```

### Python API Profiling

```python
import torch.cuda.profiler as profiler

profiler.start()
# Your code here
profiler.stop()
```

## Analyzing ProteinSearch-GPU Pipeline

### End-to-End Pipeline Trace

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o pipeline_profile \
  python -c "
import protein_search_gpu as psg
import torch

# Setup
queries = torch.randn(64, 1280, device='cuda')
database = torch.randn(100000, 1280, device='cuda')

# Warmup
_ = psg.cosine_similarity(queries, database)
_ = psg.l2_normalize(queries)

# Profile region
torch.cuda.profiler.start()
normed = psg.l2_normalize(queries)
sims = psg.cosine_similarity(normed, database)
topk_vals, topk_idx = psg.topk(sims, 10)
candidates = psg.gather(database, topk_idx)
torch.cuda.profiler.stop()
"
```

### Key Metrics to Analyze

| Metric | Target | Analysis |
|--------|--------|----------|
| Kernel Duration | < 1ms per kernel | Check individual kernel times |
| H2D/D2H Transfers | Minimal | Should only be for initial data load |
| Stream Overlap | High | Kernels should overlap with copies |
| Kernel Launch Overhead | < 10μs | Check gaps between kernels |
| SM Utilization | > 80% | Check in GPU utilization row |

### Timeline Analysis

In Nsight Systems GUI:
1. **GPU Rows**: Look for gaps between kernels
2. **CUDA Streams**: Verify concurrent execution
3. **Memory Operations**: Check for unnecessary copies
4. **OS Runtime**: Check for CPU bottlenecks

## NVTX Annotations

Add custom markers to identify pipeline stages:

```python
import torch.cuda.nvtx as nvtx

with nvtx.range("L2 Normalize"):
    normed = psg.l2_normalize(queries)

with nvtx.range("Cosine Similarity"):
    sims = psg.cosine_similarity(normed, database)

with nvtx.range("Top-K Selection"):
    topk_vals, topk_idx = psg.topk(sims, 10)

with nvtx.range("Candidate Gather"):
    candidates = psg.gather(database, topk_idx)
```

## Analyzing Batch Processing

```bash
# Profile different batch sizes
for batch in 1 8 32 64 128 256 512; do
  nsys profile -o batch_${batch} \
    python benchmark.py --batch-size $batch --db-size 100000
done
```

Compare:
- Total elapsed time
- Kernel concurrency
- Memory throughput scaling

## Multi-GPU Profiling

```bash
nsys profile --trace=cuda --gpu-metrics-device=all \
  python multi_gpu_benchmark.py
```

Check:
- NCCL communication patterns
- GPU-to-GPU transfers
- Load balancing across GPUs

## CUDA Graph Profiling

```python
# Capture CUDA graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    normed = psg.l2_normalize(queries)
    sims = psg.cosine_similarity(normed, database)
    topk_vals, topk_idx = psg.topk(sims, 10)

# Profile graph replay
nsys profile --trace=cuda python -c "
import protein_search_gpu as psg
import torch

graph = torch.cuda.CUDAGraph()
# ... capture ...

for _ in range(100):
    graph.replay()
"
```

## Performance Comparison

### Compare with PyTorch Baseline

`benchmarks/benchmark.py` runs the custom kernels and the PyTorch
(`torch.matmul`) baseline in the same process, so a single trace shows both.
A FAISS baseline is not implemented.

```bash
nsys profile -o custom_vs_pytorch python benchmarks/benchmark.py --quick
```

### Generate Comparison Report

```bash
nsys stats --report gputrace custom_vs_pytorch.nsys-rep > gputrace.txt
```

## Common Issues

### 1. Kernel Launch Overhead
**Symptom**: Many small kernels with large gaps
**Fix**: Batch operations, use persistent kernels, CUDA Graphs

### 2. Poor Stream Overlap
**Symptom**: Serial execution on single stream
**Fix**: Use multiple streams, async copies

### 3. Excessive H2D/D2H Transfers
**Symptom**: Large memory copy blocks
**Fix**: Keep data on GPU, use pinned memory

### 4. Low GPU Utilization
**Symptom**: Large gaps in GPU timeline
**Fix**: Increase batch size, optimize kernel launch config

## Automated Analysis Script

```python
# scripts/analyze_nsys.py
import subprocess
import json

def analyze_profile(report_path):
    # Extract key metrics
    result = subprocess.run([
        'nsys', 'stats', '--report', 'gputrace',
        '--format', 'csv', report_path
    ], capture_output=True, text=True)
    
    # Parse and summarize
    return parse_stats(result.stdout)

if __name__ == '__main__':
    import sys
    metrics = analyze_profile(sys.argv[1])
    print(json.dumps(metrics, indent=2))
```

## CI Integration

```yaml
# .github/workflows/profile.yml
- name: Profile Pipeline
  run: |
    nsys profile -o pipeline_profile \
      --trace=cuda,nvtx \
      --stats=true \
      python benchmark.py --quick
    
    nsys stats --report gputrace pipeline_profile.nsys-rep > profile_summary.txt
    cat profile_summary.txt
    
- name: Upload Profile
  uses: actions/upload-artifact@v3
  with:
    name: nsys-profile
    path: pipeline_profile.nsys-rep
```

## References

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [NVTX API](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-annotation)