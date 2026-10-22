# NVIDIA Nsight Compute Profiling Guide

## Overview

This guide covers profiling the ProteinSearch-GPU CUDA kernels using NVIDIA Nsight Compute (NCU).

## Prerequisites

- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- CUDA Toolkit 11.0+
- Nsight Compute (`ncu` or `nv-nsight-cu-cli`)
- Python environment with `protein_search_gpu` installed

## Installation

```bash
# Ubuntu
sudo apt install nvidia-nsight-compute

# Or download from NVIDIA developer portal
```

## Profiling Commands

### Basic Kernel Profiling

```bash
# Profile all kernels in benchmark
ncu --set full python benchmark.py --quick

# Profile specific kernel with detailed metrics
ncu --metrics all --kernel-name "cosine_similarity*" python benchmark.py --quick

# Profile with specific metrics
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  dram__bytes.sum.per_second,\
  lts__t_sectors_hit_rate,\
  gpu__time_duration.sum \
  python benchmark.py --quick
```

### Key Metrics to Collect

| Metric | Description | Target |
|--------|-------------|--------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | SM occupancy | > 80% |
| `dram__bytes.sum.per_second` | DRAM bandwidth | > 80% peak |
| `lts__t_sectors_hit_rate` | L2 cache hit rate | > 90% |
| `gpu__time_duration.sum` | Kernel elapsed cycles | Minimize |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM throughput | > 70% |
| `smsp__sass_average_branch_targets_threads_uniform` | Branch efficiency | ~1.0 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global load sectors | Monitor |

### Profiling Each Kernel

#### L2 Normalize Kernel
```bash
ncu --kernel-name "l2_normalize*" \
  --metrics "sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second" \
  python -c "
import torch, protein_search_gpu as psg
x = torch.randn(10000, 1280, device='cuda')
for _ in range(10): psg.l2_normalize(x)
"
```

#### Cosine Similarity Kernel
```bash
ncu --kernel-name "cosine_similarity*" \
  --metrics "sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second,lts__t_sectors_hit_rate" \
  python -c "
import torch, protein_search_gpu as psg
q = torch.randn(64, 1280, device='cuda')
db = torch.randn(100000, 1280, device='cuda')
for _ in range(10): psg.cosine_similarity(q, db)
"
```

#### Top-K Kernel
```bash
ncu --kernel-name "topk*" \
  --metrics "sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum" \
  python -c "
import torch, protein_search_gpu as psg
sims = torch.randn(64, 100000, device='cuda')
for _ in range(10): psg.topk(sims, 32)  # custom kernel supports k <= 32
"
```

#### Gather Kernel
```bash
ncu --kernel-name "gather*" \
  --metrics "dram__bytes.sum.per_second,lts__t_sectors_hit_rate" \
  python -c "
import torch, protein_search_gpu as psg
db = torch.randn(100000, 1280, device='cuda')
indices = torch.randint(0, 100000, (64, 100), device='cuda', dtype=torch.int64)
for _ in range(10): psg.gather(db, indices)
"
```

#### Smith-Waterman Kernel
```bash
ncu --kernel-name "smith_waterman*" \
  --metrics "sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum" \
  python -c "
import torch, protein_search_gpu as psg
q = torch.randint(0, 24, (10, 512), device='cuda', dtype=torch.int8)
db = torch.randint(0, 24, (1000, 512), device='cuda', dtype=torch.int8)
ql = torch.full((10,), 512, device='cuda', dtype=torch.int64)
dl = torch.full((1000,), 512, device='cuda', dtype=torch.int64)
for _ in range(5): psg.smith_waterman(q, db, ql, dl, 11, 1)
"
```

## Analyzing Results

### Occupancy Analysis

Check the **Occupancy** section in NCU report:
- **Theoretical Occupancy**: 100% (ideal)
- **Achieved Occupancy**: Should be > 80%
- **Limiting Factor**: Registers, Shared Memory, or Block Limit

Common issues:
- Too many registers → Reduce register pressure with loop unrolling
- Too much shared memory → Reduce tile size
- Small grid size → Increase batch size

### Memory Throughput Analysis

Check **Memory Throughput** section:
- **DRAM Throughput**: Should approach peak bandwidth
- **L2 Cache Hit Rate**: Should be > 90% for tiled kernels
- **Global Memory Load/Store Efficiency**: Should be > 80%

### Compute Throughput

Check **Compute Throughput**:
- **SM Throughput**: Should be > 70%
- **Tensor Core Utilization**: For FP16/BF16 kernels
- **FP32/FP64 Utilization**: For FP32 kernels

## Optimization Checklist

Current state of the kernels (simple, correctness-first implementations):

### For Cosine Similarity (dot product of L2-normalized inputs)
- [x] float4 loads when dim % 4 == 0 and pointers are 16-byte aligned (scalar fallback otherwise)
- [ ] Shared-memory tiling / GEMM-style blocking (not implemented; one thread per query-row pair)
- [ ] Coalesced access patterns (check NCU; database rows are read per thread)

### For Top-K (k <= 32)
- [x] Per-thread sorted candidate lists
- [x] Block-level merge via warp-shuffle argmax (k rounds)
- [ ] Warp-level bitonic sort (not implemented)

### For L2 Normalize
- [x] float4 loads (when aligned)
- [x] Warp-shuffle block reduction

### For Smith-Waterman
- [x] One thread per pair, linear-memory DP rows in global scratch
- [ ] Wavefront / striped parallelism (not implemented)

## Reporting

Generate HTML report:
```bash
ncu --set full --export profile_report --target-processes all python benchmark.py
```

View in Nsight Compute GUI:
```bash
nv-nsight-cu profile_report.ncu-rep
```

## CI Integration

```yaml
# .github/workflows/profile.yml
- name: Profile Kernels
  run: |
    ncu --set full --csv --log-file profile.csv python benchmark.py --quick
    python scripts/parse_ncu.py profile.csv
```

## Troubleshooting

### "Profiling not supported on this device"
- Ensure GPU is not in WDDM mode (Windows) or MIG mode
- Use Linux for best profiling support

### "Kernel not found"
- Check kernel name with `--list-kernels`
- Use regex patterns: `--kernel-name "cosine*"`

### High overhead
- Reduce iteration count
- Use `--sample-period` for sampling mode

## References

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)