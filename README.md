# ProteinSearch-GPU

## CUDA-Accelerated Semantic Protein Retrieval with Biological Re-ranking

A high-performance semantic protein search engine that combines transformer embeddings, custom CUDA kernels, and biological sequence alignment to retrieve similar proteins from large protein databases. This project is designed to demonstrate GPU systems engineering, CUDA kernel optimization, and AI infrastructure for computational biology.

## Project Status

This repository is a research and systems-engineering prototype focused on building an exact semantic protein retrieval pipeline on GPU.

Current scope:
- CUDA extension and kernel scaffolding for core retrieval primitives
- Python package structure for encoding, benchmarking, and evaluation
- Dataset utilities for small-scale validation and future benchmarking
- Profiling and benchmarking documentation

Current priorities:
- Validate build and runtime on a CUDA-enabled environment
- Run correctness checks on a small Swiss-Prot subset
- Benchmark custom kernels against PyTorch and FAISS baselines
- Measure retrieval quality before and after biological re-ranking

## Motivation

Modern protein foundation models such as ESM-2 produce embeddings that capture structural and functional similarity beyond traditional sequence alignment. Searching these embeddings efficiently across hundreds of thousands or millions of proteins requires GPU-optimized retrieval pipelines rather than brute-force CPU computation.

ProteinSearch-GPU is an educational but production-inspired implementation of a semantic protein retrieval engine. Instead of relying only on libraries such as cuBLAS or FAISS, this project explores how core retrieval primitives can be implemented directly in CUDA to study GPU architecture, memory behavior, and kernel-level optimization.

The pipeline includes:
- Transformer-based protein embedding generation
- Custom CUDA cosine similarity computation
- Warp-level Top-K candidate selection
- Candidate compression for efficient downstream processing
- Biological re-ranking using Smith–Waterman alignment
- End-to-end benchmarking and GPU profiling

The goal is to understand how modern AI and GPU retrieval systems are built, optimized, and evaluated.

## Architecture

```text
FASTA Query
   │
   ▼
ESM-2 (PyTorch Inference)
   │
   ▼
CUDA Kernel 1: L2 Normalize
   │
   ▼
CUDA Kernel 2: Batched Cosine Similarity
   │
   ▼
CUDA Kernel 3: Warp-Level Top-K
   │
   ▼
CUDA Kernel 4: Candidate Compression
   │
   ▼
Smith–Waterman Biological Re-ranking
   │
   ▼
Ranked Protein Hits
```

## Features

- Custom CUDA kernels written from scratch
- PyTorch C++/CUDA extension
- Shared-memory optimized kernels
- Warp-level reductions using shuffle intrinsics
- Vectorized memory accesses
- Mixed-precision experimentation
- Biological sequence re-ranking
- GPU profiling with NVIDIA Nsight Compute and Nsight Systems
- End-to-end retrieval benchmark scaffolding
- Reproducible evaluation structure

## Quick Start

### Requirements

- Python 3.10+
- NVIDIA GPU with a CUDA-compatible PyTorch installation
- CUDA toolkit compatible with the installed PyTorch version
- Linux environment recommended for extension builds and profiling

### Installation

```bash
git clone https://github.com/VineetGK/ProteinSearch-GPU.git
cd ProteinSearch-GPU
pip install -r requirements.txt
pip install -e .
```

### Smoke Test

Run a basic dataset validation step before compiling or benchmarking kernels:

```bash
python test_dataset.py
```

Expected outcome:
- Dataset-loading check completes successfully
- A small sample of protein records is printed or validated
- Confirms the local environment can access the expected input format

### Benchmark Entry Point

```bash
python -m protein_search_gpu.benchmark
```

Use this on a CUDA-enabled machine after the extension build succeeds.

Recommended benchmark metadata to log with every run:
- GPU model
- CUDA version
- PyTorch version
- Database size
- Batch size
- Embedding dimension
- Precision mode

### Profiling

Kernel-level profiling:

```bash
ncu --set full python -m protein_search_gpu.benchmark
```

Pipeline-level profiling can be done with Nsight Systems once the full retrieval path is validated.

## CUDA Kernels

### Kernel 1 — L2 Normalize

Normalizes each protein embedding to unit length before similarity search.

CUDA concepts:
- Parallel reductions
- Warp shuffle intrinsics
- Shared memory
- Coalesced memory access

### Kernel 2 — Tiled Cosine Similarity

Computes cosine similarity between a batch of query embeddings and a large embedding database.

CUDA concepts:
- Shared-memory tiling
- Grid-stride loops
- Float4 vectorized loads
- Loop unrolling
- Occupancy optimization

### Kernel 3 — Warp-Level Top-K

Finds the highest scoring candidates without sorting the full similarity matrix.

CUDA concepts:
- Warp intrinsics
- Bitonic sorting
- Register-only computation
- Branch minimization

### Kernel 4 — Candidate Compression

Compacts Top-K indices and scores into contiguous buffers for downstream reranking.

CUDA concepts:
- Parallel gather
- Coalesced writes
- Prefix indexing

### Kernel 5 — GPU Smith–Waterman (Optional / Experimental)

Wavefront dynamic programming for biological sequence alignment on GPU.

CUDA concepts:
- Anti-diagonal dynamic programming
- Shared-memory banding
- Synchronization
- Dynamic programming optimization

## Technology Stack

### Languages
- CUDA C++
- C++17
- Python

### AI / ML
- PyTorch
- Hugging Face Transformers
- ESM-2

### GPU Tooling
- CUDA
- NVIDIA Nsight Compute
- NVIDIA Nsight Systems

### Bioinformatics
- UniProt
- Swiss-Prot
- UniRef50
- Pfam
- Smith–Waterman
- Parasail

## Datasets

### Development Dataset

**Swiss-Prot (reviewed)**
- Kernel debugging
- Initial benchmarking
- Functional validation

### Scaling Dataset

**UniRef50 subset**
- Large-scale throughput testing
- Memory bandwidth analysis
- Retrieval scalability experiments

### Biological Evaluation Dataset

**Pfam seed alignments**
- Recall@K
- Precision@K
- Family retrieval accuracy
- Re-ranking quality validation

## Pipeline

```text
FASTA
  ↓
ESM-2 Embedding
  ↓
L2 Normalization
  ↓
CUDA Similarity Search
  ↓
Warp Top-K
  ↓
Candidate Compression
  ↓
Smith–Waterman Re-ranking
  ↓
Final Ranked Results
```

## Repository Structure

```text
ProteinSearch-GPU/
├── cuda/
│   ├── normalize.cu
│   ├── cosine.cu
│   ├── topk.cu
│   ├── gather.cu
│   └── smith_waterman.cu
├── bindings/
│   └── torch_extension.cpp
├── python/
│   ├── encoder.py
│   ├── benchmark.py
│   ├── evaluate.py
│   └── datasets.py
├── profiling/
│   ├── nsight_compute.md
│   ├── nsight_systems.md
│   └── reports/
├── benchmarks/
├── notebooks/
├── data/
├── test_dataset.py
├── requirements.txt
├── setup.py
└── README.md
```

## Benchmark Plan

The project is designed to collect both systems-level and retrieval-quality metrics once benchmarking is complete.

### GPU Performance Metrics

- Kernel latency
- Queries per second
- Average query latency
- GPU occupancy
- SM efficiency
- Warp execution efficiency
- DRAM bandwidth
- L2 cache hit rate
- Global memory efficiency
- Achieved FLOPS

### CUDA Metrics

Typical Nsight Compute metrics include:
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `dram__bytes.sum.per_second`
- `lts__t_sectors_hit_rate`
- `smsp__sass_average_branch_targets_threads_uniform`
- `gpu__time_duration`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`

### Scalability Settings

Benchmarks should be run across multiple database sizes:
- 10K proteins for correctness checks
- 100K proteins for optimization
- 560K proteins for larger-scale benchmark runs
- 1M+ proteins for scalability evaluation

Experiments should also sweep:
- Batch = 1
- Batch = 8
- Batch = 32
- Batch = 64
- Batch = 128
- Batch = 256
- Batch = 512

## Retrieval Evaluation

Beyond raw GPU speed, the project evaluates biological retrieval quality.

Planned metrics:
- Recall@10
- Recall@50
- Precision@10
- Precision@50
- Mean Reciprocal Rank (MRR)
- nDCG@10
- Mean Average Precision (mAP)

### Biological Evaluation Goals

The retrieved proteins can be evaluated using curated annotations to measure:
- Same Pfam family retrieval
- Same UniProt annotation retrieval
- Improvement after Smith–Waterman re-ranking
- Exact match recovery
- Functional similarity

## Baselines

### GPU Baselines
- PyTorch + `torch.matmul`
- PyTorch + cosine similarity
- FAISS flat index

### CPU / Bioinformatics Baselines
- NumPy
- SciPy
- BLASTP
- MMseqs2

## Mixed Precision Experiments

The project is structured to evaluate:
- FP32
- FP16
- BF16

For each precision mode, measure:
- Throughput
- Latency
- Memory usage
- Retrieval quality
- Numerical error

## Profiling

### Nsight Compute
Used for kernel-level optimization:
- Occupancy
- Memory bandwidth
- Warp efficiency
- Register pressure
- Shared-memory utilization
- Instruction mix

### Nsight Systems
Used for pipeline-level profiling:
- CUDA kernel timeline
- CPU scheduling
- Memory copies
- Stream overlap
- Kernel launch overhead

## Current Gaps

The repository structure and documentation are in place, but benchmarking results should only be reported after validation on a CUDA-enabled environment.

Before adding headline performance claims, complete:
- Extension build verification
- Correctness checks against small reference data
- Baseline comparisons with PyTorch and FAISS
- Retrieval quality evaluation with curated annotations
- Profiling on a documented hardware/software setup

## Future Work

- Approximate nearest-neighbor search (IVF, HNSW, Product Quantization)
- Multi-GPU database sharding with NCCL
- CUDA Graphs
- Persistent CUDA kernels
- Tensor Core optimization
- Triton kernel implementation
- FP8 inference experiments
- Distributed retrieval across multiple GPUs

## Suggested Reporting Format for Final Results

Once benchmarks are complete, report results with:
- Hardware setup
- CUDA and PyTorch versions
- Dataset used
- Database size
- Batch size
- Precision mode
- Baseline comparison
- End-to-end latency
- Queries per second
- Retrieval-quality metrics

This makes the results easier to reproduce and compare fairly.

## Resume Framing After Validation

After benchmarking and validation are complete, this project can support bullets such as:
- Built a CUDA-accelerated semantic protein retrieval engine using ESM-2 embeddings and custom GPU kernels for exact similarity search over large protein databases.
- Implemented shared-memory tiled cosine similarity, warp-level Top-K selection, and candidate compression kernels optimized for memory throughput and GPU efficiency.
- Profiled GPU kernels with NVIDIA Nsight to analyze occupancy, bandwidth, and latency bottlenecks.
- Integrated biological re-ranking with Smith–Waterman alignment to improve retrieval quality while limiting expensive sequence alignment to a small candidate set.
- Benchmarked end-to-end retrieval performance against PyTorch and FAISS baselines across multiple database sizes and batch configurations.

## License

MIT License

## Acknowledgments

- Meta AI for the ESM-2 protein language models
- UniProt Consortium for Swiss-Prot and UniRef datasets
- NVIDIA for CUDA and profiling tools
- The Parasail and Striped Smith–Waterman projects for alignment algorithms