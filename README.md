protein-semsearch-gpu

### CUDA-Accelerated Semantic Protein Retrieval with Biological Re-ranking

> A semantic protein search engine that combines transformer embeddings, custom CUDA kernels, and biological sequence alignment to retrieve similar proteins. Built to learn GPU systems engineering, CUDA kernel design, and AI infrastructure for computational biology.

**Status:** early stage. The kernels are deliberately simple, correctness-first implementations with tests against PyTorch references. They are not yet tuned, and no performance or retrieval numbers have been measured yet (see [Results](#results)).

---

## Motivation

Modern protein foundation models such as ESM-2 produce embeddings that capture structural and functional similarity beyond traditional sequence alignment. Searching these embeddings across hundreds of thousands or millions of proteins calls for GPU retrieval pipelines rather than brute-force CPU computation.

ProteinSearch-GPU is an educational implementation of an exact semantic protein search engine. Instead of relying solely on existing libraries such as cuBLAS or FAISS, this project implements the core retrieval primitives directly in CUDA to explore GPU architecture, memory behaviour, and kernel design.

The pipeline performs:

* Transformer-based protein embedding generation (ESM-2)
* L2 normalization and dot-product (cosine) similarity in custom CUDA kernels
* Block-level Top-K candidate selection (k ≤ 32)
* Biological re-ranking using Smith–Waterman alignment
* Kernel benchmarking and GPU profiling guides

---

# Architecture

```text
                     FASTA Query
                          │
                          ▼
              ESM-2 (PyTorch Inference)
                          │
                          ▼
              CUDA Kernel 1
                L2 Normalize
                          │
                          ▼
              CUDA Kernel 2
     Batched Dot-Product (Cosine) Similarity
                          │
                          ▼
              CUDA Kernel 3
          Block-Level Top-K (k ≤ 32)
                          │
                          ▼
              CUDA Kernel 4
             Candidate Gather
        (embeddings of the Top-K rows)
                          │
                          ▼
      Smith–Waterman Biological Re-ranking
                          │
                          ▼
                Ranked Protein Hits
```

---

# Features

* Custom CUDA kernels exposed as a PyTorch C++/CUDA extension
* Warp-shuffle reductions (L2 norm, Top-K merge)
* float4 vectorized loads when the dimension is a multiple of 4 and pointers are 16-byte aligned, with a scalar fallback otherwise
* FP32, FP16 and BF16 inputs for normalize, similarity and gather (always fp32 accumulation)
* Kernels launch on PyTorch's current CUDA stream, with input validation and launch error checks
* Smith–Waterman (BLOSUM62, affine gaps) for biological re-ranking
* Retrieval metrics: Recall@K, Precision@K, MRR, nDCG@K, mAP
* Correctness tests against PyTorch / pure-Python references (`tests/test_ops.py`)
* Profiling guides for Nsight Compute and Nsight Systems

---

# CUDA Kernels

All ops live in `protein_search_gpu._C` and are re-exported as `protein_search_gpu.<op>`.

## Kernel 1 — L2 Normalize (`l2_normalize`, `l2_normalize_inplace`)

Normalizes each row to unit length: `x / max(||x||, 1e-12)`, matching `torch.nn.functional.normalize(x, dim=1)`.

* One block per row
* Warp-shuffle block reduction
* float4 path for aligned fp32 rows

---

## Kernel 2 — Dot-Product Similarity (`cosine_similarity`)

Computes `query @ database.T` in fp32. This is **cosine similarity only when both inputs are already L2-normalized**, so run `l2_normalize` first.

* One thread per (query, database row) pair, fp32 accumulation
* float4 loads for aligned fp32 inputs
* Not shared-memory tiled. `torch.matmul` (cuBLAS) is expected to be faster; a tiled GEMM-style kernel is future work.

---

## Kernel 3 — Block-Level Top-K (`topk`, k ≤ 32)

Finds the k highest scores per row without sorting the whole row.

* One block per query row
* Each thread keeps a small sorted candidate list for its strided slice of the row
* The block merges the lists with k rounds of warp-shuffle argmax
* Values match `torch.topk(sorted=True)`. Ties go to the smaller index. NaN ranks highest, as in `torch.topk`.
* `k > 32` is rejected; use `torch.topk` for larger k

---

## Kernel 4 — Candidate Gather (`gather`)

Copies the embeddings of the Top-K indices into a contiguous `[num_queries, k, dim]` buffer.

* One block per output row, float4 copies when aligned
* Out-of-range indices (for example `-1` padding) produce zero rows

---

## Kernel 5 — GPU Smith–Waterman (`smith_waterman`, `smith_waterman_batch`)

Local alignment scores for re-ranking candidates.

* Inputs are int8 residue indices from `datasets.sequence_to_indices` (alphabet `ARNDCQEGHILKMFPSTWYVXUOBZJ`), plus per-row lengths
* BLOSUM62 scoring (U, O and J are scored as X), stored in constant memory
* Affine gaps (Gotoh): a gap of length L costs `gap_open + (L - 1) * gap_extend`. The defaults are 11/1, and the convention matches parasail and Biopython's `PairwiseAligner`.
* `smith_waterman` scores all query×database pairs; `smith_waterman_batch` scores selected `(query, db)` pairs given as `pair_indices [num_pairs, 2]`
* One thread per pair, with a linear-memory DP: one row of H/E per thread in a global scratch buffer, interleaved for coalescing
* Not a striped or anti-diagonal wavefront kernel; those are future work

---

# Technology Stack

## Languages

* CUDA C++
* C++17
* Python

## AI

* PyTorch
* Hugging Face Transformers
* ESM-2

## GPU

* CUDA
* Nsight Compute
* Nsight Systems

## Biology

* UniProt
* Swiss-Prot
* UniRef50
* Pfam
* Smith–Waterman

---

# Installation

Requirements: an NVIDIA GPU, a **CUDA build of PyTorch ≥ 2.0**, and the matching CUDA toolkit (`nvcc`; `CUDA_HOME` must be set or `nvcc` must be on `PATH`). The build fails with an error if the toolkit is missing. It does not silently skip the extension.

```bash
pip install torch            # CUDA build matching your toolkit
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

`--no-build-isolation` is needed because `setup.py` imports PyTorch to build the extension.

## Usage

```python
import torch
import protein_search_gpu as psg

db = psg.l2_normalize(torch.randn(100_000, 1280, device="cuda"))
q = psg.l2_normalize(torch.randn(64, 1280, device="cuda"))

sims = psg.cosine_similarity(q, db)          # [64, 100000] float32
values, indices = psg.topk(sims, 10)         # k <= 32
candidates = psg.gather(db, indices)         # [64, 10, 1280]
```

## Tests

```bash
pytest tests/
```

The op tests are skipped automatically when no CUDA device is present or the extension has not been built.

---

# Datasets

## Development

Swiss-Prot (Reviewed)

Purpose:

* Kernel debugging
* Initial benchmarking
* Functional validation

---

## Scaling

UniRef50 Subset

Purpose:

* Large-scale throughput testing
* Memory bandwidth analysis

---

## Biological Evaluation

Pfam Seed Alignments

Purpose:

* Recall@K
* Precision@K
* Family retrieval accuracy

---

# Pipeline

```text
FASTA

↓

ESM-2 Embedding

↓

L2 Normalization

↓

CUDA Similarity Search

↓

Top-K Selection

↓

Candidate Gather

↓

Smith–Waterman Re-ranking

↓

Final Results
```

---

# Repository Structure

```text
ProteinSearch-GPU/

├── cuda/
│   ├── common.cuh
│   ├── normalize.cu
│   ├── cosine.cu
│   ├── topk.cu
│   ├── gather.cu
│   └── smith_waterman.cu
│
├── bindings/
│   └── torch_extension.cpp
│
├── python/
│   └── protein_search_gpu/
│       ├── __init__.py
│       ├── encoder.py
│       ├── benchmark.py
│       ├── evaluate.py
│       ├── datasets.py
│       └── fasta.py
│
├── tests/
│   └── test_ops.py
│
├── benchmarks/
│   └── benchmark.py
│
├── profiling/
│   ├── nsight_compute.md
│   └── nsight_systems.md
│
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Benchmarks

`benchmarks/benchmark.py` times the custom kernels and the PyTorch baselines (`torch.matmul`, `torch.topk`):

```bash
python benchmarks/benchmark.py --quick   # or --full
```

## GPU Performance

These metrics are planned. They can be collected with NVIDIA Nsight Compute; none have been measured yet.

| Metric                    | Description                          |
| ------------------------- | ------------------------------------ |
| Kernel latency            | Execution time per kernel            |
| Queries/sec               | End-to-end throughput                |
| Average query latency     | Time per query                       |
| GPU occupancy             | Percentage of active warps           |
| SM efficiency             | Streaming multiprocessor utilization |
| Warp execution efficiency | Active lanes per warp                |
| DRAM bandwidth            | Memory throughput                    |
| L2 cache hit rate         | Cache effectiveness                  |
| Global load efficiency    | Memory coalescing                    |
| Achieved FLOPS            | Floating-point throughput            |

---

## CUDA Metrics

Can be collected with:

```bash
ncu --set full python benchmarks/benchmark.py --quick
```

Relevant metrics include:

* sm__warps_active.avg.pct_of_peak_sustained_active
* dram__bytes.sum.per_second
* lts__t_sectors_hit_rate
* smsp__sass_average_branch_targets_threads_uniform
* gpu__time_duration
* sm__throughput.avg.pct_of_peak_sustained_elapsed

---

## Scalability

Planned database sizes:

| Database Size | Purpose                    |
| ------------- | -------------------------- |
| 10K proteins  | Correctness                |
| 100K proteins | Optimization               |
| 560K proteins | Production-scale benchmark |
| 1M+ proteins  | Scalability evaluation     |

---

## Batch Sizes

Planned batch sizes:

* Batch = 1
* Batch = 8
* Batch = 32
* Batch = 64
* Batch = 128
* Batch = 256
* Batch = 512

---

# Retrieval Evaluation

Beyond raw GPU performance, ProteinSearch-GPU measures biological retrieval quality (`python/protein_search_gpu/evaluate.py`).

Metrics are averaged over queries that have at least one relevant item:

* Recall@10
* Recall@50
* Precision@10
* Precision@50
* Mean Reciprocal Rank (MRR)
* nDCG@10
* Mean Average Precision (mAP)

---

# Biological Evaluation

Planned experiments, using curated annotations:

* Same Pfam family retrieval
* Same UniProt annotation retrieval
* Improvement after Smith–Waterman re-ranking
* Exact match recovery
* Functional similarity

---

# Baselines

## Implemented

* PyTorch `torch.matmul` on normalized embeddings
* PyTorch `torch.topk`

## Planned (not implemented)

* FAISS (Flat Index)
* NumPy / SciPy CPU search
* BLASTP and MMseqs2 (retrieval quality and speed references)

---

# Mixed Precision

Normalize, similarity and gather accept FP32, FP16 and BF16 inputs with fp32 accumulation. Top-K takes fp32 scores. The benchmarks currently run FP32 only; FP16/BF16 throughput, memory, retrieval quality and numerical-error comparisons are not yet measured.

---

# Profiling

See `profiling/nsight_compute.md` and `profiling/nsight_systems.md`.

## Nsight Compute

Kernel-level analysis

* Occupancy
* Memory bandwidth
* Warp efficiency
* Register pressure
* Shared-memory utilization
* Instruction mix

## Nsight Systems

Pipeline-level analysis

* CUDA kernel timeline
* CPU scheduling
* Memory copies
* Stream overlap
* Kernel launch overhead

---

# Future Work

* Shared-memory tiled / Tensor Core similarity kernel
* Warp-level bitonic Top-K and support for k > 32
* Striped or anti-diagonal wavefront Smith–Waterman
* FAISS baseline
* Approximate nearest-neighbor search (IVF, HNSW, Product Quantization)
* Multi-GPU database sharding with NCCL
* CUDA Graphs
* Persistent CUDA kernels
* Triton kernel implementation
* FP8 inference experiments

---

# Results

**Not yet measured.** Every value below is a placeholder until benchmarks are run.

## Throughput (not yet measured)

| Database | Batch | Queries/sec | Avg Latency |
| -------- | ----- | ----------: | ----------: |
| 100K     | 1     |           — |           — |
| 100K     | 64    |           — |           — |
| 100K     | 512   |           — |           — |
| 560K     | 64    |           — |           — |
| 560K     | 512   |           — |           — |

---

## GPU Metrics (not yet measured)

| Metric             | Result |
| ------------------ | -----: |
| Occupancy          |      — |
| DRAM Bandwidth     | — GB/s |
| L2 Cache Hit Rate  |    — % |
| Warp Efficiency    |    — % |
| Kernel Time        |   — ms |
| End-to-End Latency |   — ms |
| GPU Memory Usage   |   — GB |

---

## Retrieval Metrics (not yet measured)

| Metric       | Result |
| ------------ | -----: |
| Recall@10    |      — |
| Recall@50    |      — |
| Precision@10 |      — |
| MRR          |      — |
| nDCG@10      |      — |
| mAP          |      — |

---

# License

MIT License. See [LICENSE](LICENSE).

---

# Acknowledgments

* Meta AI for the ESM-2 protein language models.
* UniProt Consortium for Swiss-Prot and UniRef datasets.
* NVIDIA for CUDA, Nsight Compute, and GPU profiling tools.
* The Parasail and Striped Smith–Waterman projects for sequence alignment algorithms.
