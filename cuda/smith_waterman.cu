// Smith-Waterman local alignment scores with BLOSUM62 and affine gaps
// (Gotoh recurrence).
//
// Sequences are int8 residue indices in the alphabet of
// protein_search_gpu.datasets.AA_TO_IDX:
//   A R N D C Q E G H I L K M F P S T W Y V X U O B Z J   (0..25)
// U, O and J are not in BLOSUM62 and are scored like X. Out-of-range values
// are also treated as X.
//
// Gap model: a gap of length L costs gap_open + (L - 1) * gap_extend
// (the same convention as parasail and Biopython's PairwiseAligner with
// open_gap_score = -gap_open, extend_gap_score = -gap_extend).
//
// Design: one thread per (query, database) pair, linear-memory DP that keeps
// a single row of H and E (vertical-gap) scores in a global scratch buffer.
// The scratch is interleaved across threads so that accesses are coalesced.
// This is a straightforward implementation, not a striped/wavefront kernel.

#include "common.cuh"

namespace psg {
namespace {

constexpr int kAlphabetSize = 26;
constexpr int kUnknownResidue = 20;  // 'X'
constexpr int kNegInf = -(1 << 29);  // far below any reachable score, no overflow
constexpr int kSwThreads = 128;
constexpr int64_t kScratchBudgetBytes = int64_t(256) << 20;  // 256 MiB

// BLOSUM62 (NCBI), rows/columns in AA_TO_IDX order. Lives in constant memory
// and is initialised at module load, so there is no per-call upload.
__constant__ int8_t kBlosum62[kAlphabetSize][kAlphabetSize] = {
    /* A */ {  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0,  0,  0,  0, -2, -1,  0},
    /* R */ { -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1, -1, -1, -1,  0, -1},
    /* N */ { -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, -1, -1, -1,  3,  0, -1},
    /* D */ { -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, -1, -1, -1,  4,  1, -1},
    /* C */ {  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -2, -2, -2, -3, -3, -2},
    /* Q */ { -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, -1, -1, -1,  0,  3, -1},
    /* E */ { -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, -1, -1, -1,  1,  4, -1},
    /* G */ {  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -1, -1, -1, -2, -1},
    /* H */ { -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, -1, -1, -1,  0,  0, -1},
    /* I */ { -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -1, -1, -1, -3, -3, -1},
    /* L */ { -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -1, -1, -1, -4, -3, -1},
    /* K */ { -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, -1, -1, -1,  0,  1, -1},
    /* M */ { -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -1, -1, -1, -3, -1, -1},
    /* F */ { -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -1, -1, -1, -3, -3, -1},
    /* P */ { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -2, -2, -2, -1, -2},
    /* S */ {  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0,  0,  0,  0},
    /* T */ {  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,  0,  0,  0, -1, -1,  0},
    /* W */ { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -2, -2, -2, -4, -3, -2},
    /* Y */ { -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -1, -1, -1, -3, -2, -1},
    /* V */ {  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -1, -1, -1, -3, -2, -1},
    /* X */ {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -1, -1, -1},
    /* U */ {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -1, -1, -1},
    /* O */ {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -1, -1, -1},
    /* B */ { -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3, -1, -1, -1,  4,  1, -1},
    /* Z */ { -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2, -1, -1, -1,  1,  4, -1},
    /* J */ {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -1, -1, -1},
};

__device__ __forceinline__ int residue(int8_t r) {
  return (r >= 0 && r < kAlphabetSize) ? static_cast<int>(r) : kUnknownResidue;
}

__device__ __forceinline__ int64_t clamp_len(int64_t len, int64_t width) {
  return len < 0 ? 0 : (len > width ? width : len);
}

// pairs == nullptr: all pairs, pair p = (p / num_db, p % num_db).
// pairs != nullptr: pair p = (pairs[2p], pairs[2p + 1]).
// scratch holds 2 * db_width ints per worker thread, interleaved:
//   H[j] of worker w at scratch[j * workers + w],
//   E[j] of worker w at scratch[(db_width + j) * workers + w].
__global__ void smith_waterman_kernel(
    const int8_t* __restrict__ query_seqs, int64_t query_width,
    const int64_t* __restrict__ query_lens,
    const int8_t* __restrict__ db_seqs, int64_t db_width,
    const int64_t* __restrict__ db_lens, const int64_t* __restrict__ pairs,
    int64_t num_db, int64_t num_pairs, int gap_open, int gap_extend,
    int* __restrict__ scratch, float* __restrict__ scores) {
  const int64_t workers = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int* H = scratch + w;
  int* E = scratch + db_width * workers + w;

  for (int64_t p = w; p < num_pairs; p += workers) {
    int64_t qi, di;
    if (pairs != nullptr) {
      qi = pairs[2 * p];
      di = pairs[2 * p + 1];
    } else {
      qi = p / num_db;
      di = p % num_db;
    }
    const int8_t* q = query_seqs + qi * query_width;
    const int8_t* d = db_seqs + di * db_width;
    const int64_t qlen = clamp_len(query_lens[qi], query_width);
    const int64_t dlen = clamp_len(db_lens[di], db_width);

    // Row 0: H = 0, E = -inf.
    for (int64_t j = 0; j < dlen; ++j) {
      H[j * workers] = 0;
      E[j * workers] = kNegInf;
    }

    int best = 0;
    for (int64_t i = 0; i < qlen; ++i) {
      const int8_t* subst = kBlosum62[residue(q[i])];
      int diag = 0;      // H[i-1][j-1]
      int left = 0;      // H[i][j-1]
      int f = kNegInf;   // F[i][j-1] (horizontal gap)
      for (int64_t j = 0; j < dlen; ++j) {
        const int up = H[j * workers];  // H[i-1][j]
        const int e = max(E[j * workers] - gap_extend, up - gap_open);
        f = max(f - gap_extend, left - gap_open);
        const int h = max(max(0, diag + subst[residue(d[j])]), max(e, f));
        H[j * workers] = h;
        E[j * workers] = e;
        diag = up;
        left = h;
        best = max(best, h);
      }
    }
    scores[p] = static_cast<float>(best);
  }
}

void check_sequences(const at::Tensor& seqs, const at::Tensor& lens,
                     const char* seq_name, const char* len_name) {
  check_cuda(seqs, seq_name);
  check_cuda(lens, len_name);
  TORCH_CHECK(seqs.dim() == 2, seq_name, " must be 2D [num_seqs, max_len]");
  TORCH_CHECK(seqs.scalar_type() == at::kChar, seq_name,
              " must be int8 residue indices (see datasets.sequence_to_indices)");
  TORCH_CHECK(lens.dim() == 1 && lens.size(0) == seqs.size(0), len_name,
              " must be 1D with one length per row of ", seq_name);
  TORCH_CHECK(lens.scalar_type() == at::kLong, len_name, " must be int64");
  if (lens.numel() > 0) {
    // One host sync, so that bad lengths are reported instead of clamped.
    const bool bad = at::logical_or(lens.lt(0), lens.gt(seqs.size(1))).any().item<bool>();
    TORCH_CHECK(!bad, len_name, " must be in [0, ", seqs.size(1), "] (the width of ",
                seq_name, ")");
  }
}

// Shared implementation. pair_indices is undefined for the all-pairs variant.
at::Tensor smith_waterman_impl(const at::Tensor& query_seqs,
                               const at::Tensor& db_seqs,
                               const at::Tensor& query_lens,
                               const at::Tensor& db_lens,
                               const at::Tensor& pair_indices, int64_t gap_open,
                               int64_t gap_extend) {
  check_sequences(query_seqs, query_lens, "query_seqs", "query_lens");
  check_sequences(db_seqs, db_lens, "db_seqs", "db_lens");
  check_same_device(query_seqs, db_seqs, "query_seqs", "db_seqs");
  check_same_device(query_seqs, query_lens, "query_seqs", "query_lens");
  check_same_device(query_seqs, db_lens, "query_seqs", "db_lens");
  TORCH_CHECK(gap_open >= 0 && gap_extend >= 0 && gap_open <= 1000 &&
                  gap_extend <= 1000,
              "gap_open and gap_extend must be in [0, 1000]");
  const c10::cuda::CUDAGuard guard(query_seqs.device());

  const int64_t num_queries = query_seqs.size(0);
  const int64_t num_db = db_seqs.size(0);
  const bool all_pairs = !pair_indices.defined();

  at::Tensor pairs;
  int64_t num_pairs;
  if (all_pairs) {
    num_pairs = num_queries * num_db;
  } else {
    check_cuda(pair_indices, "pair_indices");
    check_same_device(query_seqs, pair_indices, "query_seqs", "pair_indices");
    TORCH_CHECK(pair_indices.dim() == 2 && pair_indices.size(1) == 2,
                "pair_indices must be [num_pairs, 2] (query index, db index)");
    TORCH_CHECK(pair_indices.scalar_type() == at::kLong, "pair_indices must be int64");
    pairs = pair_indices.contiguous();
    num_pairs = pairs.size(0);
    if (num_pairs > 0) {
      const at::Tensor qi = pairs.select(1, 0);
      const at::Tensor di = pairs.select(1, 1);
      const bool bad = at::logical_or(
                           at::logical_or(qi.lt(0), qi.ge(num_queries)),
                           at::logical_or(di.lt(0), di.ge(num_db)))
                           .any()
                           .item<bool>();
      TORCH_CHECK(!bad, "pair_indices out of range: query index must be in [0, ",
                  num_queries, ") and db index in [0, ", num_db, ")");
    }
  }

  at::Tensor scores = at::zeros({num_pairs}, query_seqs.options().dtype(at::kFloat));
  const int64_t db_width = db_seqs.size(1);
  if (num_pairs == 0 || db_width == 0 || query_seqs.size(1) == 0) {
    return all_pairs ? scores.view({num_queries, num_db}) : scores;
  }

  const at::Tensor q = query_seqs.contiguous();
  const at::Tensor d = db_seqs.contiguous();
  const at::Tensor ql = query_lens.contiguous();
  const at::Tensor dl = db_lens.contiguous();

  // Enough workers to fill the GPU, capped so the scratch stays within budget.
  const int64_t bytes_per_worker = 2 * db_width * static_cast<int64_t>(sizeof(int));
  int64_t workers = std::min(num_pairs, std::max<int64_t>(
                                            kSwThreads, kScratchBudgetBytes / bytes_per_worker));
  const int64_t blocks = ceil_div(workers, kSwThreads);
  workers = blocks * kSwThreads;
  at::Tensor scratch = at::empty({2 * db_width * workers}, q.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  smith_waterman_kernel<<<grid_x(blocks), kSwThreads, 0, stream>>>(
      q.data_ptr<int8_t>(), q.size(1), ql.data_ptr<int64_t>(),
      d.data_ptr<int8_t>(), db_width, dl.data_ptr<int64_t>(),
      all_pairs ? nullptr : pairs.data_ptr<int64_t>(), num_db, num_pairs,
      static_cast<int>(gap_open), static_cast<int>(gap_extend),
      scratch.data_ptr<int>(), scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return all_pairs ? scores.view({num_queries, num_db}) : scores;
}

}  // namespace

// All pairs: returns float32 [num_queries, num_db].
at::Tensor smith_waterman(const at::Tensor& query_seqs, const at::Tensor& db_seqs,
                          const at::Tensor& query_lens, const at::Tensor& db_lens,
                          int64_t gap_open, int64_t gap_extend) {
  return smith_waterman_impl(query_seqs, db_seqs, query_lens, db_lens,
                             at::Tensor(), gap_open, gap_extend);
}

// Selected pairs: pair_indices is int64 [num_pairs, 2] of (query index, db
// index). Returns float32 [num_pairs].
at::Tensor smith_waterman_batch(const at::Tensor& query_seqs,
                                const at::Tensor& db_seqs,
                                const at::Tensor& query_lens,
                                const at::Tensor& db_lens,
                                const at::Tensor& pair_indices,
                                int64_t gap_open, int64_t gap_extend) {
  TORCH_CHECK(pair_indices.defined(), "pair_indices must be a tensor");
  return smith_waterman_impl(query_seqs, db_seqs, query_lens, db_lens,
                             pair_indices, gap_open, gap_extend);
}

}  // namespace psg
