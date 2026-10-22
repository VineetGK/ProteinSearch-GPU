// Row-wise top-k (largest first) for k <= 32. One block per query row.
//
// Phase 1: every thread scans a strided slice of the row and keeps the k best
//          (value, index) pairs it has seen in a small sorted local list.
// Phase 2: the block merges the per-thread lists with k rounds of block-wide
//          argmax over the list heads; the winning thread pops its head.
//
// The union of the per-thread top-k lists always contains the row's top-k,
// so the result matches torch.topk(sorted=True) values. Ties are broken by the
// smaller index. NaN is treated as larger than any number, like torch.topk.

#include "common.cuh"

namespace psg {
namespace {

constexpr int kTopkMaxK = 32;
constexpr int kTopkThreads = 256;  // must be a multiple of 32

// a strictly greater than b, with NaN ordered above every number.
__device__ __forceinline__ bool greater_than(float a, float b) {
  const bool a_nan = isnan(a);
  const bool b_nan = isnan(b);
  if (a_nan || b_nan) return a_nan && !b_nan;
  return a > b;
}

// Candidate ordering for the merge. idx < 0 marks an empty slot.
__device__ __forceinline__ bool better(float va, int64_t ia, float vb,
                                       int64_t ib) {
  if ((ia >= 0) != (ib >= 0)) return ia >= 0;
  if (greater_than(va, vb)) return true;
  if (greater_than(vb, va)) return false;
  return ia < ib;
}

// After the loop, lane 0 holds the warp's best candidate.
__device__ __forceinline__ void warp_argmax(float& val, int64_t& idx) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_val = __shfl_down_sync(kFullMask, val, offset);
    const int64_t other_idx = static_cast<int64_t>(
        __shfl_down_sync(kFullMask, static_cast<long long>(idx), offset));
    if (better(other_val, other_idx, val, idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

__global__ void topk_kernel(const float* __restrict__ sims, int64_t n, int k,
                            float* __restrict__ out_values,
                            int64_t* __restrict__ out_indices) {
  const int64_t row = blockIdx.x;
  const float* s = sims + row * n;

  // Phase 1: thread-local sorted (descending) list of up to k candidates.
  float local_val[kTopkMaxK];
  int64_t local_idx[kTopkMaxK];
  int count = 0;
  for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
    const float v = s[j];
    // Equal values do not displace earlier (smaller) indices.
    if (count == k && !greater_than(v, local_val[k - 1])) continue;
    int pos = count < k ? count : k - 1;  // when full, the last entry is dropped
    while (pos > 0 && greater_than(v, local_val[pos - 1])) {
      local_val[pos] = local_val[pos - 1];
      local_idx[pos] = local_idx[pos - 1];
      --pos;
    }
    local_val[pos] = v;
    local_idx[pos] = j;
    if (count < k) ++count;
  }

  // Phase 2: k rounds of block-wide argmax over the list heads.
  __shared__ float warp_val[32];
  __shared__ int64_t warp_idx[32];
  __shared__ int64_t winner_idx;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;
  int head = 0;

  for (int r = 0; r < k; ++r) {
    float v = head < count ? local_val[head] : 0.0f;
    int64_t i = head < count ? local_idx[head] : -1;
    warp_argmax(v, i);
    if (lane == 0) {
      warp_val[warp] = v;
      warp_idx[warp] = i;
    }
    __syncthreads();

    if (warp == 0) {
      v = lane < num_warps ? warp_val[lane] : 0.0f;
      i = lane < num_warps ? warp_idx[lane] : -1;
      warp_argmax(v, i);
      if (lane == 0) {
        winner_idx = i;
        out_values[row * k + r] = v;
        out_indices[row * k + r] = i;
      }
    }
    __syncthreads();

    // Each index lives in exactly one thread's list, so exactly one thread pops.
    if (head < count && local_idx[head] == winner_idx) ++head;
    // The next round's first __syncthreads() orders this read of winner_idx
    // before warp 0 overwrites it.
  }
}

}  // namespace

std::vector<at::Tensor> topk(const at::Tensor& similarities, int64_t k) {
  check_cuda(similarities, "similarities");
  TORCH_CHECK(similarities.dim() == 2,
              "similarities must be 2D [num_queries, db_size]");
  TORCH_CHECK(similarities.scalar_type() == at::kFloat,
              "similarities must be float32 (got ", similarities.scalar_type(), ")");
  const int64_t num_queries = similarities.size(0);
  const int64_t n = similarities.size(1);
  TORCH_CHECK(k >= 1 && k <= kTopkMaxK, "k must be in [1, ", kTopkMaxK,
              "] (got ", k, "); use torch.topk for larger k");
  TORCH_CHECK(k <= n, "k (", k, ") must be <= db_size (", n, ")");
  const c10::cuda::CUDAGuard guard(similarities.device());

  const at::Tensor sims = similarities.contiguous();
  at::Tensor values = at::empty({num_queries, k}, sims.options());
  at::Tensor indices = at::empty({num_queries, k}, sims.options().dtype(at::kLong));
  if (num_queries == 0) return {values, indices};

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  topk_kernel<<<grid_x(num_queries), kTopkThreads, 0, stream>>>(
      sims.data_ptr<float>(), n, static_cast<int>(k), values.data_ptr<float>(),
      indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {values, indices};
}

}  // namespace psg
