// Dot-product similarity between queries and database rows:
//   out[q, d] = sum_i query[q, i] * database[d, i]
// This equals cosine similarity only when both inputs are already
// L2-normalized (see l2_normalize). Accumulation is in fp32 for every input
// dtype. One thread per (query, database row); this is a simple, correct
// kernel, not a tuned GEMM (torch.matmul / cuBLAS will be faster).

#include "common.cuh"

namespace psg {
namespace {

constexpr int kDotThreads = 256;
constexpr int64_t kMaxGridY = 65535;

template <typename scalar_t>
__global__ void dot_similarity_kernel(const scalar_t* __restrict__ queries,
                                      const scalar_t* __restrict__ database,
                                      float* __restrict__ out,
                                      int64_t num_queries, int64_t db_size,
                                      int64_t dim) {
  const int64_t d = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (d >= db_size) return;
  const scalar_t* db_row = database + d * dim;

  for (int64_t q = blockIdx.y; q < num_queries; q += gridDim.y) {
    const scalar_t* q_row = queries + q * dim;
    float acc = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
      acc += static_cast<float>(q_row[i]) * static_cast<float>(db_row[i]);
    }
    out[q * db_size + d] = acc;
  }
}

// fp32 fast path; only used when dim % 4 == 0 and both base pointers are
// 16-byte aligned.
__global__ void dot_similarity_vec4_kernel(const float* __restrict__ queries,
                                           const float* __restrict__ database,
                                           float* __restrict__ out,
                                           int64_t num_queries,
                                           int64_t db_size, int64_t dim) {
  const int64_t d = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (d >= db_size) return;
  const int64_t n4 = dim / 4;
  const float4* db_row = reinterpret_cast<const float4*>(database + d * dim);

  for (int64_t q = blockIdx.y; q < num_queries; q += gridDim.y) {
    const float4* q_row = reinterpret_cast<const float4*>(queries + q * dim);
    float acc = 0.0f;
    for (int64_t i = 0; i < n4; ++i) {
      const float4 a = q_row[i];
      const float4 b = db_row[i];
      acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    out[q * db_size + d] = acc;
  }
}

}  // namespace

// query: [dim] or [num_queries, dim]; database: [db_size, dim].
// Returns float32 [db_size] or [num_queries, db_size].
at::Tensor cosine_similarity(const at::Tensor& query,
                             const at::Tensor& database) {
  check_cuda(query, "query");
  check_cuda(database, "database");
  check_same_device(query, database, "query", "database");
  TORCH_CHECK(query.dim() == 1 || query.dim() == 2,
              "query must be 1D [dim] or 2D [num_queries, dim]");
  TORCH_CHECK(database.dim() == 2, "database must be 2D [db_size, dim]");
  check_float_dtype(query, "query");
  TORCH_CHECK(query.scalar_type() == database.scalar_type(),
              "query and database must have the same dtype");
  TORCH_CHECK(query.size(-1) == database.size(1),
              "query and database dimensions must match (got ",
              query.size(-1), " and ", database.size(1), ")");
  const c10::cuda::CUDAGuard guard(query.device());

  const bool single = query.dim() == 1;
  const at::Tensor q = (single ? query.unsqueeze(0) : query).contiguous();
  const at::Tensor db = database.contiguous();
  const int64_t num_queries = q.size(0);
  const int64_t db_size = db.size(0);
  const int64_t dim = q.size(1);

  at::Tensor out = at::empty({num_queries, db_size}, q.options().dtype(at::kFloat));
  if (num_queries > 0 && db_size > 0) {
    if (dim == 0) {
      out.zero_();
    } else {
      const dim3 block(kDotThreads);
      const dim3 grid(grid_x(ceil_div(db_size, kDotThreads)),
                      static_cast<unsigned>(std::min(num_queries, kMaxGridY)));
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      if (q.scalar_type() == at::kFloat && dim % 4 == 0 &&
          is_aligned16(q.data_ptr()) && is_aligned16(db.data_ptr())) {
        dot_similarity_vec4_kernel<<<grid, block, 0, stream>>>(
            q.data_ptr<float>(), db.data_ptr<float>(), out.data_ptr<float>(),
            num_queries, db_size, dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, q.scalar_type(), "cosine_similarity", [&] {
              dot_similarity_kernel<scalar_t><<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), db.data_ptr<scalar_t>(),
                  out.data_ptr<float>(), num_queries, db_size, dim);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      }
    }
  }
  return single ? out.squeeze(0) : out;
}

}  // namespace psg
