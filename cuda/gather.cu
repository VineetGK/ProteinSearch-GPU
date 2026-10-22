// Gathers database rows by index: out[q, j, :] = database[indices[q, j], :].
// Indices outside [0, db_size) (for example -1 padding) produce a zero row.

#include "common.cuh"

namespace psg {
namespace {

constexpr int kGatherThreads = 256;
constexpr int64_t kMaxGatherBlocks = 1 << 20;

template <typename scalar_t>
__global__ void gather_kernel(const int64_t* __restrict__ indices,
                              const scalar_t* __restrict__ database,
                              scalar_t* __restrict__ out, int64_t num_rows,
                              int64_t db_size, int64_t dim) {
  for (int64_t r = blockIdx.x; r < num_rows; r += gridDim.x) {
    const int64_t src = indices[r];
    scalar_t* dst = out + r * dim;
    if (src < 0 || src >= db_size) {
      for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = static_cast<scalar_t>(0.0f);
      }
      continue;
    }
    const scalar_t* row = database + src * dim;
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = row[i];
  }
}

// fp32 fast path; only used when dim % 4 == 0 and both base pointers are
// 16-byte aligned.
__global__ void gather_vec4_kernel(const int64_t* __restrict__ indices,
                                   const float* __restrict__ database,
                                   float* __restrict__ out, int64_t num_rows,
                                   int64_t db_size, int64_t dim) {
  const int64_t n4 = dim / 4;
  for (int64_t r = blockIdx.x; r < num_rows; r += gridDim.x) {
    const int64_t src = indices[r];
    float4* dst = reinterpret_cast<float4*>(out + r * dim);
    if (src < 0 || src >= db_size) {
      for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
        dst[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      continue;
    }
    const float4* row = reinterpret_cast<const float4*>(database + src * dim);
    for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = row[i];
  }
}

}  // namespace

// database: [db_size, dim]; indices: int64 [num_queries, k].
// Returns [num_queries, k, dim] with the database dtype.
at::Tensor gather(const at::Tensor& database, const at::Tensor& indices) {
  check_cuda(database, "database");
  check_cuda(indices, "indices");
  check_same_device(database, indices, "database", "indices");
  TORCH_CHECK(database.dim() == 2, "database must be 2D [db_size, dim]");
  TORCH_CHECK(indices.dim() == 2, "indices must be 2D [num_queries, k]");
  check_float_dtype(database, "database");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  const c10::cuda::CUDAGuard guard(database.device());

  const at::Tensor db = database.contiguous();
  const at::Tensor idx = indices.contiguous();
  const int64_t db_size = db.size(0);
  const int64_t dim = db.size(1);
  const int64_t num_rows = idx.numel();

  at::Tensor out = at::empty({idx.size(0), idx.size(1), dim}, db.options());
  if (num_rows == 0 || dim == 0) return out;

  const dim3 grid(static_cast<unsigned>(std::min(num_rows, kMaxGatherBlocks)));
  const dim3 block(kGatherThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (db.scalar_type() == at::kFloat && dim % 4 == 0 &&
      is_aligned16(db.data_ptr()) && is_aligned16(out.data_ptr())) {
    gather_vec4_kernel<<<grid, block, 0, stream>>>(
        idx.data_ptr<int64_t>(), db.data_ptr<float>(), out.data_ptr<float>(),
        num_rows, db_size, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, db.scalar_type(), "gather", [&] {
        gather_kernel<scalar_t><<<grid, block, 0, stream>>>(
            idx.data_ptr<int64_t>(), db.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), num_rows, db_size, dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return out;
}

}  // namespace psg
