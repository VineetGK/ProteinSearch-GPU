// Row-wise L2 normalization: y = x / max(||x||_2, eps), matching
// torch.nn.functional.normalize(x, dim=1). One block per row.

#include "common.cuh"

namespace psg {
namespace {

constexpr int kNormThreads = 256;  // must be a multiple of 32
constexpr float kNormEps = 1e-12f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(kFullMask, val, offset);
  }
  return val;
}

// Sum over the whole block. The result is returned to every thread.
__device__ float block_reduce_sum(float val) {
  __shared__ float warp_sums[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) warp_sums[warp] = val;
  __syncthreads();

  if (warp == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    float v = lane < num_warps ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// Generic path (any float dtype, any dim). `in` and `out` may alias
// (in-place normalization), so no __restrict__ here.
template <typename scalar_t>
__global__ void l2_normalize_kernel(const scalar_t* in, scalar_t* out,
                                    int64_t dim) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = in + row * dim;
  scalar_t* y = out + row * dim;

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = static_cast<float>(x[i]);
    sum_sq += v * v;
  }
  const float norm = fmaxf(sqrtf(block_reduce_sum(sum_sq)), kNormEps);

  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    y[i] = static_cast<scalar_t>(static_cast<float>(x[i]) / norm);
  }
}

// fp32 fast path. Only used when dim % 4 == 0 and both base pointers are
// 16-byte aligned, so every row start is float4-aligned.
__global__ void l2_normalize_vec4_kernel(const float* in, float* out,
                                         int64_t dim) {
  const int64_t row = blockIdx.x;
  const float4* x = reinterpret_cast<const float4*>(in + row * dim);
  float4* y = reinterpret_cast<float4*>(out + row * dim);
  const int64_t n4 = dim / 4;

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
    const float4 v = x[i];
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  const float norm = fmaxf(sqrtf(block_reduce_sum(sum_sq)), kNormEps);

  for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
    float4 v = x[i];
    v.x /= norm;
    v.y /= norm;
    v.z /= norm;
    v.w /= norm;
    y[i] = v;
  }
}

// Normalizes the rows of contiguous `input` into contiguous `output`
// (which may be the same tensor).
void launch_l2_normalize(const at::Tensor& input, at::Tensor& output) {
  const int64_t num_vectors = input.size(0);
  const int64_t dim = input.size(1);
  if (num_vectors == 0 || dim == 0) return;

  const dim3 grid(grid_x(num_vectors));
  const dim3 block(kNormThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::kFloat && dim % 4 == 0 &&
      is_aligned16(input.data_ptr()) && is_aligned16(output.data_ptr())) {
    l2_normalize_vec4_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

}  // namespace

at::Tensor l2_normalize(const at::Tensor& input) {
  check_cuda(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [num_vectors, dim]");
  check_float_dtype(input, "input");
  const c10::cuda::CUDAGuard guard(input.device());

  const at::Tensor in = input.contiguous();
  at::Tensor out = at::empty_like(in);
  launch_l2_normalize(in, out);
  return out;
}

at::Tensor l2_normalize_(at::Tensor input) {
  check_cuda(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [num_vectors, dim]");
  TORCH_CHECK(input.is_contiguous(),
              "l2_normalize_ (in-place) requires a contiguous tensor");
  check_float_dtype(input, "input");
  const c10::cuda::CUDAGuard guard(input.device());

  launch_l2_normalize(input, input);
  return input;
}

}  // namespace psg
