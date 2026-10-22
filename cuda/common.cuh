#pragma once

// Shared helpers for the ProteinSearch-GPU CUDA sources.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace psg {

constexpr unsigned kFullMask = 0xffffffffu;

inline void check_cuda(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_same_device(const at::Tensor& a, const at::Tensor& b,
                              const char* name_a, const char* name_b) {
  TORCH_CHECK(a.device() == b.device(), name_a, " and ", name_b,
              " must be on the same device (got ", a.device(), " and ",
              b.device(), ")");
}

// Floating dtypes the embedding kernels accept. All arithmetic is done in fp32.
inline void check_float_dtype(const at::Tensor& t, const char* name) {
  const auto st = t.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kHalf || st == at::kBFloat16,
              name, " must be float32, float16 or bfloat16 (got ", st, ")");
}

inline bool is_aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// gridDim.x may be up to 2^31 - 1.
inline unsigned grid_x(int64_t blocks) {
  TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(),
              "problem is too large for a single launch");
  return static_cast<unsigned>(blocks);
}

}  // namespace psg
