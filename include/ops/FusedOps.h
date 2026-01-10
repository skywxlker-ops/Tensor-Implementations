#pragma once
/**
 * FusedOps.h - Header declarations for fused CUDA kernels
 * 
 * Benefits of kernel fusion:
 * 1. Reduces kernel launch overhead
 * 2. Improves memory locality
 * 3. Reduces global memory traffic
 */

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "core/Tensor.h"

namespace OwnTensor {
namespace fused {

// output = (a + b) * scale
void cuda_add_scale(const Tensor& A, const Tensor& B, float scale, Tensor& output, cudaStream_t stream = 0);

// output = a * scale + b
void cuda_scale_add(const Tensor& A, float scale, const Tensor& B, Tensor& output, cudaStream_t stream = 0);

// output = (a + b) * c
void cuda_add_mul(const Tensor& A, const Tensor& B, const Tensor& C, Tensor& output, cudaStream_t stream = 0);

} // namespace fused
} // namespace OwnTensor

#endif // WITH_CUDA
