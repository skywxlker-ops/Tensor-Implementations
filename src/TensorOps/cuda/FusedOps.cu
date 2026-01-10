#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/Tensor.h"
#include "core/TensorDispatch.h"

namespace OwnTensor {
namespace fused {

// ============================================================================
// add_scale: output = (a + b) * scale
// ============================================================================

template<typename T>
__global__ void add_scale_kernel(const T* a, const T* b, T scale, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (a[idx] + b[idx]) * scale;
    }
}

template<>
__global__ void add_scale_kernel<__half>(const __half* a, const __half* b, __half scale, __half* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hmul(__hadd(a[idx], b[idx]), scale);
    }
}

void cuda_add_scale(const Tensor& A, const Tensor& B, float scale, Tensor& output, cudaStream_t stream) {
    size_t n = output.numel();
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    
    dispatch_by_dtype(A.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        const T* b_ptr = B.data<T>();
        T* out_ptr = output.data<T>();
        T scale_t = static_cast<T>(scale);
        
        add_scale_kernel<<<grid, block, 0, stream>>>(a_ptr, b_ptr, scale_t, out_ptr, n);
    });
}

// ============================================================================
// scale_add: output = a * scale + b
// ============================================================================

template<typename T>
__global__ void scale_add_kernel(const T* a, T scale, const T* b, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * scale + b[idx];
    }
}

void cuda_scale_add(const Tensor& A, float scale, const Tensor& B, Tensor& output, cudaStream_t stream) {
    size_t n = output.numel();
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    
    dispatch_by_dtype(A.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        scale_add_kernel<<<grid, block, 0, stream>>>(A.data<T>(), static_cast<T>(scale), B.data<T>(), output.data<T>(), n);
    });
}

// ============================================================================
// add_mul: output = (a + b) * c
// ============================================================================

template<typename T>
__global__ void add_mul_kernel(const T* a, const T* b, const T* c, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (a[idx] + b[idx]) * c[idx];
    }
}

void cuda_add_mul(const Tensor& A, const Tensor& B, const Tensor& C, Tensor& output, cudaStream_t stream) {
    size_t n = output.numel();
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    
    dispatch_by_dtype(A.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        add_mul_kernel<<<grid, block, 0, stream>>>(A.data<T>(), B.data<T>(), C.data<T>(), output.data<T>(), n);
    });
}

} // namespace fused
} // namespace OwnTensor

#endif // WITH_CUDA
