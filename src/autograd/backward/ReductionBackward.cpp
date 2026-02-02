#include "autograd/backward/ReductionBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// SumBackward
// ============================================================================

SumBackward::SumBackward(const Shape& input_shape)
    : Node(1), input_shape_(input_shape) {}

std::vector<Tensor> SumBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SumBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // Scale factor
    double scale = 1.0;
    if (grad_output.ndim() == 0 || grad_output.numel() == 1) {
        if (grad_output.is_cuda()) {
#ifdef WITH_CUDA
            scale = static_cast<double>(grad_output.to_cpu().data<float>()[0]);
#endif
        } else {
            scale = static_cast<double>(*grad_output.data<float>());
        }
    }

    // Single-pass creation of the gradient tensor
    Tensor grad_input = Tensor::full(input_shape_, 
        TensorOptions()
            .with_dtype(grad_output.dtype())
            .with_device(grad_output.device()),
        static_cast<float>(scale));
    
    return {grad_input};
}

// ============================================================================
// MeanBackward
// ============================================================================

MeanBackward::MeanBackward(const Shape& input_shape, int64_t numel)
    : Node(1), input_shape_(input_shape), numel_(numel) {}

std::vector<Tensor> MeanBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MeanBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // Scale factor (1.0 / numel)
    double scale = 1.0 / static_cast<double>(numel_);
    if (grad_output.ndim() == 0 || grad_output.numel() == 1) {
        double grad_val;
        if (grad_output.is_cuda()) {
            grad_val = static_cast<double>(grad_output.to_cpu().data<float>()[0]);
        } else {
            grad_val = static_cast<double>(*grad_output.data<float>());
        }
        scale *= grad_val;
    }

    // Single-pass creation
    Tensor grad_input = Tensor::full(input_shape_,
        TensorOptions()
            .with_dtype(grad_output.dtype())
            .with_device(grad_output.device()),
        static_cast<float>(scale));
    
    return {grad_input};
}

} // namespace autograd
} // namespace OwnTensor