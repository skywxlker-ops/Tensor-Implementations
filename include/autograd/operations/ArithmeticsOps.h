#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

Tensor square(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor neg(const Tensor& input);
Tensor abs(const Tensor& input);
Tensor reciprocal(const Tensor& input);
Tensor pow(const Tensor& input, float exponent);

// Scalar Arithmetic
Tensor add(const Tensor& a, float b);
Tensor add(float a, const Tensor& b);
Tensor sub(const Tensor& a, float b);
Tensor sub(float a, const Tensor& b);
Tensor mul(const Tensor& a, float b);
Tensor mul(float a, const Tensor& b);
Tensor div(const Tensor& a, float b);
Tensor div(float a, const Tensor& b);

} // namespace autograd
} // namespace OwnTensor
