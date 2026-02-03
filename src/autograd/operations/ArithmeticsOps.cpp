#include "autograd/operations/ArithmeticsOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ArithmeticsBackward.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace autograd {

Tensor square(const Tensor& input) {
    return make_unary_op<SquareBackward>(input,
        [](const Tensor& x) { return OwnTensor::square(x.detach()); },
        input);
}

Tensor sqrt(const Tensor& input) {
    return make_unary_op<SqrtBackward>(input,
        [](const Tensor& x) { return OwnTensor::sqrt(x.detach()); },
        input); // Recompute approach or changed design
}

Tensor neg(const Tensor& input) {
    return make_unary_op<NegBackward>(input,
        [](const Tensor& x) { return OwnTensor::neg(x.detach()); });
}

Tensor abs(const Tensor& input) {
    return make_unary_op<AbsBackward>(input,
        [](const Tensor& x) { return OwnTensor::abs(x.detach()); },
        input);
}

Tensor reciprocal(const Tensor& input) {
    // Reciprocal backward needs output or input.
    // d(1/x) = -1/x^2 = -y^2.
    // Let's modify ReciprocalBackward to take Input (safer for now).
    // Wait, actually let's just stick to the pattern.
    // I will modify Sqrt and Reciprocal backward to take INPUT and compute what they need.
    return make_unary_op<ReciprocalBackward>(input,
        [](const Tensor& x) { return OwnTensor::reciprocal(x.detach()); },
        input); // Pass input instead of output
}

Tensor pow(const Tensor& input, float exponent) {
    return make_unary_op<PowBackward>(input,
        [exponent](const Tensor& x) { return OwnTensor::pow(x.detach(), exponent); },
        input, exponent);
}

// Scalar Arithmetic
Tensor add(const Tensor& a, float b) {
    return make_unary_op<ScalarAddBackward>(a,
        [b](const Tensor& x) { return OwnTensor::operator+(x.detach(), b); });
}

Tensor add(float a, const Tensor& b) {
    return make_unary_op<ScalarAddBackward>(b,
        [a](const Tensor& x) { return OwnTensor::operator+(a, x.detach()); });
}

Tensor sub(const Tensor& a, float b) {
    return make_unary_op<ScalarSubBackward>(a,
        [b](const Tensor& x) { return OwnTensor::operator-(x.detach(), b); },
        true); // tensor_on_lhs
}

Tensor sub(float a, const Tensor& b) {
    return make_unary_op<ScalarSubBackward>(b,
        [a](const Tensor& x) { return OwnTensor::operator-(a, x.detach()); },
        false); // tensor_on_lhs = false
}

Tensor mul(const Tensor& a, float b) {
    return make_unary_op<ScalarMulBackward>(a,
        [b](const Tensor& x) { return OwnTensor::operator*(x.detach(), b); },
        static_cast<double>(b));
}

Tensor mul(float a, const Tensor& b) {
    return make_unary_op<ScalarMulBackward>(b,
        [a](const Tensor& x) { return OwnTensor::operator*(a, x.detach()); },
        static_cast<double>(a));
}

Tensor div(const Tensor& a, float b) {
    return make_unary_op<ScalarDivBackward>(a,
        [b](const Tensor& x) { return OwnTensor::operator/(x.detach(), b); },
        static_cast<double>(b), true);
}

Tensor div(float a, const Tensor& b) {
    return make_unary_op<ScalarDivBackward>(b,
        [a](const Tensor& x) { return OwnTensor::operator/(a, x.detach()); },
        static_cast<double>(a), false, b.detach()); // saved_input = b.detach()
}

} // namespace autograd
} // namespace OwnTensor
