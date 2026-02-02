#include "TensorLib.h"
#include <iostream>
#include <vector>

using namespace OwnTensor;
using namespace OwnTensor::nn;

void test_adam_corruption() {
    auto linear = new Linear(10, 10);
    
    // CASE 1: Normal registration
    std::vector<Tensor> params_single = linear->parameters();
    Adam optim_single(params_single, 0.1);
    
    // CASE 2: Double registration (simulating what the user does)
    std::vector<Tensor> params_double = params_single;
    params_double.insert(params_double.end(), params_single.begin(), params_single.end());
    Adam optim_double(params_double, 0.1);

    std::cout << "Single registration params count: " << params_single.size() << std::endl;
    std::cout << "Double registration params count: " << params_double.size() << std::endl;

    // Verify duplication
    if (params_double[0].unsafeGetTensorImpl() == params_double[params_single.size()].unsafeGetTensorImpl()) {
        std::cout << "Confirmed: params_double contains duplicate TensorImpls!" << std::endl;
    }

    // Now, if they use this in training, optim_double will have 2 entries in its internal state 
    // (m and v vectors) for the same underlying tensor.
    // Each update will apply twice with different momentum estimates.
}

int main() {
    try {
        test_adam_corruption();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
