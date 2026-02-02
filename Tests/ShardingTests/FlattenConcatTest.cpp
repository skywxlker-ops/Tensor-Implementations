#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_flatten_concat() {
    print_separator("TEST: flatten_concat(tensor_list)");

    Tensor t1 = Tensor({{1, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor t2 = Tensor({{2, 2}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor t3 = Tensor({{1, 5}}, TensorOptions().with_dtype(Dtype::Float32));

    t1.set_data(std::vector<float>{1.0f, 2.0f, 3.0f});
    t2.set_data(std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f});
    t3.set_data(std::vector<float>{8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::cout << "Input tensors:" << std::endl;
    print_tensor_info("t1 (1x3)", t1);
    print_tensor_data("t1", t1, 3);
    print_tensor_info("t2 (2x2)", t2);
    print_tensor_data("t2", t2, 4);
    print_tensor_info("t3 (1x5)", t3);
    print_tensor_data("t3", t3, 5);

    std::vector<Tensor> tensors = {t1, t2, t3};
    Tensor result = Tensor::flatten_concat(tensors);

    std::cout << "\nConcatenated result:" << std::endl;
    print_tensor_info("Result", result);
    print_tensor_data("Result", result, 12);

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    std::cout << "Element Count:   " << (result.numel() == 12 ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(result.numel() == 12);
    
    std::cout << "Owns Data:       " << (result.owns_data() == true ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(result.owns_data() == true);

    float* r_ptr = result.data<float>();
    bool values_correct = true;
    for (int i = 0; i < 12; ++i) {
        if (r_ptr[i] != (float)(i + 1)) values_correct = false;
    }
    std::cout << "Value Check:     " << (values_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(values_correct);

    std::cout << "\n[PASS] flatten_concat() correctly concatenates tensors" << std::endl;
}

int main() {
    try {
        test_flatten_concat();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
