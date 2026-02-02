#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_narrow() {
    print_separator("TEST: narrow(axis, start, length)");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (3x4):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);

    Tensor narrowed = source.narrow(0, 1, 2);

    std::cout << "\nNarrowed along axis=0, start=1, length=2 (rows 1-2):" << std::endl;
    print_tensor_info("Narrowed", narrowed);
    print_tensor_data("Narrowed", narrowed, 8);

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    std::cout << "Element Count:   " << (narrowed.numel() == 8 ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(narrowed.numel() == 8);

    std::cout << "Shape[0]:        " << (narrowed.shape().dims[0] == 2 ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(narrowed.shape().dims[0] == 2);

    std::cout << "Shape[1]:        " << (narrowed.shape().dims[1] == 4 ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(narrowed.shape().dims[1] == 4);

    std::cout << "Owns Data:       " << (narrowed.owns_data() == true ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(narrowed.owns_data() == true);

    float* n_ptr = narrowed.data<float>();
    bool first_correct = (n_ptr[0] == 4.0f);
    bool last_correct = (n_ptr[7] == 11.0f);
    std::cout << "First Value:     " << (first_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    std::cout << "Last Value:      " << (last_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(first_correct && last_correct);

    std::cout << "\n[PASS] narrow() extracts correct slice with proper shape" << std::endl;
}

int main() {
    try {
        test_narrow();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
