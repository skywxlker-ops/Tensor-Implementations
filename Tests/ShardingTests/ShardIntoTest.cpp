#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_shard_into() {
    print_separator("TEST: shard_into(destinations)");

    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x10):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);

    Tensor dest1 = Tensor::zeros({{1, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor dest2 = Tensor::zeros({{1, 2}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor dest3 = Tensor::zeros({{1, 5}}, TensorOptions().with_dtype(Dtype::Float32));

    void* dest1_addr = dest1.data();
    void* dest2_addr = dest2.data();
    void* dest3_addr = dest3.data();

    std::cout << "\nDestinations BEFORE shard_into:" << std::endl;
    print_tensor_data("dest1 (3)", dest1, 3);
    print_tensor_data("dest2 (2)", dest2, 2);
    print_tensor_data("dest3 (5)", dest3, 5);

    std::vector<Tensor> destinations = {dest1, dest2, dest3};
    source.shard_into(destinations);

    std::cout << "\nDestinations AFTER shard_into:" << std::endl;
    print_tensor_data("dest1 (3)", dest1, 3);
    print_tensor_data("dest2 (2)", dest2, 2);
    print_tensor_data("dest3 (5)", dest3, 5);

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    std::cout << "dest1 Address Preserved: " << (dest1.data() == dest1_addr ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(dest1.data() == dest1_addr);
    
    std::cout << "dest2 Address Preserved: " << (dest2.data() == dest2_addr ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(dest2.data() == dest2_addr);
    
    std::cout << "dest3 Address Preserved: " << (dest3.data() == dest3_addr ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(dest3.data() == dest3_addr);

    float* d1 = dest1.data<float>();
    bool d1_correct = (d1[0] == 0.0f && d1[1] == 1.0f && d1[2] == 2.0f);
    std::cout << "dest1 Values:            " << (d1_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(d1_correct);

    float* d2 = dest2.data<float>();
    bool d2_correct = (d2[0] == 3.0f && d2[1] == 4.0f);
    std::cout << "dest2 Values:            " << (d2_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(d2_correct);

    float* d3 = dest3.data<float>();
    bool d3_correct = true;
    for (int i = 0; i < 5; ++i) {
        if (d3[i] != 5.0f + i) d3_correct = false;
    }
    std::cout << "dest3 Values:            " << (d3_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(d3_correct);

    std::cout << "\n[PASS] shard_into() copies data to pre-allocated destinations" << std::endl;
}

int main() {
    try {
        test_shard_into();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
