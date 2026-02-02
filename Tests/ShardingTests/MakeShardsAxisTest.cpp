#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_make_shards_axis() {
    print_separator("TEST: make_shards_axis(num_shards, axis)");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (3x4):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);
    std::cout << "  Layout: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]" << std::endl;

    std::vector<Tensor> shards = source.make_shards_axis(2, 1);

    std::cout << "\nShards along axis=1 (split cols 4 -> 2+2):" << std::endl;

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 6);

        std::cout << "  Shard " << i << " Element Count: " << (shards[i].numel() == 6 ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].numel() == 6);
        
        std::cout << "  Shard " << i << " Shape[0]:      " << (shards[i].shape().dims[0] == 3 ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].shape().dims[0] == 3);
        
        std::cout << "  Shard " << i << " Shape[1]:      " << (shards[i].shape().dims[1] == 2 ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].shape().dims[1] == 2);
    }

    float* s0 = shards[0].data<float>();
    bool s0_correct = (s0[0] == 0.0f && s0[1] == 1.0f && s0[2] == 4.0f && s0[3] == 5.0f && s0[4] == 8.0f && s0[5] == 9.0f);
    std::cout << "\nShard 0 Values:    " << (s0_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s0_correct);

    float* s1 = shards[1].data<float>();
    bool s1_correct = (s1[0] == 2.0f && s1[1] == 3.0f && s1[2] == 6.0f && s1[3] == 7.0f && s1[4] == 10.0f && s1[5] == 11.0f);
    std::cout << "Shard 1 Values:    " << (s1_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s1_correct);

    std::cout << "\n[PASS] make_shards_axis() correctly splits along specified axis" << std::endl;
}

int main() {
    try {
        test_make_shards_axis();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
