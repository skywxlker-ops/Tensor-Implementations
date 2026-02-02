#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_make_shards_custom() {
    print_separator("TEST: make_shards_cust(shard_shapes, row_major)");

    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x10):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);

    std::vector<Shape> custom_shapes = {
        Shape({{1, 3}}),
        Shape({{1, 2}}),
        Shape({{1, 5}})
    };

    std::vector<Tensor> shards = source.make_shards_cust(custom_shapes, true);

    std::cout << "\nCustom shards (3, 2, 5 elements):" << std::endl;

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], (int)shards[i].numel());

        std::cout << "  Shard " << i << " Owns Data: " << (shards[i].owns_data() == true ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].owns_data() == true);
        
        std::cout << "  Shard " << i << " Is Copy:   " << (shards[i].data() != source.data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].data() != source.data());
    }

    float* s0 = shards[0].data<float>();
    bool s0_correct = (s0[0] == 0.0f && s0[1] == 1.0f && s0[2] == 2.0f);
    std::cout << "\nShard 0 Values:  " << (s0_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s0_correct);

    float* s1 = shards[1].data<float>();
    bool s1_correct = (s1[0] == 3.0f && s1[1] == 4.0f);
    std::cout << "Shard 1 Values:  " << (s1_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s1_correct);

    float* s2 = shards[2].data<float>();
    bool s2_correct = true;
    for (int i = 0; i < 5; ++i) {
        if (s2[i] != 5.0f + i) s2_correct = false;
    }
    std::cout << "Shard 2 Values:  " << (s2_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s2_correct);

    std::cout << "\n[PASS] make_shards_cust() correctly splits with custom shapes" << std::endl;
}

int main() {
    try {
        test_make_shards_custom();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
