#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_make_shards_inplace_custom() {
    print_separator("TEST: make_shards_inplace_cust(shard_shapes, row_major) - VIEWS");

    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x10):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);

    std::vector<Shape> custom_shapes = {
        Shape({{1, 2}}),
        Shape({{1, 3}}),
        Shape({{1, 5}})
    };

    std::vector<Tensor> shards = source.make_shards_inplace_cust(custom_shapes, true);

    std::cout << "\nInplace custom shards (2, 3, 5 elements as views):" << std::endl;

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    size_t expected_offset = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], (int)shards[i].numel());

        std::cout << "  Shard " << i << " Storage Offset: " << (shards[i].storage_offset() == expected_offset ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].storage_offset() == expected_offset);
        
        std::cout << "  Shard " << i << " Owns Data:      " << (shards[i].owns_data() == false ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].owns_data() == false);

        expected_offset += shards[i].numel();
    }

    shards[2].data<float>()[0] = 888.0f;
    std::cout << "\nModified shard[2][0] = 888.0f" << std::endl;
    print_tensor_data("Source after", source, 10);
    
    bool modification_visible = (source.data<float>()[5] == 888.0f);
    std::cout << "Modification Visible: " << (modification_visible ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(modification_visible);

    std::cout << "\n[PASS] make_shards_inplace_cust() creates custom views sharing storage" << std::endl;
}

int main() {
    try {
        test_make_shards_inplace_custom();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
