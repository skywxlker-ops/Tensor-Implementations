#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_make_shards_inplace() {
    print_separator("TEST: make_shards_inplace(num_shards, row_major) - VIEWS");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 12}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x12):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);

    void* source_base = source.data();

    std::vector<Tensor> shards = source.make_shards_inplace(3, true);

    std::cout << "\nInplace shards (views with storage_offset):" << std::endl;

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 4);

        std::cout << "  Shard " << i << " Element Count:   " << (shards[i].numel() == 4 ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].numel() == 4);
        
        std::cout << "  Shard " << i << " Storage Offset:  " << (shards[i].storage_offset() == i * 4 ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].storage_offset() == i * 4);
        
        std::cout << "  Shard " << i << " Owns Data:       " << (shards[i].owns_data() == false ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].owns_data() == false);
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(source_base);
    for (size_t i = 0; i < shards.size(); ++i) {
        uintptr_t shard_addr = reinterpret_cast<uintptr_t>(shards[i].data());
        uintptr_t expected_addr = base + i * 4 * sizeof(float);
        bool addr_correct = (shard_addr == expected_addr);
        std::cout << "\nShard " << i << " Address Check: " << (addr_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(addr_correct);
    }

    std::cout << "\nModifying shard[1][0] = 999.0f..." << std::endl;
    shards[1].data<float>()[0] = 999.0f;

    std::cout << "Source after modification:" << std::endl;
    print_tensor_data("Source", source, 12);
    
    bool modification_visible = (source.data<float>()[4] == 999.0f);
    std::cout << "Modification Visible: " << (modification_visible ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(modification_visible);

    std::cout << "\n[PASS] make_shards_inplace() creates views that share storage" << std::endl;
}

int main() {
    try {
        test_make_shards_inplace();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
