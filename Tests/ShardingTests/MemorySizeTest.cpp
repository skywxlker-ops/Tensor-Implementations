#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_memory_sizes() {
    print_separator("TEST: Memory Size Verification");

    const size_t N = 1000;
    Tensor source = Tensor({{1, (int64_t)N}}, TensorOptions().with_dtype(Dtype::Float32));
    source.fill(1.0f);

    std::cout << "Source: " << N << " elements, " << source.nbytes() << " bytes" << std::endl;
    std::cout << "Expected: " << N * sizeof(float) << " bytes" << std::endl;
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    std::cout << "Source nbytes: " << (source.nbytes() == N * sizeof(float) ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source.nbytes() == N * sizeof(float));

    auto copy_shards = source.make_shards(4, true);
    size_t total_copy_bytes = 0;
    for (auto& s : copy_shards) {
        total_copy_bytes += s.nbytes();
        assert(s.allocated_bytes() >= s.nbytes());
    }
    std::cout << "\nmake_shards(4): total bytes = " << total_copy_bytes << std::endl;
    std::cout << "Total bytes match: " << (total_copy_bytes == N * sizeof(float) ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(total_copy_bytes == N * sizeof(float));

    auto inplace_shards = source.make_shards_inplace(4, true);
    std::cout << "\nInplace shards:" << std::endl;
    for (size_t i = 0; i < inplace_shards.size(); ++i) {
        auto& s = inplace_shards[i];
        std::cout << "  Shard " << i << ": numel=" << s.numel() 
                  << ", nbytes=" << s.nbytes() 
                  << ", allocated=" << s.allocated_bytes() << std::endl;
        
        assert(s.nbytes() == (N / 4) * sizeof(float));
        assert(s.allocated_bytes() >= N * sizeof(float));
    }
    std::cout << "Inplace shard sizes: ✅ PASSED" << std::endl;

    std::cout << "\n[PASS] Memory sizes are correct for all sharding types" << std::endl;
}

int main() {
    try {
        test_memory_sizes();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
