#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_error_handling() {
    print_separator("TEST: Error Handling");

    Tensor source = Tensor::zeros({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));

    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    std::cout << "\nTesting make_shards with non-divisible count..." << std::endl;
    try {
        auto shards = source.make_shards(3, true);
        std::cout << "Non-divisible shards: ❌ FAILED (should have thrown)" << std::endl;
        assert(false && "Should have thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "Non-divisible shards: ✅ PASSED (caught: " << e.what() << ")" << std::endl;
    }

    std::cout << "\nTesting shard_into with overflow..." << std::endl;
    Tensor dest1 = Tensor::zeros({{1, 6}});
    Tensor dest2 = Tensor::zeros({{1, 6}});
    std::vector<Tensor> dests = {dest1, dest2};
    try {
        source.shard_into(dests);
        std::cout << "Overflow shard_into: ❌ FAILED (should have thrown)" << std::endl;
        assert(false && "Should have thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "Overflow shard_into: ✅ PASSED (caught: " << e.what() << ")" << std::endl;
    }

    std::cout << "\nTesting make_shards_cust with wrong total..." << std::endl;
    try {
        std::vector<Shape> bad_shapes = {Shape({{1, 5}}), Shape({{1, 6}})};
        auto shards = source.make_shards_cust(bad_shapes, true);
        std::cout << "Wrong total shards: ❌ FAILED (should have thrown)" << std::endl;
        assert(false && "Should have thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "Wrong total shards: ✅ PASSED (caught: " << e.what() << ")" << std::endl;
    }

    std::cout << "\n[PASS] Error handling works correctly" << std::endl;
}

int main() {
    try {
        test_error_handling();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Unexpected Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
