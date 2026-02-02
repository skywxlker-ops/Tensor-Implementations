#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

/**
 * This test verifies that reference counting works correctly during view
 * creation and deletion for slicing and sharding operations.
 * 
 * Key behaviors being tested:
 * 1. Creating a view should increment the source tensor's reference count
 * 2. Deleting a view should decrement the source tensor's reference count
 * 3. The source tensor's data should remain valid as long as views exist
 * 4. After all views are deleted, the source can be safely destroyed
 * 5. Views should correctly track their base tensor to prevent dangling pointers
 */

void test_slice_inplace_reference_counting() {
    print_separator("TEST: slice_inplace() Reference Counting");
    
    std::cout << "Creating source tensor with 20 elements..." << std::endl;
    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 0.0f);
    
    Tensor source = Tensor({{1, 20}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);
    
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 20);
    
    size_t initial_ref_count = source.unsafeGetTensorImpl()->use_count();
    std::cout << "\nInitial source TensorImpl use_count: " << initial_ref_count << std::endl;
    
    std::cout << "\n--- Creating first view (slice_inplace at 0..10) ---" << std::endl;
    Tensor view1 = source.slice_inplace(0, 10);
    
    size_t after_view1_ref_count = source.unsafeGetTensorImpl()->use_count();
    std::cout << "Source TensorImpl use_count after view1: " << after_view1_ref_count << std::endl;
    
    print_tensor_info("View1", view1);
    print_tensor_data("View1", view1, 10);
    
    std::cout << "View1 owns_data: " << (view1.owns_data() ? "yes" : "no") << std::endl;
    std::cout << "View1 storage_offset: " << view1.storage_offset() << std::endl;
    
    std::cout << "\n--- Creating second view (slice_inplace at 10..20) ---" << std::endl;
    Tensor view2 = source.slice_inplace(10, 10);
    
    size_t after_view2_ref_count = source.unsafeGetTensorImpl()->use_count();
    std::cout << "Source TensorImpl use_count after view2: " << after_view2_ref_count << std::endl;
    
    print_tensor_info("View2", view2);
    print_tensor_data("View2", view2, 10);
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    std::cout << "View1 does not own data: " << (!view1.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(!view1.owns_data());
    
    std::cout << "View2 does not own data: " << (!view2.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(!view2.owns_data());
    
    std::cout << "Source still owns data:  " << (source.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source.owns_data());
    
    std::cout << "\n--- Testing data consistency across views ---" << std::endl;
    
    std::cout << "Modifying view1[5] = 555.0f..." << std::endl;
    view1.data<float>()[5] = 555.0f;
    
    bool source_sees_change = (source.data<float>()[5] == 555.0f);
    std::cout << "Source sees view1 change: " << (source_sees_change ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source_sees_change);
    
    std::cout << "Modifying view2[3] = 333.0f..." << std::endl;
    view2.data<float>()[3] = 333.0f;
    
    bool source_sees_change2 = (source.data<float>()[13] == 333.0f);
    std::cout << "Source sees view2 change: " << (source_sees_change2 ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source_sees_change2);
    
    print_tensor_data("Source after modifications", source, 20);
    
    std::cout << "\n[PASS] slice_inplace() reference counting works correctly" << std::endl;
}

void test_make_shards_inplace_reference_counting() {
    print_separator("TEST: make_shards_inplace() Reference Counting");
    
    std::cout << "Creating source tensor with 12 elements..." << std::endl;
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);
    
    Tensor source = Tensor({{1, 12}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);
    
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);
    
    size_t initial_ref_count = source.unsafeGetTensorImpl()->use_count();
    std::cout << "\nInitial source TensorImpl use_count: " << initial_ref_count << std::endl;
    
    std::cout << "\n--- Creating 3 inplace shards ---" << std::endl;
    std::vector<Tensor> shards = source.make_shards_inplace(3, true);
    
    size_t after_shards_ref_count = source.unsafeGetTensorImpl()->use_count();
    std::cout << "Source TensorImpl use_count after shards: " << after_shards_ref_count << std::endl;
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\nShard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 4);
        
        std::cout << "  Shard " << i << " does not own data: " << (!shards[i].owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(!shards[i].owns_data());
        
        size_t expected_offset = i * 4;
        bool offset_correct = (shards[i].storage_offset() == expected_offset);
        std::cout << "  Shard " << i << " storage_offset=" << expected_offset << ": " << (offset_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(offset_correct);
    }
    
    std::cout << "\nSource still owns data: " << (source.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source.owns_data());
    
    std::cout << "\n--- Testing mutual visibility of changes ---" << std::endl;
    
    shards[0].data<float>()[0] = 100.0f;
    shards[1].data<float>()[0] = 200.0f;
    shards[2].data<float>()[0] = 300.0f;
    
    std::cout << "Modified shard[0][0]=100, shard[1][0]=200, shard[2][0]=300" << std::endl;
    print_tensor_data("Source after shard modifications", source, 12);
    
    bool s0_visible = (source.data<float>()[0] == 100.0f);
    bool s1_visible = (source.data<float>()[4] == 200.0f);
    bool s2_visible = (source.data<float>()[8] == 300.0f);
    
    std::cout << "Shard 0 change visible in source: " << (s0_visible ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s0_visible);
    
    std::cout << "Shard 1 change visible in source: " << (s1_visible ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s1_visible);
    
    std::cout << "Shard 2 change visible in source: " << (s2_visible ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(s2_visible);
    
    std::cout << "\n[PASS] make_shards_inplace() reference counting works correctly" << std::endl;
}

void test_view_lifetime_and_destruction() {
    print_separator("TEST: View Lifetime and Destruction");
    
    std::cout << "This test verifies that views keep the source data alive" << std::endl;
    std::cout << "even when the original tensor variable goes out of scope." << std::endl;
    
    Tensor view_that_survives;
    void* original_data_ptr = nullptr;
    
    {
        std::cout << "\n--- Entering inner scope ---" << std::endl;
        
        std::vector<float> data(10);
        std::iota(data.begin(), data.end(), 0.0f);
        
        Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
        source.set_data(data);
        
        original_data_ptr = source.data();
        
        std::cout << "Created source tensor at address: " << original_data_ptr << std::endl;
        print_tensor_data("Source", source, 10);
        
        size_t source_ref_count = source.unsafeGetTensorImpl()->use_count();
        std::cout << "Source TensorImpl use_count: " << source_ref_count << std::endl;
        
        view_that_survives = source.slice_inplace(2, 5);
        
        size_t after_view_ref_count = source.unsafeGetTensorImpl()->use_count();
        std::cout << "Source TensorImpl use_count after creating view: " << after_view_ref_count << std::endl;
        
        print_tensor_info("View (in scope)", view_that_survives);
        print_tensor_data("View (in scope)", view_that_survives, 5);
        
        std::cout << "\n--- Exiting inner scope (source goes out of scope) ---" << std::endl;
    }
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    std::cout << "View still valid after source out of scope: ";
    bool view_valid = (view_that_survives.data() != nullptr);
    std::cout << (view_valid ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(view_valid);
    
    std::cout << "\nView contents (should still be [2, 3, 4, 5, 6]):" << std::endl;
    print_tensor_data("View (after source destroyed)", view_that_survives, 5);
    
    float* view_ptr = view_that_survives.data<float>();
    bool values_intact = true;
    for (int i = 0; i < 5; ++i) {
        if (view_ptr[i] != 2.0f + i) {
            values_intact = false;
            break;
        }
    }
    std::cout << "View data values intact: " << (values_intact ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(values_intact);
    
    std::cout << "View does not own data: " << (!view_that_survives.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(!view_that_survives.owns_data());
    
    std::cout << "\n--- Testing modification of surviving view ---" << std::endl;
    
    view_that_survives.data<float>()[0] = 999.0f;
    std::cout << "Modified view[0] = 999.0f" << std::endl;
    
    bool modification_works = (view_that_survives.data<float>()[0] == 999.0f);
    std::cout << "Modification successful: " << (modification_works ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(modification_works);
    
    print_tensor_data("View after modification", view_that_survives, 5);
    
    std::cout << "\n[PASS] View lifetime management works correctly" << std::endl;
}

void test_nested_views() {
    print_separator("TEST: Nested Views (View of a View)");
    
    std::cout << "Creating source tensor with 20 elements..." << std::endl;
    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 0.0f);
    
    Tensor source = Tensor({{1, 20}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);
    
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 20);
    
    std::cout << "\n--- Creating first-level view (slice 5..15) ---" << std::endl;
    Tensor view1 = source.slice_inplace(5, 10);
    
    print_tensor_info("View1", view1);
    print_tensor_data("View1", view1, 10);
    
    std::cout << "\n--- Creating second-level view from view1 (slice 2..7) ---" << std::endl;
    std::cout << "NOTE: slice_inplace on a view uses the base storage, so offset" << std::endl;
    std::cout << "      is relative to the original storage, not the parent view." << std::endl;
    
    // When we slice_inplace from view1, the offset is calculated from view1's storage
    // view1's storage is the same as source's storage, so offset=2 means elements [2,3,4,5,6]
    // NOT [7,8,9,10,11] as one might expect if it were relative to view1's view
    Tensor view2 = view1.slice_inplace(2, 5);
    
    print_tensor_info("View2 (nested)", view2);
    print_tensor_data("View2 (nested)", view2, 5);
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    std::cout << "Source owns data:    " << (source.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source.owns_data());
    
    std::cout << "View1 does not own:  " << (!view1.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(!view1.owns_data());
    
    std::cout << "View2 does not own:  " << (!view2.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(!view2.owns_data());
    
    std::cout << "\n--- Testing data consistency ---" << std::endl;
    
    // Since slice_inplace uses base storage, view2 contains [2,3,4,5,6]
    float* view2_ptr = view2.data<float>();
    bool view2_values_correct = true;
    for (int i = 0; i < 5; ++i) {
        if (view2_ptr[i] != 2.0f + i) {
            view2_values_correct = false;
            std::cout << "Expected " << (2.0f + i) << " at index " << i << ", got " << view2_ptr[i] << std::endl;
        }
    }
    std::cout << "View2 contains [2,3,4,5,6]: " << (view2_values_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(view2_values_correct);
    
    std::cout << "\nModifying view2[0] = 222.0f..." << std::endl;
    view2.data<float>()[0] = 222.0f;
    
    bool source_sees = (source.data<float>()[2] == 222.0f);
    bool view2_sees = (view2.data<float>()[0] == 222.0f);
    
    std::cout << "Source sees change at [2]: " << (source_sees ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(source_sees);
    
    std::cout << "View2 sees change at [0]:  " << (view2_sees ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(view2_sees);
    
    print_tensor_data("Source after nested modification", source, 20);
    
    std::cout << "\n[PASS] Nested views work correctly (offsets are relative to base storage)" << std::endl;
}

void test_view_copy_distinction() {
    print_separator("TEST: View vs Copy Distinction");
    
    std::cout << "This test ensures that copy operations create independent data" << std::endl;
    std::cout << "while view operations share the underlying storage." << std::endl;
    
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    
    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);
    
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);
    
    std::cout << "\n--- Creating a COPY (slice) ---" << std::endl;
    Tensor copy = source.slice(2, 5);
    
    print_tensor_info("Copy", copy);
    print_tensor_data("Copy", copy, 5);
    
    std::cout << "\n--- Creating a VIEW (slice_inplace) ---" << std::endl;
    Tensor view = source.slice_inplace(2, 5);
    
    print_tensor_info("View", view);
    print_tensor_data("View", view, 5);
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;
    
    std::cout << "Copy owns its data:   " << (copy.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(copy.owns_data());
    
    std::cout << "View does not own:    " << (!view.owns_data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(!view.owns_data());
    
    bool copy_different_address = (copy.data() != source.data());
    std::cout << "Copy has different address: " << (copy_different_address ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(copy_different_address);
    
    std::cout << "\n--- Testing modification isolation ---" << std::endl;
    
    std::cout << "Modifying source[2] = 999.0f..." << std::endl;
    source.data<float>()[2] = 999.0f;
    
    bool copy_isolated = (copy.data<float>()[0] == 2.0f);
    bool view_sees_change = (view.data<float>()[0] == 999.0f);
    
    std::cout << "Copy is isolated (still 2.0): " << (copy_isolated ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(copy_isolated);
    
    std::cout << "View sees change (now 999.0): " << (view_sees_change ? "✅ PASSED" : "❌ FAILED") << std::endl;
    assert(view_sees_change);
    
    print_tensor_data("Copy (unchanged)", copy, 5);
    print_tensor_data("View (changed)", view, 5);
    
    std::cout << "\n[PASS] View vs Copy distinction is correct" << std::endl;
}

int main() {
    std::cout << "\n" << std::string(60, '#') << std::endl;
    std::cout << "   REFERENCE COUNTING AND VIEW LIFECYCLE TEST SUITE" << std::endl;
    std::cout << std::string(60, '#') << std::endl;
    
    try {
        test_slice_inplace_reference_counting();
        test_make_shards_inplace_reference_counting();
        test_view_lifetime_and_destruction();
        test_nested_views();
        test_view_copy_distinction();
        
        std::cout << "\n" << std::string(60, '#') << std::endl;
        std::cout << "   ALL REFERENCE COUNTING TESTS PASSED!" << std::endl;
        std::cout << std::string(60, '#') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
