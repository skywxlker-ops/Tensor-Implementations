#include "TensorLib.h"
#include "ShardTestUtils.h"

using namespace OwnTensor;

void test_outplace_slice(int src_numel, int start, int slice_len)
{
    print_separator("TEST: COPY SLICE(START, LENGTH) - COPY BASED SLICING");

    std::vector<float> data(src_numel);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, src_numel}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor [0..19]:" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, src_numel);

    Tensor sliced = source.slice(start, slice_len);

    std::cout << "\nSliced tensor (start=5, length=10):" << std::endl;
    print_tensor_info("Sliced", sliced);
    print_tensor_data("Sliced", sliced, slice_len);

    std::cout << "\n\n=============== ASSERTION CASES ===============" << std::endl;
    std::cout << "Address Check:         " << (is_address_between(sliced.data(), source.data(), source.data() + source.allocated_bytes())
                ? "❌ FAILED - NOT A COPY" : "✅ PASSED - IS A COPY")<< std::endl;
            
    std::cout << "Element Count:         " << (sliced.numel() == uint64_t(slice_len) ? "✅ PASSED " : "❌ FAILED ") << std::endl;
    std::cout << "Confirming Copy check: " << (sliced.owns_data() == true ? "✅ PASSED " : "❌ FAILED " )<< std::endl;

    std::cout << "Value Assertion Test for the Slice: ";       
    float* s_ptr = sliced.data<float>();
    for (int i = 0; i < slice_len; ++i) {
        assert(s_ptr[i] == float(start) + i);
    }
    std::cout << "✅ PASSED " << std::endl;

    std::cout << "\n[PASS] slice() creates independent copy with correct data" << std::endl;
    
}

void test_inplace_slice(int src_numel, int start, int slice_len)
{
    print_separator("TEST: COPY INPLACE(START, LENGTH) - VIEW BASED SLICING");

    std::vector<float> data(src_numel);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, src_numel}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor [0..19]:" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, src_numel);

    Tensor sliced = source.slice_inplace(start, slice_len);

    std::cout << "\nSliced tensor (start=5, length=10):" << std::endl;
    print_tensor_info("Sliced", sliced);
    print_tensor_data("Sliced", sliced, slice_len);

    std::cout << "\n\n=============== ASSERTION CASES ===============" << std::endl;
    std::cout << "Address Check:         " << (!is_address_between(sliced.data(), source.data(), source.data() + source.allocated_bytes()) 
                ? "❌ FAILED - A COPY" : "✅ PASSED - NOT A COPY")<< std::endl;
            
    std::cout << "Element Count:         " << (sliced.numel() == uint64_t(slice_len) ? "✅ PASSED " : "❌ FAILED ") << std::endl;
    std::cout << "Confirming Inplace check: " << (sliced.owns_data() == false ? "✅ PASSED " : "❌ FAILED " )<< std::endl;

    std::cout << "Value Assertion Test for the Slice: ";       
    float* s_ptr = sliced.data<float>();
    for (int i = 0; i < slice_len; ++i) {
        assert(s_ptr[i] == float(start) + i);
        assert(&s_ptr[i] == source.data() + (start + i) * 4);
    }
    std::cout << "✅ PASSED " << std::endl;

    std::cout << "\n[PASS] slice() creates independent copy with correct data" << std::endl;
    
}

int main()
{
    try {
        test_outplace_slice(15, 3, 8);
        test_inplace_slice(15, 3, 8);
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
    }
}