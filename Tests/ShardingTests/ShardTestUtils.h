#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <cstdint>
#include "core/Tensor.h"

using namespace OwnTensor;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_tensor_info(const std::string& name, Tensor& t) {
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Device:                     " << (t.is_cpu() ? "CPU" : "CUDA") << std::endl;
    std::cout << "    Address:                    " << t.data() << std::endl;
    std::cout << "    Numel:                      " << t.numel() << std::endl;
    std::cout << "    Nbytes:                     " << t.nbytes() << " Bytes" << std::endl;
    std::cout << "    Actual Bytes Allocated:     " << t.allocated_bytes() << " Bytes" << std::endl;
    std::cout << "    Storage Offset:             " << t.storage_offset() << std::endl;
    std::cout << "    Owns Data:                  " << (t.owns_data() ? "yes" : "no") << std::endl;
    std::cout << "    Is Contiguous:              " << (t.is_contiguous() ? "yes" : "no") << std::endl;
}

void print_tensor_data(const std::string& name, Tensor& t, int max_elems = 10) {
    std::cout << "  " << name << " data: [";
    float* ptr = t.data<float>();
    int count = std::min((int)t.numel(), max_elems);
    for (int i = 0; i < count; ++i) {
        std::cout << ptr[i] << (i < count - 1 ? ", " : "");
    }
    if ((int)t.numel() > max_elems) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

bool is_address_between(const void* address_to_check, const void* start_address, const void* end_address) {
    // Pointers can be directly compared in C++.
    // This checks if 'address_to_check' is greater than or equal to 'start_address'
    // AND less than 'end_address'.
    return address_to_check >= start_address && address_to_check < end_address;
}