#pragma once
/**
 * CachingCUDAAllocator - Bridges DTensor's CachingAllocator to TensorLib's Allocator interface
 * 
 * This enables memory pooling for all OwnTensor::Tensor allocations by wrapping
 * the CachingAllocator in an Allocator-compatible interface.
 */

#include "device/Allocator.h"
#include "device/DeviceCore.h"
#include "memory/cachingAllocator.hpp"
#include <unordered_map>
#include <mutex>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {

extern CachingAllocator gAllocator;

class CachingCUDAAllocator : public Allocator {
public:
    CachingCUDAAllocator() = default;
    ~CachingCUDAAllocator() = default;
    
    void* allocate(size_t bytes) override {
#ifdef WITH_CUDA
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
        Block* block = gAllocator.allocateMemory(bytes, stream);
        if (!block || !block->addr) {
            throw std::runtime_error("CachingCUDAAllocator: allocation failed");
        }
        
        // Store block pointer for deallocation lookup
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            ptr_to_block_[block->addr] = block;
        }
        
        return block->addr;
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    void deallocate(void* ptr) override {
#ifdef WITH_CUDA
        if (!ptr) return;
        
        Block* block = nullptr;
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = ptr_to_block_.find(ptr);
            if (it != ptr_to_block_.end()) {
                block = it->second;
                ptr_to_block_.erase(it);
            }
        }
        
        if (block) {
            gAllocator.freeMemory(block);
        } else {
            // Fallback to cudaFree if not from our pool
            cudaFree(ptr);
        }
#endif
    }
    
    void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override {
#ifdef WITH_CUDA
        if (bytes > 0 && ptr == nullptr) {
            std::cerr << "[ERROR] CachingCUDAAllocator: memsetAsync called with NULL pointer and bytes=" << bytes << std::endl;
        }
        int device = -1;
        cudaGetDevice(&device);
        cudaError_t err = cudaMemsetAsync(ptr, value, bytes, stream);
        if (err != cudaSuccess) {
            cudaPointerAttributes attr;
            cudaError_t attr_err = cudaPointerGetAttributes(&attr, ptr);
            std::cerr << "[ERROR] cudaMemsetAsync failed for ptr=" << ptr << ", value=" << value << ", bytes=" << bytes << ", stream=" << stream << ", device=" << device << " (" << cudaGetErrorString(err) << ")" << std::endl;
            if (attr_err == cudaSuccess) {
                std::cerr << "[DEBUG] Pointer attributes: device=" << attr.device << ", type=" << (int)attr.type << std::endl;
            } else {
                std::cerr << "[DEBUG] cudaPointerGetAttributes failed: " << cudaGetErrorString(attr_err) << std::endl;
            }
            
            // Try sync memset as fallback/diagnostic
            cudaError_t err2 = cudaMemset(ptr, value, bytes);
            if (err2 != cudaSuccess) {
                std::cerr << "[ERROR] Synchronous cudaMemset also failed: " << cudaGetErrorString(err2) << std::endl;
            } else {
                std::cerr << "[INFO] Synchronous cudaMemset SUCCEEDED!" << std::endl;
            }
            
            throw std::runtime_error(std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(err));
        }
#endif
    }
    
    void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override {
#ifdef WITH_CUDA
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
        }
#endif
    }
    
    // Access underlying allocator for stats
    CachingAllocator& get_allocator() { return gAllocator; }
    const CachingAllocator& get_allocator() const { return gAllocator; }
    
    void printStats() const { gAllocator.printStats(); }
    size_t memoryAllocated() const { return gAllocator.memoryAllocated(); }
    size_t memoryFree() const { return gAllocator.memoryFree(); }

private:
    // Uses global gAllocator defined in DTensor
    std::unordered_map<void*, Block*> ptr_to_block_;
    mutable std::mutex map_mutex_;
};

} // namespace OwnTensor
