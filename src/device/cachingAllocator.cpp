#include "memory/cachingAllocator.hpp"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iostream>

namespace OwnTensor {

CachingAllocator gAllocator;

CachingAllocator::CachingAllocator()
    : total_allocated_(0), total_free_(0) {}

CachingAllocator::~CachingAllocator() {
    try {
        emptyCache(OwnTensor::Device::CUDA);
    } catch (...) {
        // destructor must not throw
    }
}

size_t CachingAllocator::roundMemorySize(size_t size) const {
    if (size == 0) return 0;
    if (size < ONE_MB) {
        size_t rem = size % ALIGNMENT;
        return rem ? size + (ALIGNMENT - rem) : size;
    } else {
        size_t rem = size % ONE_MB;
        return rem ? size + (ONE_MB - rem) : size;
    }
}

PoolType CachingAllocator::selectPoolType(size_t size) const {
    return (size < ONE_MB) ? PoolType::SMALL : PoolType::LARGE;
}

Block* CachingAllocator::allocNewBlock(size_t size, cudaStream_t stream, PoolType pool_type) {
    // Get current device to track which GPU this block belongs to
    int current_device = 0;
    cudaGetDevice(&current_device);
    
    void* dev_ptr = nullptr;
    size_t rsize = roundMemorySize(size);
    cudaError_t err = cudaMalloc(&dev_ptr, rsize);
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "cudaMalloc failed (" << cudaGetErrorString(err) << ") for size " << rsize;
        throw std::runtime_error(os.str());
    }

    Block* b = new Block(dev_ptr, rsize, pool_type, stream, current_device);
    auto& internal = stream_to_cache_[stream];
    internal.all_blocks.push_back(b);
    total_allocated_ += rsize;
    return b;
}

Block* CachingAllocator::findBestFit(std::multiset<Block*, CompareBySize>& pool, size_t size) {
    if (pool.empty()) return nullptr;
    Block fake(nullptr, size, PoolType::SMALL, 0);
    auto it = pool.lower_bound(&fake);
    return (it == pool.end()) ? nullptr : *it;
}

Block* CachingAllocator::allocateMemory(size_t size, cudaStream_t stream) {
    if (size == 0) return nullptr;

    size_t rsize = roundMemorySize(size);
    PoolType pool_type = selectPoolType(rsize);

    std::lock_guard<std::mutex> lk(mutex_);
    auto& internal = stream_to_cache_[stream];
    auto& pool = (pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

    // Get current device to ensure we only reuse blocks from the same device
    int current_device = 0;
    cudaGetDevice(&current_device);
    
    // Find a suitable block that matches both size AND device
    Block* candidate = nullptr;
    typename std::multiset<Block*, CompareBySize>::iterator candidate_iter = pool.end();
    
    // Search for best-fit block on the correct device
    Block fake(nullptr, rsize, PoolType::SMALL, 0, 0);
    for (auto it = pool.lower_bound(&fake); it != pool.end(); ++it) {
        if ((*it)->device == current_device) {
            candidate = *it;
            candidate_iter = it;
            break;
        }
    }
    
    if (candidate && candidate_iter != pool.end()) {
        size_t cur_free = total_free_.load();
        total_free_ -= std::min(cur_free, candidate->size);
        pool.erase(candidate_iter);
        candidate->active = true;

        // Optional split for large blocks
        if (candidate->size > rsize + MIN_SPLIT) {
            size_t rem_size = candidate->size - rsize;
            void* rem_addr = reinterpret_cast<void*>(reinterpret_cast<char*>(candidate->addr) + rsize);
            candidate->size = rsize;
            Block* remainder = new Block(rem_addr, rem_size, pool_type, stream, current_device);
            
            // Link the remainder correctly
            remainder->next = candidate->next;
            remainder->prev = candidate;
            if (candidate->next) candidate->next->prev = remainder;
            candidate->next = remainder;

            internal.all_blocks.push_back(remainder);
            remainder->active = false;
            pool.insert(remainder);
            total_free_ += remainder->size;
        }

        return candidate;
    }

    // No free block — allocate fresh
    Block* nb = allocNewBlock(rsize, stream, pool_type);
    nb->active = true;
    return nb;
}

void CachingAllocator::freeMemory(Block* block) {
    if (!block) return;

    std::lock_guard<std::mutex> lk(mutex_);
    if (!block->active) {
        std::cerr << "[Allocator Warning] Double free on block " << block->addr << std::endl;
        return;
    }

    block->active = false;
    // Don't update total_free_ here, it will be done in pool.insert or mergeAdjacent

    auto it = stream_to_cache_.find(block->stream);
    if (it == stream_to_cache_.end()) return;
    StreamInternal& internal = it->second;
    auto& pool = (block->pool_type == PoolType::SMALL) ? internal.small_pool : internal.large_pool;

    block = mergeAdjacent(block, internal);
    block->active = false;
    pool.insert(block);
    total_free_ += block->size;
}

Block* CachingAllocator::mergeAdjacent(Block* block, StreamInternal& internal) {
    // Merge next first (to keep 'block' pointer valid for previous merge)
    if (block->next && !block->next->active) {
        Block* nxt = block->next;
        
        // Remove nxt from pool BEFORE modifying sizes or pointers
        if (nxt->pool_type == PoolType::SMALL) {
             auto it = internal.small_pool.find(nxt);
             if (it != internal.small_pool.end()) {
                 total_free_ -= nxt->size;
                 internal.small_pool.erase(it);
             }
        } else {
             auto it = internal.large_pool.find(nxt);
             if (it != internal.large_pool.end()) {
                 total_free_ -= nxt->size;
                 internal.large_pool.erase(it);
             }
        }

        // Now safe to merge and modify
        block->size += nxt->size;
        block->next = nxt->next;
        if (nxt->next) nxt->next->prev = block;
        
        // Remove from tracking to prevent double-free
        auto it_all = std::find(internal.all_blocks.begin(), internal.all_blocks.end(), nxt);
        if (it_all != internal.all_blocks.end()) internal.all_blocks.erase(it_all);
        
        delete nxt;
    }

    // Merge previous
    if (block->prev && !block->prev->active) {
        Block* prv = block->prev;

        // Remove prv from pool BEFORE modifying sizes or pointers
        if (prv->pool_type == PoolType::SMALL) {
            auto it = internal.small_pool.find(prv);
            if (it != internal.small_pool.end()) {
                total_free_ -= prv->size;
                internal.small_pool.erase(it);
            }
        } else {
            auto it = internal.large_pool.find(prv);
            if (it != internal.large_pool.end()) {
                total_free_ -= prv->size;
                internal.large_pool.erase(it);
            }
        }

        // Now safe to merge and modify
        prv->size += block->size;
        prv->next = block->next;
        if (block->next) block->next->prev = prv;

        // Remove 'block' from all_blocks
        auto it_all = std::find(internal.all_blocks.begin(), internal.all_blocks.end(), block);
        if (it_all != internal.all_blocks.end()) internal.all_blocks.erase(it_all);
        
        delete block;
        block = prv;
    }
    return block;
}

void CachingAllocator::emptyCache(OwnTensor::Device device) {
    if (device == OwnTensor::Device::CPU) return;

    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& entry : stream_to_cache_) {
        StreamInternal& internal = entry.second;
        for (Block* b : internal.all_blocks) {
            if (b->addr) cudaFree(b->addr);
            delete b;
        }
        internal.all_blocks.clear();
        internal.small_pool.clear();
        internal.large_pool.clear();
    }
    stream_to_cache_.clear();
    total_allocated_ = 0;
    total_free_ = 0;
}

size_t CachingAllocator::memoryAllocated() const {
    return total_allocated_.load(std::memory_order_relaxed);
}

size_t CachingAllocator::memoryFree() const {
    return total_free_.load(std::memory_order_relaxed);
}

bool CachingAllocator::isAllocated(Block* block) const {
    return block && block->active;
}

void CachingAllocator::printStats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::cout << "\n[GPU CachingAllocator Stats]\n";
    std::cout << "  Total Allocated: " << memoryAllocated() << " bytes\n";
    std::cout << "  Total Free:      " << memoryFree() << " bytes\n";
    std::cout << "  Streams Cached:  " << stream_to_cache_.size() << "\n";
    for (auto& kv : stream_to_cache_) {
        const StreamInternal& s = kv.second;
        std::cout << "    Stream[" << reinterpret_cast<std::uintptr_t>(kv.first)
                  << "]: small=" << s.small_pool.size()
                  << ", large=" << s.large_pool.size()
                  << ", total_blocks=" << s.all_blocks.size() << "\n";
    }
}

} // namespace OwnTensor
