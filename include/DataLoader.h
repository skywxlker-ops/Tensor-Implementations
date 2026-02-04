#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "TensorLib.h"

namespace fs = std::filesystem;

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    if (!fs::exists(root)) return shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(uint16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ / 2;
        tokens_ = std::min(total_tokens, max_tokens);

        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    void close() {
        if (data_) { ::munmap(data_, file_bytes_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        file_bytes_ = 0; tokens_ = 0; path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

    void read_block(size_t start, size_t count, std::vector<uint16_t>& out) const {
        out.resize(count);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(data_);
        for (size_t i = 0; i < count; ++i) {
            out[i] = p[(start + i) % tokens_];
        }
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};

struct Batch {
    int B = 0, T = 0;
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T, int rank, int world_size, const std::string& split, const std::string& data_root, size_t max_tokens_per_shard = 400000000)
        : B_(B), T_(T), rank_(rank), world_(world_size), split_(split), root_(data_root), max_tokens_(max_tokens_per_shard) {
        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty()) throw std::runtime_error("no shards found in " + root_);
        reset();
    }

    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        if (pos_ + BT + 1 > shard_.size_tokens()) advance_shard();

        std::vector<uint16_t> buf;
        shard_.read_block(pos_, BT + 1, buf);

        Batch b;
        b.B = B_; b.T = T_;
        
        std::vector<uint16_t> x(BT), y(BT);
        for (size_t i = 0; i < BT; ++i) { x[i] = buf[i]; y[i] = buf[i + 1]; }

        OwnTensor::Device dev = OwnTensor::device::cuda_available() ? OwnTensor::Device::CUDA : OwnTensor::Device::CPU;
        
        b.input = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(dev, 0)});
        b.input.set_data(x);
        
        b.target = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(dev, 0)});
        b.target.set_data(y);

        pos_ += BT * static_cast<size_t>(world_);
        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    int B_, T_, rank_, world_;
    std::string split_, root_;
    size_t max_tokens_, current_shard_ = 0, pos_ = 0;
    std::vector<std::string> shards_;
    UInt16ShardView shard_;
};
