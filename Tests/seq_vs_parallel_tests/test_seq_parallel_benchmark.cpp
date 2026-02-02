#include "core/Tensor.h"
#include "autograd/Engine.h"
#include "ops/TensorOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/BinaryBackward.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <sys/resource.h>
#include <unistd.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// ============================================================================
// Memory and Time Measurement Utilities
// ============================================================================

long get_current_rss() {
    long rss = 0;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        long size, resident, share, text, lib, data, dt;
        if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld", &size, &resident, &share, &text, &lib, &data, &dt) == 7) {
            rss = resident * sysconf(_SC_PAGESIZE) / 1024;
        }
        fclose(fp);
    }
    return rss;
}

size_t get_cuda_used_memory() {
#ifdef WITH_CUDA
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) return 0;
    return total - free;
#else
    return 0;
#endif
}

class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

struct PerformanceMetrics {
    double time_ms_mean;
    double time_ms_stddev;
    long peak_rss_kb;
    size_t peak_cuda_bytes;
    
    void print(const std::string& label) const {
        std::cout << label << ":\n";
        std::cout << "  Time:        " << std::fixed << std::setprecision(2) 
                  << time_ms_mean << " ± " << time_ms_stddev << " ms\n";
        std::cout << "  Peak RSS:    " << std::fixed << std::setprecision(2) 
                  << peak_rss_kb / 1024.0 << " MB\n";
        if (peak_cuda_bytes > 0) {
            std::cout << "  Peak CUDA:   " << std::fixed << std::setprecision(2) 
                      << peak_cuda_bytes / (1024.0 * 1024.0) << " MB\n";
        }
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

Tensor add_autograd(const Tensor& a, const Tensor& b) {
    auto fwd = [](const Tensor& x, const Tensor& y) { return x + y; };
    return autograd::make_binary_op<autograd::AddBackward>(a, b, fwd, a, b);
}

template<typename BenchFunc>
PerformanceMetrics run_with_stats(BenchFunc benchmark_func, int num_runs = 100, int warmup = 10) {
    std::vector<double> timings;
    long peak_rss = 0;
    size_t peak_cuda = 0;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        benchmark_func();
    }
    
    // Actual runs
    for (int i = 0; i < num_runs; ++i) {
        auto [time, rss, cuda] = benchmark_func();
        timings.push_back(time);
        if (rss > peak_rss) peak_rss = rss;
        if (cuda > peak_cuda) peak_cuda = cuda;
    }
    
    // Calculate statistics
    double mean = 0.0;
    for (double t : timings) mean += t;
    mean /= timings.size();
    
    double variance = 0.0;
    for (double t : timings) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= timings.size();
    double stddev = std::sqrt(variance);
    
    return {mean, stddev, peak_rss, peak_cuda};
}

// ============================================================================
// Benchmark Implementations
// ============================================================================

std::tuple<double, long, size_t> simple_arithmetic_test(
    DeviceIndex device, ExecutionMode mode, int iters) {
    set_execution_mode(mode);
    TensorOptions opts;
    opts.device = device;
    
    auto x = Tensor::randn(Shape{{100, 100}}, opts, 42, 1.0f);
    x.set_requires_grad(true);
    auto y = Tensor::full(Shape{{100, 100}}, opts, 2.0f);
    
    Timer timer;
    long peak_rss = 0;
    size_t peak_cuda = 0;
    
    timer.start();
    for (int i = 0; i < iters; ++i) {
        x.zero_grad();
        auto z = add_autograd(x, y);
        auto loss = autograd::sum(z);
        loss.backward();
    }
    peak_rss = get_current_rss();
    peak_cuda = 0;
    if (device.is_cuda()) peak_cuda = get_cuda_used_memory();
    return {timer.elapsed_ms(), peak_rss, peak_cuda};
}

std::tuple<double, long, size_t> matmul_test(
    DeviceIndex device, ExecutionMode mode, int iters) {
    set_execution_mode(mode);
    TensorOptions opts;
    opts.device = device;
    
    auto x = Tensor::randn(Shape{{64, 128}}, opts, 42, 1.0f);
    x.set_requires_grad(true);
    auto w = Tensor::randn(Shape{{128, 256}}, opts, 43, 1.0f);
    w.set_requires_grad(true);
    
    Timer timer;
    long peak_rss = 0;
    size_t peak_cuda = 0;
    
    timer.start();
    for (int i = 0; i < iters; ++i) {
        x.zero_grad();
        w.zero_grad();
        auto out = matmul(x, w);
        auto loss = autograd::sum(out);
        loss.backward();
    }
    peak_rss = get_current_rss();
    peak_cuda = 0;
    if (device.is_cuda()) peak_cuda = get_cuda_used_memory();
    return {timer.elapsed_ms(), peak_rss, peak_cuda};
}

std::tuple<double, long, size_t> mlp_test(
    DeviceIndex device, ExecutionMode mode, int iters) {
    set_execution_mode(mode);
    TensorOptions opts;
    opts.device = device;
    
    auto x = Tensor::randn(Shape{{32, 128}}, opts, 42, 1.0f);
    x.set_requires_grad(true);
    auto w1 = Tensor::randn(Shape{{128, 256}}, opts, 43, 1.0f);
    w1.set_requires_grad(true);
    auto b1 = Tensor::randn(Shape{{256}}, opts, 44, 1.0f);
    b1.set_requires_grad(true);
    auto w2 = Tensor::randn(Shape{{256, 64}}, opts, 45, 1.0f);
    w2.set_requires_grad(true);
    auto b2 = Tensor::randn(Shape{{64}}, opts, 46, 1.0f);
    b2.set_requires_grad(true);
    
    Timer timer;
    long peak_rss = 0;
    size_t peak_cuda = 0;
    
    timer.start();
    for (int i = 0; i < iters; ++i) {
        x.zero_grad();
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();
        
        auto h1 = add_autograd(matmul(x, w1), b1);
        auto h1_relu = relu(h1);
        auto out = add_autograd(matmul(h1_relu, w2), b2);
        auto loss = autograd::sum(out);
        loss.backward();
    }
    peak_rss = get_current_rss();
    peak_cuda = 0;
    if (device.is_cuda()) peak_cuda = get_cuda_used_memory();
    return {timer.elapsed_ms(), peak_rss, peak_cuda};
}

std::tuple<double, long, size_t> deep_network_test(
    DeviceIndex device, ExecutionMode mode, int iters) {
    set_execution_mode(mode);
    TensorOptions opts;
    opts.device = device;
    
    auto x = Tensor::randn(Shape{{16, 512}}, opts, 42, 1.0f);
    x.set_requires_grad(true);
    auto w1 = Tensor::randn(Shape{{512, 1024}}, opts, 43, 1.0f);
    w1.set_requires_grad(true);
    auto b1 = Tensor::randn(Shape{{1024}}, opts, 44, 1.0f);
    b1.set_requires_grad(true);
    auto w2 = Tensor::randn(Shape{{1024, 512}}, opts, 45, 1.0f);
    w2.set_requires_grad(true);
    auto b2 = Tensor::randn(Shape{{512}}, opts, 46, 1.0f);
    b2.set_requires_grad(true);
    auto w3 = Tensor::randn(Shape{{512, 256}}, opts, 47, 1.0f);
    w3.set_requires_grad(true);
    auto b3 = Tensor::randn(Shape{{256}}, opts, 48, 1.0f);
    b3.set_requires_grad(true);
    auto w4 = Tensor::randn(Shape{{256, 128}}, opts, 49, 1.0f);
    w4.set_requires_grad(true);
    auto b4 = Tensor::randn(Shape{{128}}, opts, 50, 1.0f);
    b4.set_requires_grad(true);
    
    Timer timer;
    long peak_rss = 0;
    size_t peak_cuda = 0;
    
    timer.start();
    for (int i = 0; i < iters; ++i) {
        x.zero_grad();
        w1.zero_grad(); b1.zero_grad();
        w2.zero_grad(); b2.zero_grad();
        w3.zero_grad(); b3.zero_grad();
        w4.zero_grad(); b4.zero_grad();
        
        auto h1 = relu(add_autograd(matmul(x, w1), b1));
        auto h2 = relu(add_autograd(matmul(h1, w2), b2));
        auto h3 = relu(add_autograd(matmul(h2, w3), b3));
        auto out = add_autograd(matmul(h3, w4), b4);
        auto loss = autograd::sum(out);
        loss.backward();
    }
    peak_rss = get_current_rss();
    peak_cuda = 0;
    if (device.is_cuda()) peak_cuda = get_cuda_used_memory();
    return {timer.elapsed_ms(), peak_rss, peak_cuda};
}

// ============================================================================
// Reporting
// ============================================================================

void print_comparison(const std::string& test_name, 
                     const PerformanceMetrics& seq_metrics,
                     const PerformanceMetrics& par_metrics) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << test_name << "\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Sequential Mode
    std::cout << "Sequential Mode:\n";
    std::cout << "  Time:       " << std::fixed << std::setprecision(2) 
              << seq_metrics.time_ms_mean << " ms\n";
    std::cout << "  Peak RSS:   " << std::fixed << std::setprecision(2) 
              << seq_metrics.peak_rss_kb / 1024.0 << " MB\n";
    if (seq_metrics.peak_cuda_bytes > 0) {
        std::cout << "  Peak CUDA:  " << std::fixed << std::setprecision(2) 
                  << seq_metrics.peak_cuda_bytes / (1024.0 * 1024.0) << " MB\n";
    }
    
    std::cout << "\n";
    
    // Parallel Mode
    std::cout << "Parallel Mode:\n";
    std::cout << "  Time:       " << std::fixed << std::setprecision(2) 
              << par_metrics.time_ms_mean << " ms\n";
    std::cout << "  Peak RSS:   " << std::fixed << std::setprecision(2) 
              << par_metrics.peak_rss_kb / 1024.0 << " MB\n";
    if (par_metrics.peak_cuda_bytes > 0) {
        std::cout << "  Peak CUDA:  " << std::fixed << std::setprecision(2) 
                  << par_metrics.peak_cuda_bytes / (1024.0 * 1024.0) << " MB\n";
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
}

void run_benchmark_suite(DeviceIndex device) {
    std::string device_name = device.is_cuda() ? "GPU" : "CPU";
    std::cout << "\n" << std::string(80, '#') << "\n";
    std::cout << "RUNNING BENCHMARK SUITE ON " << device_name << "\n";
    std::cout << "(100 runs each + 10 warmup runs for statistical stability)\n";
    std::cout << std::string(80, '#') << "\n";
    
    int iters_short = 200;  // More iterations for fast ops
    int iters_medium = 100;
    int iters_long = 50;
    
    // Test 1
    {
        std::cout << "\n[1/4] Benchmarking Simple Arithmetic...\n";
        auto seq = run_with_stats([&](){ return simple_arithmetic_test(device, ExecutionMode::SEQUENTIAL, iters_short); });
        auto par = run_with_stats([&](){ return simple_arithmetic_test(device, ExecutionMode::PARALLEL, iters_short); });
        print_comparison("Simple Arithmetic (" + std::to_string(iters_short) + " iterations)", seq, par);
    }
    
    // Test 2
    {
        std::cout << "\n[2/4] Benchmarking Matrix Multiplication...\n";
        auto seq = run_with_stats([&](){ return matmul_test(device, ExecutionMode::SEQUENTIAL, iters_medium); });
        auto par = run_with_stats([&](){ return matmul_test(device, ExecutionMode::PARALLEL, iters_medium); });
        print_comparison("Matrix Multiplication (" + std::to_string(iters_medium) + " iterations)", seq, par);
    }
    
    // Test 3
    {
        std::cout << "\n[3/4] Benchmarking MLP (2 layers)...\n";
        auto seq = run_with_stats([&](){ return mlp_test(device, ExecutionMode::SEQUENTIAL, iters_medium); });
        auto par = run_with_stats([&](){ return mlp_test(device, ExecutionMode::PARALLEL, iters_medium); });
        print_comparison("MLP - 2 Layers (" + std::to_string(iters_medium) + " iterations)", seq, par);
    }
    
    // Test 4
    {
        std::cout << "\n[4/4] Benchmarking Deep Network (4 layers)...\n";
        auto seq = run_with_stats([&](){ return deep_network_test(device, ExecutionMode::SEQUENTIAL, iters_long); });
        auto par = run_with_stats([&](){ return deep_network_test(device, ExecutionMode::PARALLEL, iters_long); });
        print_comparison("Deep Network - 4 Layers (" + std::to_string(iters_long) + " iterations)", seq, par);
    }
}

int main() {
    std::cout << std::string(80, '#') << "\n";
    std::cout << "SEQUENTIAL vs PARALLEL MODE BENCHMARK\n";
    std::cout << "Time and Memory Comparison with Statistical Analysis\n";
    std::cout << std::string(80, '#') << "\n";
    
    try {
        std::cout << "\n>>> TESTING ON CPU <<<\n";
        DeviceIndex cpu_device(Device::CPU);
        run_benchmark_suite(cpu_device);
        
        if (OwnTensor::device::cuda_available()) {
            std::cout << "\n\n>>> TESTING ON GPU <<<\n";
            DeviceIndex gpu_device(Device::CUDA, 0);
            run_benchmark_suite(gpu_device);
        } else {
            std::cout << "\n\nCUDA is not available. Skipping GPU tests.\n";
        }
        
        std::cout << "\n" << std::string(80, '#') << "\n";
        std::cout << "ALL BENCHMARKS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(80, '#') << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nBenchmark failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
