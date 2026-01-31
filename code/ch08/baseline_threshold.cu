// Baseline threshold binary using branch-heavy kernel.

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "threshold_common.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace ch08;

int main() {
    NVTX_RANGE("main");
    const int count = 1 << 26;  // 64M elements
    const float threshold = 0.5f;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);

    std::vector<float> h_input(count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; ++i) {
        NVTX_RANGE("setup");
        h_input[i] = dist(gen);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        NVTX_RANGE("warmup");
        launch_threshold_naive(d_input, d_output, threshold, count, 0);
    }
    cudaDeviceSynchronize();

    const int iterations = 50;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        launch_threshold_naive(d_input, d_output, threshold, count, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / iterations;

    std::cout << "=== Baseline Threshold (branch divergence) ===\n";
    std::cout << "Elements: " << count << " (" << bytes / 1e6 << " MB)\n";
    std::cout << "Average kernel time: " << avg_ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
