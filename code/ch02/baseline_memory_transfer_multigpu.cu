// baseline_memory_transfer_multigpu.cu - GPU-to-GPU transfer via host staging (baseline).
// Demonstrates PCIe-limited transfers by bouncing through host memory.
// Compile: nvcc -O3 -std=c++17 -arch=sm_121 baseline_memory_transfer_multigpu.cu -o baseline_memory_transfer_multigpu_sm121

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::printf("SKIPPED: requires >=2 GPUs\n");
        return 0;
    }

    const int src_device = 0;
    const int dst_device = 1;
    const size_t N = 100 * 1024 * 1024;  // 100M elements
    const size_t bytes = N * sizeof(float);
    const int iterations = 100;

    std::printf("=== Baseline: GPU-to-GPU via Host Staging (PCIe) ===\n");
    std::printf("Devices: %d -> %d\n", src_device, dst_device);
    std::printf("Array size: %zu elements (%.1f MB)\n\n", N, bytes / 1e6);

    float *d_src = nullptr;
    float *d_dst = nullptr;
    float *h_buffer = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_buffer, bytes));

    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, bytes));

    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaMemcpy(h_buffer, d_src, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaMemcpy(d_dst, h_buffer, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(cudaSetDevice(src_device));
        CUDA_CHECK(cudaMemcpy(h_buffer, d_src, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(dst_device));
        CUDA_CHECK(cudaMemcpy(d_dst, h_buffer, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;
    double bandwidth_gbs = (2.0 * bytes / 1e9) / (avg_ms / 1000.0);  // D2H + H2D

    std::printf("Average time per iteration: %.3f ms\n", avg_ms);
    std::printf("Bandwidth: %.2f GB/s (host-staged)\n", bandwidth_gbs);

    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_buffer));

    return 0;
}
