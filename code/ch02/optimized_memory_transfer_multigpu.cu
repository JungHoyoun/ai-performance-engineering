// optimized_memory_transfer_multigpu.cu - GPU-to-GPU P2P transfers (NVLink/NVSwitch optimized).
// Uses peer-to-peer copies to avoid host staging.
// Compile: nvcc -O3 -std=c++17 -arch=sm_121 optimized_memory_transfer_multigpu.cu -o optimized_memory_transfer_multigpu_sm121

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

static void enable_peer_access(int src, int dst) {
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src, dst));
    if (!can_access) {
        std::printf("SKIPPED: P2P access not supported between GPU %d and GPU %d\n", src, dst);
        std::exit(EXIT_SUCCESS);
    }
    CUDA_CHECK(cudaSetDevice(src));
    cudaError_t status = cudaDeviceEnablePeerAccess(dst, 0);
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(status);
    }
}

int main() {
    NVTX_RANGE("main");
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

    std::printf("=== Optimized: GPU-to-GPU P2P Transfers ===\n");
    std::printf("Devices: %d -> %d\n", src_device, dst_device);
    std::printf("Array size: %zu elements (%.1f MB)\n\n", N, bytes / 1e6);

    float *d_src = nullptr;
    float *d_dst = nullptr;

    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, bytes));

    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));

    enable_peer_access(src_device, dst_device);
    enable_peer_access(dst_device, src_device);

    // Warmup
    CUDA_CHECK(cudaMemcpyPeer(d_dst, dst_device, d_src, src_device, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        NVTX_RANGE("transfer_sync");
        CUDA_CHECK(cudaMemcpyPeer(d_dst, dst_device, d_src, src_device, bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;
    double bandwidth_gbs = (bytes / 1e9) / (avg_ms / 1000.0);

    std::printf("Average time per iteration: %.3f ms\n", avg_ms);
    std::printf("Bandwidth: %.2f GB/s (P2P)\n", bandwidth_gbs);

    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaFree(d_dst));

    return 0;
}
