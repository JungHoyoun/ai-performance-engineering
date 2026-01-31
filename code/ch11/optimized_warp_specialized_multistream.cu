// Chapter 11: Book-aligned optimized version overlapping batches across CUDA streams.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdint>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace {
constexpr int TILE = 32;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int THREADS = 96;

__device__ void compute_tile(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int lane) {
    for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        int row = idx / TILE;
        int col = idx % TILE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += A[row * TILE + k] * B[k * TILE + col];
        }
        C[idx] = acc;
    }
}

__global__ void simple_warp_specialized_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C) {
    extern __shared__ float shared[];
    float* A_tile = shared;
    float* B_tile = shared + TILE_ELEMS;
    float* C_tile = shared + 2 * TILE_ELEMS;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (warp_id == 0) {
        for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
            A_tile[idx] = A[idx];
            B_tile[idx] = B[idx];
        }
    }

    __syncthreads();

    if (warp_id == 1) {
        compute_tile(A_tile, B_tile, C_tile, lane_id);
    }

    __syncthreads();

    if (warp_id == 2) {
        for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
            C[idx] = C_tile[idx];
        }
    }
}

void run_optimized() {
    constexpr int batches = 4096;
    // Increased from 8 to 16 streams for better overlap on modern GPUs
    // More streams allow better pipelining of H2D/compute/D2H operations
    constexpr int num_streams = 16;
    const size_t bytes = TILE_ELEMS * sizeof(float);

    // Use pinned host memory so H2D/D2H can overlap with compute.
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, static_cast<size_t>(batches) * TILE_ELEMS * sizeof(float));
    cudaMallocHost(&h_B, static_cast<size_t>(batches) * TILE_ELEMS * sizeof(float));
    cudaMallocHost(&h_C, static_cast<size_t>(batches) * TILE_ELEMS * sizeof(float));
    
    // Initialize host data.
    {
        NVTX_RANGE("setup:host_init");
        for (int i = 0; i < batches * TILE_ELEMS; ++i) {
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(i + 1);
            h_C[i] = 0.0f;
        }
    }

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Pre-allocate device memory pools per stream to avoid allocation overhead.
    float* dA_pool[num_streams];
    float* dB_pool[num_streams];
    float* dC_pool[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaMalloc(&dA_pool[i], bytes);
        cudaMalloc(&dB_pool[i], bytes);
        cudaMalloc(&dC_pool[i], bytes);
    }

    cudaDeviceSynchronize();
    const auto start = std::chrono::high_resolution_clock::now();

    // Launch batches across streams for maximum overlap
    // Operations on the same stream are serialized automatically by CUDA,
    // so reusing buffers is safe - each batch waits for previous batch on that stream
    for (int b = 0; b < batches; ++b) {
        const int stream_idx = b % num_streams;
        cudaStream_t st = streams[stream_idx];

        char batch_label[32];
        std::snprintf(batch_label, sizeof(batch_label), "batch:%d", b);
        NVTX_RANGE(batch_label);

        // Reuse pre-allocated device memory - CUDA serializes operations on same stream
        float* dA = dA_pool[stream_idx];
        float* dB = dB_pool[stream_idx];
        float* dC = dC_pool[stream_idx];

        // Launch async operations - these overlap across different streams
        // H2D transfers overlap with compute/kernels on other streams
        {
            NVTX_RANGE("transfer_async:h2d");
            cudaMemcpyAsync(dA, h_A + static_cast<size_t>(b) * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(dB, h_B + static_cast<size_t>(b) * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, st);
        }

        // Kernel execution overlaps with transfers/kernels on other streams
        {
            NVTX_RANGE("compute_kernel:warp_specialized");
            simple_warp_specialized_kernel<<<1, THREADS, 3 * bytes, st>>>(dA, dB, dC);
        }

        // D2H transfer overlaps with other streams' operations
        {
            NVTX_RANGE("transfer_async:d2h");
            cudaMemcpyAsync(h_C + static_cast<size_t>(b) * TILE_ELEMS, dC, bytes, cudaMemcpyDeviceToHost, st);
        }
    }

    // Wait for all streams to complete
    cudaDeviceSynchronize();
    const auto stop = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    double checksum = 0.0;
    {
        NVTX_RANGE("verify:checksum");
        for (int i = 0; i < batches * TILE_ELEMS; ++i) {
            checksum += h_C[i];
        }
    }

    const float verify_checksum = static_cast<float>(checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
    printf("TIME_MS: %.3f\n", ms);
    
    // Cleanup
    for (int i = 0; i < num_streams; ++i) {
        cudaFree(dA_pool[i]);
        cudaFree(dB_pool[i]);
        cudaFree(dC_pool[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}
}  // namespace

int main() {
    NVTX_RANGE("step:main");
    run_optimized();
    return 0;
}
