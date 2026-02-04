// baseline_cublaslt_gemm_fp16.cu -- Naive FP16 GEMM baseline for cuBLASLt comparison
//
// This baseline uses FP16 data types to provide an apples-to-apples comparison
// with the optimized FP16 cuBLASLt version that uses tensor cores.
//
// BOOK REFERENCE (Ch9): Tensor cores provide massive throughput improvements
// for matrix operations with FP16/BF16 precision. This baseline shows the
// performance of a naive tiled FP16 GEMM without tensor core acceleration.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Naive tiled FP16 GEMM kernel (no tensor cores)
// Uses shared memory tiling but standard FP16 arithmetic
template<int TILE_SIZE = 32>
__global__ void tiled_fp16_gemm_kernel(const __half* __restrict__ A,
                                        const __half* __restrict__ B,
                                        __half* __restrict__ C,
                                        int M,
                                        int N,
                                        int K,
                                        __half alpha,
                                        __half beta) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;  // Accumulate in FP32 for numerical stability
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        // Load tile of B into shared memory
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        float alpha_f = __half2float(alpha);
        float beta_f = __half2float(beta);
        float c_val = __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(alpha_f * sum + beta_f * c_val);
    }
}

int main() {
    NVTX_RANGE("main");
    // Match optimized version's matrix sizes
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 64;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A = elements_A * sizeof(__half) * kBatchCount;
    const size_t size_B = elements_B * sizeof(__half) * kBatchCount;
    const size_t size_C = elements_C * sizeof(__half) * kBatchCount;

    // Host allocation with pinned memory
    __half* h_A = nullptr;
    __half* h_B = nullptr;
    __half* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    // Initialize with random FP16 values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < elements_A * kBatchCount; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = __float2half(dis(gen));
    }
    for (size_t i = 0; i < elements_B * kBatchCount; ++i) {
        NVTX_RANGE("setup");
        h_B[i] = __float2half(dis(gen));
    }
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        NVTX_RANGE("setup");
        h_C[i] = __float2half(0.0f);
    }

    // Device allocation
    __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // Launch configuration
    dim3 block(32, 32);  // 32x32 tile
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Warmup
    tiled_fp16_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, __float2half(1.0f), __float2half(0.0f));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("iteration");
        for (int batch = 0; batch < kBatchCount; ++batch) {
            NVTX_RANGE("compute_kernel:fp16_naive");
            const size_t offset = static_cast<size_t>(batch) * elements_A;
            tiled_fp16_gemm_kernel<<<grid, block>>>(
                d_A + offset,
                d_B + offset,
                d_C + offset,
                M, N, K,
                __float2half(1.0f),
                __float2half(0.0f));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / (kIterations * kBatchCount);
    
    std::cout << "Naive FP16 GEMM (baseline): " << avg_ms << " ms" << std::endl;

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    const size_t elements = elements_C * kBatchCount;
    double checksum = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        checksum += std::abs(__half2float(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}
