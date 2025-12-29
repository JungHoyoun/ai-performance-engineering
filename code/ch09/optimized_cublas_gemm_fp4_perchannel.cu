// optimized_cublas_gemm_fp4_perchannel.cu -- NVFP4 GEMM with per-channel output scaling (optimized cuBLASLt)
//
// Optimized uses cuBLASLt NVFP4 with a larger workspace budget to allow
// higher-performance algorithms on Blackwell tensor cores.

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLASLT_CHECK(call)                                                     \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " "        \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

constexpr int FP4_BLOCK_SIZE = 16;

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int kIterations = 10;
constexpr int kBatchCount = 1;
constexpr size_t kWorkspaceBytes = 64ull * 1024ull * 1024ull;  // 64MB optimized workspace

__global__ void apply_per_channel_scale(__half* __restrict__ output,
                                        const float* __restrict__ scales,
                                        int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    if (row >= rows || col >= cols) {
        return;
    }

    const int idx = batch * rows * cols + row * cols + col;
    const float scale = scales[col];
    const float val = __half2float(output[idx]) * scale;
    output[idx] = __float2half(val);
}

void quantize_to_nvfp4(const float* input, uint8_t* output_packed,
                       __nv_fp8_e4m3* scales,
                       int rows, int cols) {
    const int packed_cols = cols / 2;
    const int num_scale_cols = cols / FP4_BLOCK_SIZE;

    for (int r = 0; r < rows; ++r) {
        for (int block = 0; block < num_scale_cols; ++block) {
            const int block_start = block * FP4_BLOCK_SIZE;

            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                max_abs = std::max(max_abs, std::abs(input[r * cols + block_start + i]));
            }

            float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
            scales[r * num_scale_cols + block] = __nv_fp8_e4m3(scale);

            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                float v0 = input[r * cols + block_start + i];
                float v1 = input[r * cols + block_start + i + 1];

                __nv_fp4_storage_t fp4_0 = __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
                __nv_fp4_storage_t fp4_1 = __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);

                int packed_idx = r * packed_cols + (block_start + i) / 2;
                output_packed[packed_idx] = ((fp4_1 & 0x0F) << 4) | (fp4_0 & 0x0F);
            }
        }
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 10) {
        std::cerr << "SKIPPED: NVFP4 requires SM100+ (found SM" << prop.major << "." << prop.minor << ")\n";
        return 3;
    }

    static_assert(M % FP4_BLOCK_SIZE == 0, "M must be multiple of 16");
    static_assert(N % FP4_BLOCK_SIZE == 0, "N must be multiple of 16");
    static_assert(K % FP4_BLOCK_SIZE == 0, "K must be multiple of 16");
    static_assert(N % 2 == 0 && K % 2 == 0, "N and K must be even for FP4 packing");

    const size_t packed_K = K / 2;
    const size_t packed_N = N / 2;
    const size_t elements_A_packed = static_cast<size_t>(M) * packed_K;
    const size_t elements_B_packed = static_cast<size_t>(K) * packed_N;
    const size_t elements_C = static_cast<size_t>(M) * N;

    const size_t num_scales_per_row_A = K / FP4_BLOCK_SIZE;
    const size_t num_scales_per_row_B = N / FP4_BLOCK_SIZE;
    const size_t num_scales_A = M * num_scales_per_row_A;
    const size_t num_scales_B = K * num_scales_per_row_B;

    std::vector<float> h_A_fp32(M * K * kBatchCount);
    std::vector<float> h_B_fp32(K * N * kBatchCount);
    std::vector<uint8_t> h_A_packed(elements_A_packed * kBatchCount);
    std::vector<uint8_t> h_B_packed(elements_B_packed * kBatchCount);
    std::vector<__nv_fp8_e4m3> h_A_scales(num_scales_A * kBatchCount);
    std::vector<__nv_fp8_e4m3> h_B_scales(num_scales_B * kBatchCount);
    std::vector<__half> h_C(elements_C * kBatchCount);
    std::vector<float> h_channel_scales(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dis(0.75f, 1.25f);
    for (auto& v : h_A_fp32) v = dis(gen);
    for (auto& v : h_B_fp32) v = dis(gen);
    for (auto& v : h_channel_scales) v = scale_dis(gen);
    std::fill(h_C.begin(), h_C.end(), __float2half(0.0f));

    for (int batch = 0; batch < kBatchCount; ++batch) {
        quantize_to_nvfp4(h_A_fp32.data() + batch * M * K,
                          h_A_packed.data() + batch * elements_A_packed,
                          h_A_scales.data() + batch * num_scales_A,
                          M, K);
        quantize_to_nvfp4(h_B_fp32.data() + batch * K * N,
                          h_B_packed.data() + batch * elements_B_packed,
                          h_B_scales.data() + batch * num_scales_B,
                          K, N);
    }

    uint8_t *d_A = nullptr, *d_B = nullptr;
    __nv_fp8_e4m3 *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    float *d_channel_scales = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A_packed * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B_packed * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_A_scales, num_scales_A * kBatchCount * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, num_scales_B * kBatchCount * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * kBatchCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_channel_scales, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), elements_A_packed * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), elements_B_packed * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales.data(), num_scales_A * kBatchCount * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales.data(), num_scales_B * kBatchCount * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), elements_C * kBatchCount * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_channel_scales, h_channel_scales.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasLtHandle_t lt_handle;
    CUBLASLT_CHECK(cublasLtCreate(&lt_handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasLtMatmulDesc_t matmul_desc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

    void* d_A_scales_ptr = d_A_scales;
    void* d_B_scales_ptr = d_B_scales;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scales_ptr, sizeof(d_A_scales_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scales_ptr, sizeof(d_B_scales_ptr)));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, M, K, M));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, N, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, M, N, M));

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes, sizeof(kWorkspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt_handle, matmul_desc, layoutA, layoutB, layoutC, layoutC,
        preference, 1, &heuristic, &returned));
    if (returned == 0) {
        std::cerr << "No suitable cuBLASLt algorithm found for NVFP4 GEMM." << std::endl;
        return 1;
    }

    void* d_workspace = nullptr;
    if (kWorkspaceBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, kWorkspaceBytes));
    }

    for (int batch = 0; batch < kBatchCount; ++batch) {
        const size_t offset_A = batch * elements_A_packed;
        const size_t offset_B = batch * elements_B_packed;
        const size_t offset_C = batch * elements_C;
        CUBLASLT_CHECK(cublasLtMatmul(
            lt_handle, matmul_desc,
            &alpha,
            d_A + offset_A, layoutA,
            d_B + offset_B, layoutB,
            &beta,
            d_C + offset_C, layoutC,
            d_C + offset_C, layoutC,
            &heuristic.algo,
            d_workspace, kWorkspaceBytes,
            stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A_packed;
            const size_t offset_B = batch * elements_B_packed;
            const size_t offset_C = batch * elements_C;
            CUBLASLT_CHECK(cublasLtMatmul(
                lt_handle, matmul_desc,
                &alpha,
                d_A + offset_A, layoutA,
                d_B + offset_B, layoutB,
                &beta,
                d_C + offset_C, layoutC,
                d_C + offset_C, layoutC,
                &heuristic.algo,
                d_workspace, kWorkspaceBytes,
                stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);
    const double flops = 2.0 * static_cast<double>(M) * N * K * kBatchCount * kIterations;
    const double tflops = flops / (total_ms * 1e9);

    std::cout << "cuBLASLt NVFP4 GEMM (optimized, per-channel): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, kBatchCount);
    apply_per_channel_scale<<<grid, block, 0, stream>>>(d_C, d_channel_scales, M, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, elements_C * kBatchCount * sizeof(__half), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        checksum += std::abs(__half2float(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(matmul_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLASLT_CHECK(cublasLtDestroy(lt_handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_channel_scales));

    return 0;
}
