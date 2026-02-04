// Chapter 19: Optimized KV prefetch example overlapping memcpy + compute via dual streams.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace {
constexpr size_t KV_BYTES = 2ull << 20;

__global__ void forward_kernel(float* kv, int elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems) return;
    float v = kv[idx];
    kv[idx] = v * 1.0001f + 0.1f;
}

void run_optimized(int iterations) {
    const int elems_per_chunk = KV_BYTES / sizeof(float);

    float* host_in = nullptr;
    float* host_out = nullptr;
    cudaMallocHost(&host_in, iterations * elems_per_chunk * sizeof(float));
    cudaMallocHost(&host_out, iterations * elems_per_chunk * sizeof(float));
    for (size_t i = 0; i < static_cast<size_t>(iterations) * elems_per_chunk; ++i) {
        NVTX_RANGE("setup");
        host_in[i] = static_cast<float>((i % 17) * 0.5f);
    }

    constexpr int kStreams = 2;
    cudaStream_t compute_streams[kStreams];
    float* device_slots[kStreams];
    for (int i = 0; i < kStreams; ++i) {
        NVTX_RANGE("setup");
        cudaStreamCreateWithFlags(&compute_streams[i], cudaStreamNonBlocking);
        cudaMalloc(&device_slots[i], KV_BYTES);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 0; it < iterations; ++it) {
        NVTX_RANGE("transfer_async:h2d");
        const int stream_id = it % kStreams;
        cudaStream_t st = compute_streams[stream_id];
        float* device_slot = device_slots[stream_id];
        const float* src = host_in + static_cast<size_t>(it) * elems_per_chunk;
        float* dst = host_out + static_cast<size_t>(it) * elems_per_chunk;

        cudaMemcpyAsync(device_slot, src, KV_BYTES, cudaMemcpyHostToDevice, st);
        forward_kernel<<<(elems_per_chunk + 255) / 256, 256, 0, st>>>(device_slot, elems_per_chunk);
        cudaMemcpyAsync(dst, device_slot, KV_BYTES, cudaMemcpyDeviceToHost, st);
    }

    for (int i = 0; i < kStreams; ++i) {
        NVTX_RANGE("iteration");
        cudaStreamSynchronize(compute_streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("optimized_kv_prefetch_overlap: %d iters, %.3f ms\n", iterations, ms);

#ifdef VERIFY
    double checksum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(iterations) * elems_per_chunk; ++i) {
        checksum += std::abs(host_out[i]);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < kStreams; ++i) {
        NVTX_RANGE("cleanup");
        cudaStreamDestroy(compute_streams[i]);
        cudaFree(device_slots[i]);
    }
    cudaFreeHost(host_in);
    cudaFreeHost(host_out);
}
}  // namespace

int main() {
    NVTX_RANGE("main");
    run_optimized(16);
    return 0;
}
