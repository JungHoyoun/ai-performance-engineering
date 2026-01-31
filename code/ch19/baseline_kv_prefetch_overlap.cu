// Chapter 19: Baseline KV prefetch example without stream overlap.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "../core/common/nvtx_utils.cuh"

namespace {
constexpr size_t KV_BYTES = 2ull << 20; // 2 MiB chunk keeps run times manageable

__global__ void forward_kernel(float* kv, int elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems) return;
    float v = kv[idx];
    kv[idx] = v * 1.0001f + 0.1f;
}

void run_baseline(int iterations) {
    const int elems_per_chunk = KV_BYTES / sizeof(float);
    std::vector<float> host_in(iterations * elems_per_chunk);
    std::vector<float> host_out(iterations * elems_per_chunk);
    for (size_t i = 0; i < host_in.size(); ++i) {
        NVTX_RANGE("setup");
        host_in[i] = static_cast<float>((i % 17) * 0.5f);
    }

    float* device_slot = nullptr;
    cudaMalloc(&device_slot, KV_BYTES);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int it = 0; it < iterations; ++it) {
        NVTX_RANGE("transfer_sync:h2d");
        const float* src = host_in.data() + it * elems_per_chunk;
        float* dst = host_out.data() + it * elems_per_chunk;
        cudaMemcpy(device_slot, src, KV_BYTES, cudaMemcpyHostToDevice);
        forward_kernel<<<(elems_per_chunk + 255) / 256, 256>>>(device_slot, elems_per_chunk);
        cudaMemcpy(dst, device_slot, KV_BYTES, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("baseline_kv_prefetch_overlap: %d iters, %.3f ms\n", iterations, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_slot);
}
}  // namespace

int main() {
    NVTX_RANGE("main");
    run_baseline(16);
    return 0;
}
