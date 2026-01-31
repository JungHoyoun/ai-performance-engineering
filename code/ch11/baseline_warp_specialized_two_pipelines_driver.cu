#include <cuda_runtime.h>

#include <cstdio>
#include <cstddef>

#include "baseline_warp_specialized_two_pipelines_common.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace {

constexpr int kNumStreams = 2;
constexpr int kBatches = 8;
constexpr size_t kBytesPerTile =
    static_cast<size_t>(ch11::kBaselineTileElems) * sizeof(float);

void check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main() {
    NVTX_RANGE("main");
  cudaStream_t streams[kNumStreams];
  for (int i = 0; i < kNumStreams; ++i) {
      NVTX_RANGE("setup");
    check(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }

  float *hA = nullptr, *hB = nullptr, *hC = nullptr;
  check(cudaMallocHost(&hA, kBatches * kBytesPerTile));
  check(cudaMallocHost(&hB, kBatches * kBytesPerTile));
  check(cudaMallocHost(&hC, kBatches * kBytesPerTile));

  for (int b = 0; b < kBatches; ++b) {
      NVTX_RANGE("setup");
    for (int i = 0; i < ch11::kBaselineTileElems; ++i) {
        NVTX_RANGE("setup");
      hA[static_cast<size_t>(b) * ch11::kBaselineTileElems + i] = static_cast<float>(i);
      hB[static_cast<size_t>(b) * ch11::kBaselineTileElems + i] = 1.0f;
    }
  }

  for (int b = 0; b < kBatches; ++b) {
      NVTX_RANGE("iteration");
    cudaStream_t st = streams[b % kNumStreams];
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    check(cudaMallocAsync(&dA, kBytesPerTile, st));
    check(cudaMallocAsync(&dB, kBytesPerTile, st));
    check(cudaMallocAsync(&dC, kBytesPerTile, st));

    const float* srcA = hA + static_cast<size_t>(b) * ch11::kBaselineTileElems;
    const float* srcB = hB + static_cast<size_t>(b) * ch11::kBaselineTileElems;
    float* dstC = hC + static_cast<size_t>(b) * ch11::kBaselineTileElems;

    check(cudaMemcpyAsync(dA, srcA, kBytesPerTile, cudaMemcpyHostToDevice, st));
    check(cudaMemcpyAsync(dB, srcB, kBytesPerTile, cudaMemcpyHostToDevice, st));

    ch11::launch_baseline_warp_specialized_two_pipelines(dA, dB, dC, /*num_tiles=*/1, st);
    check(cudaGetLastError());

    check(cudaMemcpyAsync(dstC, dC, kBytesPerTile, cudaMemcpyDeviceToHost, st));
    check(cudaFreeAsync(dA, st));
    check(cudaFreeAsync(dB, st));
    check(cudaFreeAsync(dC, st));
  }

  for (int i = 0; i < kNumStreams; ++i) {
      NVTX_RANGE("cleanup");
    check(cudaStreamSynchronize(streams[i]));
    check(cudaStreamDestroy(streams[i]));
  }

  check(cudaFreeHost(hA));
  check(cudaFreeHost(hB));
  check(cudaFreeHost(hC));
  return 0;
}

