#include <cstdio>
#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../core/common/cuda13_demo_runner.cuh"
#include "../core/common/nvtx_utils.cuh"
#endif

int main() {
    NVTX_RANGE("main");
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    std::printf("CUDA 13 preview demos (stream-ordered memory, TMA)\n");
    cuda13_demo_runner::run_stream_ordered_demo();
    cuda13_demo_runner::run_tma_demo();
#else
    std::printf("CUDA 13 preview demos require CUDA 13.0+ headers and runtime.\n");
#endif
    return 0;
}
