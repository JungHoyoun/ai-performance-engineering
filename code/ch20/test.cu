// Minimal CUDA 13 sample for NVIDIA Blackwell (SM10x/SM12x)
// Build: nvcc -std=c++20 -O3 test.cu -o test \
//   -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100 \
//   -gencode arch=compute_103,code=sm_103 -gencode arch=compute_103,code=compute_103 \
//   -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 \
//   -gencode arch=compute_121,code=sm_121 -gencode arch=compute_121,code=compute_121
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(cmd)                                                     \
    do {                                                                    \
        cudaError_t err = cmd;                                              \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\\n",               \
                         cudaGetErrorString(err), __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    NVTX_RANGE("main");
    constexpr int N = 1 << 20;
    constexpr float A = 2.0f;

    float *x_dev = nullptr, *y_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&x_dev, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_dev, N * sizeof(float)));

    float* x_host = new float[N];
    float* y_host = new float[N];
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("setup");
        x_host[i] = 1.0f;
        y_host[i] = 2.0f;
    }

    CUDA_CHECK(cudaMemcpy(x_dev, x_host, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_dev, y_host, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    saxpy<<<grid, block>>>(N, A, x_dev, y_dev);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y_host, y_dev, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("cleanup");
        max_err = fmaxf(max_err, fabsf(y_host[i] - 4.0f));
    }
    std::printf("max error = %g\n", max_err);

    delete[] x_host;
    delete[] y_host;
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(y_dev));

    return (max_err < 1e-6f) ? 0 : 1;
}
