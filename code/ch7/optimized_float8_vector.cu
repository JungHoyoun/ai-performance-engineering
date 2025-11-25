// optimized_float8_vector.cu - 32-byte Vectorized FP8 Loads (Ch7)
//
// WHAT: Blackwell supports 32-byte (256-bit) vectorized loads, allowing
// 32 FP8 values to be loaded in a single instruction.
//
// WHY: Wider loads = better memory bandwidth utilization:
//   - 4 bytes (float): 1 value per load
//   - 16 bytes (float4): 4 values per load  
//   - 32 bytes (float8 or 32×FP8): 8 floats or 32 FP8 values per load
//
// On Blackwell with HBM3e:
//   - Peak bandwidth: ~8 TB/s
//   - Need wide loads to saturate bandwidth
//   - 32-byte loads are 2x more efficient than 16-byte
//
// WHEN TO USE:
//   - FP8 inference (32 FP8 values per 32-byte load)
//   - Memory-bound kernels on Blackwell
//   - When data is 32-byte aligned
//
// REQUIREMENTS:
//   - SM 100+ (Blackwell) for full benefit
//   - 32-byte aligned data
//   - CUDA 12.0+ for FP8 types

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;

//============================================================================
// 32-byte aligned vector types
//============================================================================

// 32-byte vector of 8 floats
struct alignas(32) float8 {
    float data[8];
    
    __host__ __device__ float& operator[](int i) { return data[i]; }
    __host__ __device__ const float& operator[](int i) const { return data[i]; }
};

// 32-byte vector of 32 FP8 values (e4m3 format)
struct alignas(32) fp8x32 {
    __nv_fp8_e4m3 data[32];
    
    __host__ __device__ __nv_fp8_e4m3& operator[](int i) { return data[i]; }
    __host__ __device__ const __nv_fp8_e4m3& operator[](int i) const { return data[i]; }
};

// 16-byte vector for comparison
struct alignas(16) float4_aligned {
    float data[4];
};

//============================================================================
// Kernel: 32-byte Vectorized Float Loads
//============================================================================

__global__ void vector_add_float8(
    const float8* __restrict__ a,
    const float8* __restrict__ b,
    float8* __restrict__ c,
    int n  // number of float8 elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float8 va = a[idx];  // 32-byte load
        float8 vb = b[idx];  // 32-byte load
        
        float8 vc;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            vc[i] = va[i] + vb[i];
        }
        
        c[idx] = vc;  // 32-byte store
    }
}

// Comparison: 16-byte loads
__global__ void vector_add_float4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];
        float4 vb = b[idx];
        
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        
        c[idx] = vc;
    }
}

// Comparison: scalar loads
__global__ void vector_add_scalar(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

//============================================================================
// Kernel: FP8 32-element Vector Loads
//============================================================================

// Load 32 FP8 values, convert to FP32, add, convert back, store
__global__ void fp8_vector_add_32x(
    const fp8x32* __restrict__ a,
    const fp8x32* __restrict__ b,
    fp8x32* __restrict__ c,
    float scale_a,
    float scale_b,
    float scale_c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 32-byte loads (32 FP8 values each)
        fp8x32 va = a[idx];
        fp8x32 vb = b[idx];
        
        fp8x32 vc;
        
        // Process in groups of 4 for register efficiency
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            float4 fa, fb, fc;
            
            // Dequantize FP8 to FP32
            fa.x = float(va[i]) * scale_a;
            fa.y = float(va[i+1]) * scale_a;
            fa.z = float(va[i+2]) * scale_a;
            fa.w = float(va[i+3]) * scale_a;
            
            fb.x = float(vb[i]) * scale_b;
            fb.y = float(vb[i+1]) * scale_b;
            fb.z = float(vb[i+2]) * scale_b;
            fb.w = float(vb[i+3]) * scale_b;
            
            // Compute
            fc.x = fa.x + fb.x;
            fc.y = fa.y + fb.y;
            fc.z = fa.z + fb.z;
            fc.w = fa.w + fb.w;
            
            // Quantize FP32 to FP8
            vc[i] = __nv_fp8_e4m3(fc.x / scale_c);
            vc[i+1] = __nv_fp8_e4m3(fc.y / scale_c);
            vc[i+2] = __nv_fp8_e4m3(fc.z / scale_c);
            vc[i+3] = __nv_fp8_e4m3(fc.w / scale_c);
        }
        
        // 32-byte store
        c[idx] = vc;
    }
}

//============================================================================
// Benchmark
//============================================================================

void benchmark_vector_widths() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("32-byte Vectorized Loads Benchmark\n");
    printf("===================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("\n");
    
    // Large array to saturate bandwidth
    const int N = 128 * 1024 * 1024;  // 128M floats = 512 MB per array
    const size_t bytes = N * sizeof(float);
    
    printf("Array size: %zu MB per array\n", bytes / (1024 * 1024));
    printf("Total data movement: %zu MB (2 reads + 1 write)\n\n", 
           3 * bytes / (1024 * 1024));
    
    // Allocate with 32-byte alignment
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Initialize
    std::vector<float> h_data(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_a, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    
    // Benchmark scalar (4-byte)
    auto benchmark_kernel = [&](const char* name, auto kernel, int elements_per_thread) {
        int num_elements = N / elements_per_thread;
        dim3 block(BLOCK_SIZE);
        dim3 grid((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Clear L2
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            kernel<<<grid, block>>>(
                reinterpret_cast<std::remove_reference_t<decltype(*d_a)>*>(d_a),
                reinterpret_cast<std::remove_reference_t<decltype(*d_b)>*>(d_b),
                reinterpret_cast<std::remove_reference_t<decltype(*d_c)>*>(d_c),
                num_elements
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            kernel<<<grid, block>>>(
                reinterpret_cast<std::remove_reference_t<decltype(*d_a)>*>(d_a),
                reinterpret_cast<std::remove_reference_t<decltype(*d_b)>*>(d_b),
                reinterpret_cast<std::remove_reference_t<decltype(*d_c)>*>(d_c),
                num_elements
            );
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        // Bandwidth: 2 reads + 1 write
        float bandwidth = (3.0f * bytes) / (ms / 1000.0f) / 1e9f;
        
        printf("%-20s %8.3f ms  %8.1f GB/s\n", name, ms, bandwidth);
        return ms;
    };
    
    printf("%-20s %8s  %10s\n", "Variant", "Time", "Bandwidth");
    printf("%-20s %8s  %10s\n", "-------", "----", "---------");
    
    float scalar_ms = benchmark_kernel("Scalar (4B)", vector_add_scalar, 1);
    float vec4_ms = benchmark_kernel("float4 (16B)", 
        [](const float4* a, const float4* b, float4* c, int n) {
            vector_add_float4<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, n);
        }, 4);
    float vec8_ms = benchmark_kernel("float8 (32B)", 
        [](const float8* a, const float8* b, float8* c, int n) {
            vector_add_float8<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, n);
        }, 8);
    
    printf("\nSpeedup vs scalar:\n");
    printf("  float4 (16B): %.2fx\n", scalar_ms / vec4_ms);
    printf("  float8 (32B): %.2fx\n", scalar_ms / vec8_ms);
    
    printf("\nNotes:\n");
    printf("  - 32-byte loads are optimal on Blackwell (SM 100+)\n");
    printf("  - For FP8: 32-byte load = 32 FP8 values (vs 4 FP32 values)\n");
    printf("  - Requires 32-byte aligned memory allocation\n");
    printf("  - Benefits memory-bound kernels most\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    benchmark_vector_widths();
    return 0;
}


