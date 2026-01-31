// PyTorch extension wrapper for warp_specialized_pipeline_enhanced.cu
// Based on Chapter 10's enhanced warp specialization pattern

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            throw std::runtime_error(cudaGetErrorString(status)); \
        } \
    } while(0)

constexpr int TILE = 64;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int PIPELINE_DEPTH = 2;

// Enhanced compute with more work to showcase pipeline benefits
__device__ void compute_tile_enhanced(const float* a, const float* b, float* c, int lane) {
    for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        float x = a[idx];
        float y = b[idx];
        
        // More compute to show pipeline overlap benefit
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(x * x + y * y) * 0.125f;
        }
        c[idx] = result;
    }
}

// Enhanced warp-specialized kernel with:
// - Double-buffer pipeline (2 stages)
// - More compute warps (6 vs 1)
// - Producer/consumer overlap without global sync
__global__ void warp_specialized_enhanced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int total_tiles
) {
    cg::thread_block block = cg::this_thread_block();
    
    // Double buffering for the pipeline
    extern __shared__ float smem[];
    float* A_tiles = smem;
    float* B_tiles = smem + PIPELINE_DEPTH * TILE_ELEMS;
    float* C_tiles = smem + 2 * PIPELINE_DEPTH * TILE_ELEMS;

    using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_DEPTH>;
    __shared__ alignas(pipe_state) unsigned char state_storage[sizeof(pipe_state)];
    auto* state = reinterpret_cast<pipe_state*>(state_storage);
    auto pipe = cuda::make_pipeline(block, state);

    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    
    // 8 warps total: 1 producer, 6 compute, 1 consumer
    constexpr int PRODUCER_WARP = 0;
    constexpr int CONSUMER_WARP = 7;
    constexpr int COMPUTE_WARPS = 6;
    
    // Warp roles
    bool is_producer = (warp_id == PRODUCER_WARP);
    bool is_consumer = (warp_id == CONSUMER_WARP);
    bool is_compute = (warp_id >= 1 && warp_id <= COMPUTE_WARPS);
    
    int global_warp_id = blockIdx.x;
    
    // Producer: Load tiles with pipelining
    if (is_producer) {
        for (int tile = global_warp_id; tile < total_tiles; tile += gridDim.x) {
            int buf_idx = tile % PIPELINE_DEPTH;
            size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
            
            pipe.producer_acquire();
            
            // Vectorized loads
            for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
                float4 a4 = *reinterpret_cast<const float4*>(&A[offset + idx * 4]);
                float4 b4 = *reinterpret_cast<const float4*>(&B[offset + idx * 4]);
                *reinterpret_cast<float4*>(&A_tiles[buf_idx * TILE_ELEMS + idx * 4]) = a4;
                *reinterpret_cast<float4*>(&B_tiles[buf_idx * TILE_ELEMS + idx * 4]) = b4;
            }
            
            pipe.producer_commit();
        }
    }
    
    // Compute: Process tiles with pipelining
    if (is_compute) {
        for (int tile = global_warp_id; tile < total_tiles; tile += gridDim.x) {
            int buf_idx = tile % PIPELINE_DEPTH;
            
            pipe.consumer_wait();
            
            compute_tile_enhanced(
                &A_tiles[buf_idx * TILE_ELEMS],
                &B_tiles[buf_idx * TILE_ELEMS],
                &C_tiles[buf_idx * TILE_ELEMS],
                lane
            );
            
            pipe.consumer_release();
        }
    }
    
    // Consumer: Store results with pipelining
    if (is_consumer) {
        for (int tile = global_warp_id; tile < total_tiles; tile += gridDim.x) {
            int buf_idx = tile % PIPELINE_DEPTH;
            size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
            
            pipe.consumer_wait();
            
            // Vectorized stores
            for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
                float4 c4 = *reinterpret_cast<float4*>(&C_tiles[buf_idx * TILE_ELEMS + idx * 4]);
                *reinterpret_cast<float4*>(&C[offset + idx * 4]) = c4;
            }
            
            pipe.consumer_release();
        }
    }
}

torch::Tensor warp_specialized_pipeline_enhanced_forward(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.sizes() == B.sizes(), "Inputs must have same shape");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Inputs must be float32");
    
    int n_elements = A.numel();
    int total_tiles = (n_elements + TILE_ELEMS - 1) / TILE_ELEMS;
    
    auto C = torch::empty_like(A);
    
    // Launch configuration: 8 warps per block (1 producer + 6 compute + 1 consumer)
    dim3 block(8 * 32);  // 8 warps * 32 threads
    dim3 grid(std::min(total_tiles, 256));
    size_t shared_bytes = (3 * PIPELINE_DEPTH * TILE_ELEMS) * sizeof(float);
    
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_dynamic_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_dynamic_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));
    TORCH_CHECK(shared_bytes <= static_cast<size_t>(max_dynamic_smem),
                "Requested shared memory (", shared_bytes,
                " bytes) exceeds device limit (", max_dynamic_smem, " bytes)");
    CUDA_CHECK(cudaFuncSetAttribute(
        warp_specialized_enhanced_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    
    warp_specialized_enhanced_kernel<<<grid, block, shared_bytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        total_tiles
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_specialized_pipeline_enhanced_forward", &warp_specialized_pipeline_enhanced_forward,
          "Enhanced warp-specialized pipeline forward (Chapter 10)");
}
