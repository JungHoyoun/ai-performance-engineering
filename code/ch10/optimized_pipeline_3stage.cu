// optimized_pipeline_3stage.cu - 3-Stage Software Pipeline (Ch10)
//
// WHAT: A 3-stage pipeline with separate buffers for:
//   Stage 0: Loading tile N+2
//   Stage 1: Loading tile N+1 (in flight)
//   Stage 2: Computing tile N
//
// WHY: 3 stages provide better latency hiding than 2 stages:
//   - 2-stage: Can hide ~1 load latency
//   - 3-stage: Can hide ~2 load latencies (better for high-latency HBM)
//
// WHEN TO USE:
//   - Memory-bound kernels with significant load latency
//   - Large tiles that take multiple cycles to load
//   - When 2-stage doesn't fully hide latency (check nsys timeline)
//
// TRADE-OFF:
//   - More shared memory (3 buffers vs 2)
//   - More complex state machine
//   - Diminishing returns beyond 3 stages
//
// CUDA Pipeline API:
//   - cuda::pipeline<cuda::thread_scope_block>
//   - producer_acquire() / producer_commit()
//   - consumer_wait() / consumer_release()

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Configuration
constexpr int TILE_SIZE = 64;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_STAGES = 3;  // 3-stage pipeline

//============================================================================
// 3-Stage Pipelined GEMV Kernel
//============================================================================
// Vector-matrix multiply: y = A * x
// A is [M, K], x is [K], y is [M]
// Each block handles TILE_SIZE rows of output
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void gemv_3stage_pipeline(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ x,   // [K]
    float* __restrict__ y,         // [M]
    int M, int K
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * TILE_SIZE;
    
    // Triple buffer for x tiles (3 stages)
    __shared__ alignas(128) float x_smem[NUM_STAGES][TILE_SIZE];
    
    // Create 3-stage pipeline
    auto pipeline = cuda::make_pipeline();
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Per-thread accumulators for TILE_SIZE/BLOCK_SIZE rows
    constexpr int ROWS_PER_THREAD = (TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float acc[ROWS_PER_THREAD] = {0.0f};
    
    //========================================================================
    // PROLOGUE: Prime the pipeline with first 2 loads (stages 0, 1)
    //========================================================================
    #pragma unroll 1
    for (int stage = 0; stage < min(NUM_STAGES - 1, num_tiles); ++stage) {
        pipeline.producer_acquire();
        
        // Async copy x tile to shared memory
        if (tid < TILE_SIZE) {
            int k_idx = stage * TILE_SIZE + tid;
            if (k_idx < K) {
                cuda::memcpy_async(&x_smem[stage][tid], &x[k_idx], 
                                   sizeof(float), pipeline);
            } else {
                x_smem[stage][tid] = 0.0f;
            }
        }
        
        pipeline.producer_commit();
    }
    
    //========================================================================
    // MAIN LOOP: Overlap load of tile N+2 with compute of tile N
    //========================================================================
    for (int tile = 0; tile < num_tiles; ++tile) {
        int compute_stage = tile % NUM_STAGES;
        int load_stage = (tile + NUM_STAGES - 1) % NUM_STAGES;
        int load_tile = tile + NUM_STAGES - 1;
        
        // LOAD: Start loading tile N+2 (if it exists)
        if (load_tile < num_tiles) {
            pipeline.producer_acquire();
            
            if (tid < TILE_SIZE) {
                int k_idx = load_tile * TILE_SIZE + tid;
                if (k_idx < K) {
                    cuda::memcpy_async(&x_smem[load_stage][tid], &x[k_idx],
                                       sizeof(float), pipeline);
                } else {
                    x_smem[load_stage][tid] = 0.0f;
                }
            }
            
            pipeline.producer_commit();
        }
        
        // WAIT: Ensure tile N is ready
        pipeline.consumer_wait();
        __syncthreads();  // Ensure all threads see the loaded data
        
        // COMPUTE: Process tile N
        int k_base = tile * TILE_SIZE;
        
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            int row_offset = r * BLOCK_SIZE + tid;
            if (row_offset < TILE_SIZE) {
                int global_row = row_start + row_offset;
                if (global_row < M) {
                    float sum = 0.0f;
                    
                    // Dot product of row with x tile
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE && k_base + k < K; ++k) {
                        sum += A[global_row * K + k_base + k] * x_smem[compute_stage][k];
                    }
                    
                    acc[r] += sum;
                }
            }
        }
        
        __syncthreads();  // Ensure compute is done before releasing buffer
        
        // RELEASE: Allow buffer reuse
        pipeline.consumer_release();
    }
    
    //========================================================================
    // EPILOGUE: Write results
    //========================================================================
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int row_offset = r * BLOCK_SIZE + tid;
        if (row_offset < TILE_SIZE) {
            int global_row = row_start + row_offset;
            if (global_row < M) {
                y[global_row] = acc[r];
            }
        }
    }
}

//============================================================================
// 2-Stage Baseline for Comparison
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void gemv_2stage_pipeline(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * TILE_SIZE;
    
    // Double buffer (2 stages)
    __shared__ alignas(128) float x_smem[2][TILE_SIZE];
    
    auto pipeline = cuda::make_pipeline();
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    constexpr int ROWS_PER_THREAD = (TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float acc[ROWS_PER_THREAD] = {0.0f};
    
    // Prologue: Load first tile
    if (num_tiles > 0) {
        pipeline.producer_acquire();
        if (tid < TILE_SIZE) {
            int k_idx = tid;
            x_smem[0][tid] = (k_idx < K) ? x[k_idx] : 0.0f;
        }
        pipeline.producer_commit();
    }
    
    // Main loop
    for (int tile = 0; tile < num_tiles; ++tile) {
        int compute_stage = tile % 2;
        int load_stage = (tile + 1) % 2;
        
        // Load next tile
        if (tile + 1 < num_tiles) {
            pipeline.producer_acquire();
            if (tid < TILE_SIZE) {
                int k_idx = (tile + 1) * TILE_SIZE + tid;
                x_smem[load_stage][tid] = (k_idx < K) ? x[k_idx] : 0.0f;
            }
            pipeline.producer_commit();
        }
        
        pipeline.consumer_wait();
        __syncthreads();
        
        int k_base = tile * TILE_SIZE;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            int row_offset = r * BLOCK_SIZE + tid;
            if (row_offset < TILE_SIZE) {
                int global_row = row_start + row_offset;
                if (global_row < M) {
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE && k_base + k < K; ++k) {
                        sum += A[global_row * K + k_base + k] * x_smem[compute_stage][k];
                    }
                    acc[r] += sum;
                }
            }
        }
        
        __syncthreads();
        pipeline.consumer_release();
    }
    
    // Write results
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int row_offset = r * BLOCK_SIZE + tid;
        if (row_offset < TILE_SIZE) {
            int global_row = row_start + row_offset;
            if (global_row < M) {
                y[global_row] = acc[r];
            }
        }
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("3-Stage vs 2-Stage Pipeline Comparison\n");
    printf("Device: %s\n\n", prop.name);
    
    // Matrix dimensions (tall matrix for GEMV)
    const int M = 16384;
    const int K = 8192;
    
    printf("GEMV: [%d, %d] x [%d] = [%d]\n", M, K, K, M);
    printf("Tile: %d, Stages: 2 vs 3\n\n", TILE_SIZE);
    
    // Allocate
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_x = K * sizeof(float);
    size_t bytes_y = M * sizeof(float);
    
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_y));
    
    // Initialize
    std::vector<float> h_A(M * K), h_x(K);
    for (size_t i = 0; i < M * K; ++i) h_A[i] = 0.001f;
    for (int i = 0; i < K; ++i) h_x[i] = 0.001f;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes_x, cudaMemcpyHostToDevice));
    
    // Launch config
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 50;
    
    //========================================================================
    // Benchmark 2-Stage
    //========================================================================
    for (int i = 0; i < warmup; ++i) {
        gemv_2stage_pipeline<<<grid, block>>>(d_A, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemv_2stage_pipeline<<<grid, block>>>(d_A, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_2stage;
    CUDA_CHECK(cudaEventElapsedTime(&ms_2stage, start, stop));
    float avg_2stage = ms_2stage / iterations;
    
    //========================================================================
    // Benchmark 3-Stage
    //========================================================================
    for (int i = 0; i < warmup; ++i) {
        gemv_3stage_pipeline<<<grid, block>>>(d_A, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemv_3stage_pipeline<<<grid, block>>>(d_A, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_3stage;
    CUDA_CHECK(cudaEventElapsedTime(&ms_3stage, start, stop));
    float avg_3stage = ms_3stage / iterations;
    
    //========================================================================
    // Results
    //========================================================================
    double bytes_accessed = (double)M * K * sizeof(float) + K * sizeof(float) + M * sizeof(float);
    double bw_2stage = bytes_accessed / (avg_2stage / 1000.0) / 1e9;
    double bw_3stage = bytes_accessed / (avg_3stage / 1000.0) / 1e9;
    
    printf("Results:\n");
    printf("  2-Stage: %.3f ms (%.1f GB/s)\n", avg_2stage, bw_2stage);
    printf("  3-Stage: %.3f ms (%.1f GB/s)\n", avg_3stage, bw_3stage);
    printf("\n");
    
    float speedup = avg_2stage / avg_3stage;
    if (speedup > 1.0f) {
        printf("  3-Stage is %.2fx FASTER\n", speedup);
    } else {
        printf("  2-Stage is %.2fx faster (3-stage overhead may dominate for small tiles)\n", 1.0f / speedup);
    }
    
    printf("\nNote: 3-stage shines when load latency >> compute time.\n");
    printf("For small tiles or fast memory, 2-stage may be sufficient.\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return 0;
}


