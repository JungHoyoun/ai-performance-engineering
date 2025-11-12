#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

namespace ch8 {

constexpr int kDoubleBufferBlock = 256;
constexpr int kDoubleBufferTile = 4;
constexpr int kDoubleBufferInnerLoops = 16;
constexpr int kValuesPerThread = 4;
constexpr int kPipelineStages = 3;

namespace cg = cooperative_groups;

__device__ __forceinline__ float pipeline_transform(float value) {
    return value * 1.0002f + value * value * 0.00001f;
}

__global__ void double_buffer_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int elements) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int lane_offset = tid * kValuesPerThread;
    const int tile_span = blockDim.x * kValuesPerThread;
    const int block_span = tile_span * kDoubleBufferTile;
    const int tile_base = blockIdx.x * block_span;

    if (tile_base >= elements) {
        return;
    }

    const int remaining = max(elements - tile_base, 0);
    const int max_tiles = min(
        kDoubleBufferTile,
        (remaining + tile_span - 1) / tile_span);
    if (max_tiles <= 0) {
        return;
    }

    for (int tile = 0; tile < max_tiles; ++tile) {
        const int base_idx = tile_base + tile * tile_span + lane_offset;
#pragma unroll
        for (int v = 0; v < kValuesPerThread; ++v) {
            const int idx = base_idx + v;
            smem[lane_offset + v] = (idx < elements) ? input[idx] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int v = 0; v < kValuesPerThread; ++v) {
            float value = smem[lane_offset + v];
#pragma unroll
            for (int loop = 0; loop < kDoubleBufferInnerLoops; ++loop) {
                value = pipeline_transform(value);
            }
            const int idx = base_idx + v;
            if (idx < elements) {
                output[idx] = value;
            }
        }
        __syncthreads();
    }
}

__global__ void double_buffer_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int elements) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int lane_offset = tid * kValuesPerThread;
    const int tile_span = blockDim.x * kValuesPerThread;
    const int block_span = tile_span * kDoubleBufferTile;
    const int tile_base = blockIdx.x * block_span;
    if (tile_base >= elements) {
        return;
    }

    const int remaining = max(elements - tile_base, 0);
    const int max_tiles = min(
        kDoubleBufferTile,
        (remaining + tile_span - 1) / tile_span);
    if (max_tiles <= 0) {
        return;
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, kPipelineStages> pipeline_state;
    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipeline_state);

    for (int tile = 0; tile < max_tiles + kPipelineStages; ++tile) {
        if (tile < max_tiles) {
            const int stage = tile % kPipelineStages;
            float* stage_ptr = smem + stage * tile_span;
            const int load_idx = tile_base + tile * tile_span + lane_offset;
            pipe.producer_acquire();
            if (load_idx + kValuesPerThread <= elements) {
                cuda::memcpy_async(
                    block,
                    stage_ptr + lane_offset,
                    input + load_idx,
                    sizeof(float) * kValuesPerThread,
                    pipe);
            } else if (load_idx < elements) {
#pragma unroll
                for (int v = 0; v < kValuesPerThread; ++v) {
                    const int idx = load_idx + v;
                    stage_ptr[lane_offset + v] =
                        (idx < elements) ? input[idx] : 0.0f;
                }
            } else {
#pragma unroll
                for (int v = 0; v < kValuesPerThread; ++v) {
                    stage_ptr[lane_offset + v] = 0.0f;
                }
            }
            pipe.producer_commit();
        }

        if (tile >= kPipelineStages - 1) {
            const int consume_tile = tile - (kPipelineStages - 1);
            if (consume_tile < max_tiles) {
                const int consume_stage = consume_tile % kPipelineStages;
                float* stage_ptr = smem + consume_stage * tile_span;
                pipe.consumer_wait();
                float values[kValuesPerThread];
#pragma unroll
                for (int v = 0; v < kValuesPerThread; ++v) {
                    values[v] = stage_ptr[lane_offset + v];
                }
#pragma unroll
                for (int loop = 0; loop < kDoubleBufferInnerLoops; ++loop) {
#pragma unroll
                    for (int v = 0; v < kValuesPerThread; ++v) {
                        values[v] = pipeline_transform(values[v]);
                    }
                }
                const int store_idx = tile_base + consume_tile * tile_span + lane_offset;
#pragma unroll
                for (int v = 0; v < kValuesPerThread; ++v) {
                    const int out_idx = store_idx + v;
                    if (out_idx < elements) {
                        output[out_idx] = values[v];
                    }
                }
                pipe.consumer_release();
            }
        }
    }
}

inline dim3 double_buffer_grid(int elements) {
    const int block_span = kDoubleBufferBlock * kDoubleBufferTile * kValuesPerThread;
    return dim3((elements + block_span - 1) / block_span);
}

inline void launch_double_buffer_baseline(
    const float* input,
    float* output,
    int elements,
    cudaStream_t stream) {
    const size_t shared_bytes = kDoubleBufferBlock * kValuesPerThread * sizeof(float);
    double_buffer_baseline_kernel<<<double_buffer_grid(elements), kDoubleBufferBlock, shared_bytes, stream>>>(
        input,
        output,
        elements);
}

inline void launch_double_buffer_optimized(
    const float* input,
    float* output,
    int elements,
    cudaStream_t stream) {
    const size_t shared_bytes = kDoubleBufferBlock * kValuesPerThread * kPipelineStages * sizeof(float);
    double_buffer_optimized_kernel<<<double_buffer_grid(elements), kDoubleBufferBlock, shared_bytes, stream>>>(
        input,
        output,
        elements);
}

}  // namespace ch8
