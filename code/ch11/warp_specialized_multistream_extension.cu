#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "warp_specialized_two_pipelines_multistream_impl.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t _status = (call);                                                \
        if (_status != cudaSuccess) {                                                \
            throw std::runtime_error(cudaGetErrorString(_status));                  \
        }                                                                            \
    } while (0)

torch::Tensor warp_specialized_multistream_forward(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    int num_streams) {
    TORCH_CHECK(input_a.is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(input_b.is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(input_a.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(input_b.dtype() == torch::kFloat32, "Input B must be float32");
    TORCH_CHECK(input_a.sizes() == input_b.sizes(), "Inputs must have identical shapes");
    TORCH_CHECK(ch11::kHasCuda13Pipeline,
                "warp_specialized_multistream requires CUDA 13+ pipeline support.");

    auto a = input_a.contiguous();
    auto b = input_b.contiguous();
    auto c = torch::empty_like(a);

    const int64_t total_elements = a.numel();
    const int tiles_total = static_cast<int>(
        (total_elements + ch11::kTileElems - 1) / ch11::kTileElems);
    TORCH_CHECK(tiles_total > 0, "Inputs must contain at least one tile (256 elements).");

    num_streams = std::max(1, num_streams);
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        NVTX_RANGE("tile");
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    const int tiles_per_stream = (tiles_total + num_streams - 1) / num_streams;
    const dim3 block(6 * ch11::kWarpSize);  // loader + 4 compute + consumer warps
    const size_t shared_bytes =
        3 * ch11::kPipelineStages * ch11::kTileElems * sizeof(float);

    for (int stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
        NVTX_RANGE("tile");
        const int tile_start = stream_idx * tiles_per_stream;
        if (tile_start >= tiles_total) {
            break;
        }
        const int tile_count =
            std::min(tiles_per_stream, tiles_total - tile_start);
        const size_t element_offset =
            static_cast<size_t>(tile_start) * ch11::kTileElems;
        const int grid_dim = std::min(tile_count, 64);
        if (grid_dim <= 0) {
            continue;
        }

        const float* a_ptr = a.data_ptr<float>() + element_offset;
        const float* b_ptr = b.data_ptr<float>() + element_offset;
        float* c_ptr = c.data_ptr<float>() + element_offset;

        ch11::warp_specialized_kernel_two_pipelines_multistream<<<
            grid_dim, block, shared_bytes, streams[stream_idx]>>>(
            a_ptr,
            b_ptr,
            c_ptr,
            tile_count);
        CUDA_CHECK(cudaGetLastError());
    }

    for (auto stream : streams) {
        NVTX_RANGE("tile");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "warp_specialized_multistream_forward",
        &warp_specialized_multistream_forward,
        "Warp specialization kernel with multi-stream launches (Chapter 11).");
}
