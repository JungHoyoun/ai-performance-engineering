#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

namespace {

constexpr int TILE_SIZE = 128;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int CLUSTER_BLOCKS = 4;      // Thread blocks per cluster (x dimension)
constexpr int DEFAULT_BATCHES = 4;
constexpr int DEFAULT_STREAMS = 2;
constexpr int DEFAULT_TILES_PER_BATCH = 2;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _status = (call);                                            \
        if (_status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(_status));       \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

__device__ void compute_rows_from_ds(const float* __restrict__ A_src,
                                     const float* __restrict__ B_src,
                                     float* __restrict__ C_dst,
                                     int row_begin,
                                     int row_end,
                                     int lane_id) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
        for (int col = 0; col < TILE_SIZE; ++col) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                acc += A_src[row * TILE_SIZE + k] * B_src[k * TILE_SIZE + col];
            }
            C_dst[row * TILE_SIZE + col] = acc;
        }
    }
}

extern "C" __global__ void warp_specialized_cluster_pipeline(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles) {
    cg::thread_block cta = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();

    extern __shared__ float smem[];
    float* A_tile_local = smem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;

    using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, 1>;
    __shared__ alignas(pipe_state) unsigned char pipe_storage[sizeof(pipe_state)];
    auto* state = reinterpret_cast<pipe_state*>(pipe_storage);
    auto pipe = cuda::make_pipeline(cta, state);

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int cluster_rank = cluster.block_rank();
    const dim3 cluster_dims = cluster.dim_blocks();
    const int blocks_in_cluster = cluster_dims.x * cluster_dims.y * cluster_dims.z;

    for (int tile = blockIdx.x / cluster_dims.x; tile < numTiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        if (cluster_rank == 0 && warp_id == 0) {
            pipe.producer_acquire();
            cuda::memcpy_async(cta,
                               A_tile_local,
                               A_global + offset,
                               TILE_ELEMS * sizeof(float),
                               pipe);
            cuda::memcpy_async(cta,
                               B_tile_local,
                               B_global + offset,
                               TILE_ELEMS * sizeof(float),
                               pipe);
            pipe.producer_commit();
            pipe.consumer_wait();
            pipe.consumer_release();
        }

        cluster.sync();

        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);

        const int rows_per_block =
            (TILE_SIZE + blocks_in_cluster - 1) / blocks_in_cluster;
        const int row_begin = min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end = min(row_begin + rows_per_block, TILE_SIZE);

        if (warp_id == 1) {
            compute_rows_from_ds(A_src, B_src, C_tile_local, row_begin, row_end, lane_id);
        }

        cta.sync();

        if (warp_id == 2) {
            for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
        }

        cluster.sync();
    }
}

void launch_warp_specialized_cluster_pipeline_multistream(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int tiles_per_batch,
    int numBatches,
    int numStreams) {
    if (tiles_per_batch <= 0 || numBatches <= 0 || numStreams <= 0) {
        std::fprintf(stderr, "Invalid arguments to launch function.\n");
        return;
    }

    const int batchLength = tiles_per_batch * TILE_ELEMS;
    const int numTiles = tiles_per_batch;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int cluster_launch = 0;
#ifdef cudaDevAttrClusterLaunch
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
#endif
    const bool supports_clusters =
        cluster_launch > 0 || prop.major >= 9;
    if (!supports_clusters) {
        std::fprintf(stderr,
                     "Thread block clusters are not supported on this GPU. "
                     "Detected SM %d.%d\n",
                     prop.major,
                     prop.minor);
        return;
    }

    CUDA_CHECK(cudaFuncSetAttribute(
        warp_specialized_cluster_pipeline,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));

    std::vector<cudaStream_t> streams(numStreams);
    for (auto& stream : streams) {
        NVTX_RANGE("tile");
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    const int blocksPerGrid = std::max(CLUSTER_BLOCKS, prop.multiProcessorCount * CLUSTER_BLOCKS);
    const dim3 blockDim(96);
    const size_t shmemBytes = 3ull * TILE_ELEMS * sizeof(float);

    cudaLaunchAttribute attr[1]{};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER_BLOCKS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    for (int b = 0; b < numBatches; ++b) {
        NVTX_RANGE("transfer");
        cudaStream_t st = streams[b % numStreams];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        const size_t bytes = static_cast<size_t>(batchLength) * sizeof(float);

        CUDA_CHECK(cudaMallocAsync(&dA, bytes, st));
        CUDA_CHECK(cudaMallocAsync(&dB, bytes, st));
        CUDA_CHECK(cudaMallocAsync(&dC, bytes, st));

        CUDA_CHECK(cudaMemcpyAsync(dA,
                                   h_A + static_cast<size_t>(b) * batchLength,
                                   bytes,
                                   cudaMemcpyHostToDevice,
                                   st));
        CUDA_CHECK(cudaMemcpyAsync(dB,
                                   h_B + static_cast<size_t>(b) * batchLength,
                                   bytes,
                                   cudaMemcpyHostToDevice,
                                   st));

        void* args[] = { &dA, &dB, &dC, (void*)&numTiles };
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(blocksPerGrid);
        cfg.blockDim = blockDim;
        cfg.dynamicSmemBytes = shmemBytes;
        cfg.stream = st;
        cfg.attrs = attr;
        cfg.numAttrs = 1;

        cudaKernel_t kernel;
        CUDA_CHECK(cudaGetKernel(&kernel, warp_specialized_cluster_pipeline));
        void* func = reinterpret_cast<void*>(kernel);
        CUDA_CHECK(cudaLaunchKernelExC(&cfg, func, args));

        CUDA_CHECK(cudaMemcpyAsync(h_C + static_cast<size_t>(b) * batchLength,
                                   dC,
                                   bytes,
                                   cudaMemcpyDeviceToHost,
                                   st));
        CUDA_CHECK(cudaFreeAsync(dA, st));
        CUDA_CHECK(cudaFreeAsync(dB, st));
        CUDA_CHECK(cudaFreeAsync(dC, st));
    }

    for (auto& stream : streams) {
        NVTX_RANGE("batch");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

struct Options {
    int tiles_per_batch = DEFAULT_TILES_PER_BATCH;
    int batches = DEFAULT_BATCHES;
    int streams = DEFAULT_STREAMS;
    bool verify = true;
};

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        NVTX_RANGE("verify");
        const std::string arg(argv[i]);
        auto parse_int = [&](int& dst) {
            if (i + 1 < argc) {
                dst = std::max(1, std::atoi(argv[++i]));
            }
        };
        if (arg == "--tiles-per-batch") {
            parse_int(opts.tiles_per_batch);
        } else if (arg == "--batches") {
            parse_int(opts.batches);
        } else if (arg == "--streams") {
            parse_int(opts.streams);
        } else if (arg == "--skip-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::printf(
                "Usage: %s [options]\n"
                "  --tiles-per-batch <int>   Number of tiles per batch (default %d)\n"
                "  --batches <int>           Number of batches (default %d)\n"
                "  --streams <int>           CUDA streams to cycle (default %d)\n"
                "  --skip-verify             Skip host-side verification\n",
                argv[0],
                DEFAULT_TILES_PER_BATCH,
                DEFAULT_BATCHES,
                DEFAULT_STREAMS);
            std::exit(EXIT_SUCCESS);
        }
    }
    return opts;
}

}  // namespace

int main(int argc, char** argv) {
    NVTX_RANGE("main");
    Options opts = parse_options(argc, argv);

    const int batchLength = opts.tiles_per_batch * TILE_ELEMS;
    const size_t total_elems = static_cast<size_t>(batchLength) * opts.batches;

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    CUDA_CHECK(cudaMallocHost(&hA, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&hB, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&hC, total_elems * sizeof(float)));

    for (size_t i = 0; i < total_elems; ++i) {
        NVTX_RANGE("tile");
        hA[i] = static_cast<float>((i % TILE_SIZE) * 0.5f);
        hB[i] = static_cast<float>((i % TILE_SIZE) * 0.25f + 1.0f);
        hC[i] = 0.0f;
    }

    launch_warp_specialized_cluster_pipeline_multistream(
        hA, hB, hC, opts.tiles_per_batch, opts.batches, opts.streams);

    if (opts.verify) {
        double max_err = 0.0;
        std::vector<float> ref(batchLength);
        for (int b = 0; b < opts.batches; ++b) {
            NVTX_RANGE("batch");
            const float* A_batch = hA + static_cast<size_t>(b) * batchLength;
            const float* B_batch = hB + static_cast<size_t>(b) * batchLength;
            float* C_batch = ref.data();
            for (int tile = 0; tile < opts.tiles_per_batch; ++tile) {
                NVTX_RANGE("tile");
                const float* A_tile = A_batch + static_cast<size_t>(tile) * TILE_ELEMS;
                const float* B_tile = B_batch + static_cast<size_t>(tile) * TILE_ELEMS;
                float* C_tile = C_batch + static_cast<size_t>(tile) * TILE_ELEMS;
                for (int row = 0; row < TILE_SIZE; ++row) {
                    NVTX_RANGE("tile");
                    for (int col = 0; col < TILE_SIZE; ++col) {
                        NVTX_RANGE("tile");
                        float acc = 0.0f;
                        for (int k = 0; k < TILE_SIZE; ++k) {
                            NVTX_RANGE("tile");
                            acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
                        }
                        C_tile[row * TILE_SIZE + col] = acc;
                    }
                }
            }
            const float* gpu = hC + static_cast<size_t>(b) * batchLength;
            for (int i = 0; i < batchLength; ++i) {
                NVTX_RANGE("cleanup");
                max_err = std::max(max_err, static_cast<double>(std::abs(ref[i] - gpu[i])));
            }
        }
        std::printf("Verification complete (max error %.3e)\n", max_err);
    }

    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
    std::printf("warp_specialized_cluster_pipeline_multistream complete.\n");
    return 0;
}

#else  // CUDA < 13

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::printf("warp_specialized_cluster_pipeline_multistream requires CUDA 13+ cluster APIs. "
                    "Detected SM %d.%d; skipping.\\n", prop.major, prop.minor);
    } else {
        std::printf("warp_specialized_cluster_pipeline_multistream requires CUDA 13+ cluster APIs.\\n");
    }
    return 0;
}

#endif  // CUDA version guard
