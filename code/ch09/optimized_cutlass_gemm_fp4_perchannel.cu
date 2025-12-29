// optimized_cutlass_gemm_fp4_perchannel.cu -- CUTLASS NVFP4 GEMM with per-channel output scaling
//
// Optimized uses a larger CUTLASS tile/cluster configuration to exploit SM100
// block-scaled tensor core throughput.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cute/tensor.hpp"

#include "helper.h"
#include "../core/common/headers/cuda_verify.cuh"

using namespace cute;

constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kIterations = 10;
constexpr int kSwizzle = 2;

struct Options {
    int m;
    int n;
    int k;
    int iterations;
    float alpha;
    float beta;
    int swizzle;

    double gflops(double runtime_s) const {
        uint64_t flop = uint64_t(2) * m * n * k;
        return (double(flop) / 1.0e9) / runtime_s;
    }
};

__global__ void apply_per_channel_scale(cutlass::bfloat16_t* __restrict__ output,
                                        const float* __restrict__ scales,
                                        int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    const int idx = row * cols + col;
    float val = static_cast<float>(output[idx]) * scales[col];
    output[idx] = cutlass::bfloat16_t(val);
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = Shape<_256, _256, _256>;
using ClusterShape = Shape<_2, _4, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));

StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
uint64_t seed = 42;

cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

template <typename Element, typename Layout>
bool initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed_value) {
    double scope_max = 2.0;
    double scope_min = -2.0;
    constexpr int bits_input = cutlass::sizeof_bits<Element>::value;
    if constexpr (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
    } else if constexpr (bits_input <= 6) {
        scope_max = 2;
        scope_min = -2;
    } else if constexpr (bits_input <= 8) {
        if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
            scope_max = 4;
            scope_min = 1;
        } else {
            scope_max = 1;
            scope_min = -1;
        }
    } else {
        scope_max = 4;
        scope_min = -4;
    }
    cutlass::reference::host::TensorFillRandomUniform(view, seed_value, scope_max, scope_min, 0);
    return true;
}

void initialize(const Options& options) {
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

    layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
    layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
    layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
    layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
    layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, 1));
    layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, 1));

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    initialize_block(block_A.host_view(), seed + 2021);
    initialize_block(block_B.host_view(), seed + 2022);
    initialize_block(block_C.host_view(), seed + 2023);
    initialize_block(block_SFA.host_view(), seed + 2024);
    initialize_block(block_SFB.host_view(), seed + 2025);

    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();
}

typename Gemm::Arguments args_from_options(const Options& options) {
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {
            block_A.device_data(), stride_A,
            block_B.device_data(), stride_B,
            block_SFA.device_data(), layout_SFA,
            block_SFB.device_data(), layout_SFB
        },
        {
            {options.alpha, options.beta},
            block_C.device_data(), stride_C,
            block_D.device_data(), stride_D
        }
    };
    arguments.scheduler.max_swizzle_size = options.swizzle;
    return arguments;
}

int run_cutlass(const Options& options) {
    initialize(options);

    Gemm gemm;
    auto arguments = args_from_options(options);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
        CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    const float elapsed_ms = timer.elapsed_millis();
    const float avg_ms = elapsed_ms / static_cast<float>(options.iterations);
    std::cout << "CUTLASS NVFP4 GEMM (optimized, per-channel): " << avg_ms << " ms" << std::endl;

    dim3 block(16, 16);
    dim3 grid((options.n + block.x - 1) / block.x, (options.m + block.y - 1) / block.y);

    std::vector<float> h_scales(options.n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> scale_dis(0.75f, 1.25f);
    for (auto& v : h_scales) {
        v = scale_dis(gen);
    }
    float* d_scales = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scales, options.n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), options.n * sizeof(float), cudaMemcpyHostToDevice));

    apply_per_channel_scale<<<grid, block>>>(block_D.device_data(), d_scales, options.m, options.n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_scales));

#ifdef VERIFY
    block_D.sync_host();
    const size_t elements = static_cast<size_t>(options.m) * options.n;
    double checksum = 0.0;
    const ElementD* h_out = block_D.host_data();
    for (size_t i = 0; i < elements; ++i) {
        checksum += std::abs(static_cast<float>(h_out[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    return 0;
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main() {
    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
        std::cerr << "SKIPPED: CUTLASS NVFP4 requires CUDA 12.8+." << std::endl;
        return 3;
    }

    cudaDeviceProp props{};
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    if (props.major < 10) {
        std::cerr << "SKIPPED: CUTLASS NVFP4 requires SM100+." << std::endl;
        return 3;
    }

    Options options{};
    options.m = kM;
    options.n = kN;
    options.k = kK;
    options.iterations = kIterations;
    options.alpha = 1.0f;
    options.beta = 0.0f;
    options.swizzle = kSwizzle;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return run_cutlass(options);
#else
    std::cerr << "SKIPPED: CUTLASS SM100 blockscaled support not compiled." << std::endl;
    return 3;
#endif
}
