// PyTorch CUDA extension: experimental SM100 NVFP4 grouped GEMM dynamic blockscaled policy.
//
// This extension intentionally keeps the same input/output semantics and metadata shape as the
// primary CUTLASS extension. It swaps the schedule policy from explicit NVF4 ptr-array to the
// generic ptr-array blockscaled 2SM schedule tag.

#include <torch/extension.h>

#include <pybind11/pybind11.h>

#include <memory>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/packed_stride.hpp"

namespace py = pybind11;

namespace {

template <typename PlanT>
void bind_plan_type(py::module_& m,
                    std::unordered_map<std::type_index, py::object>& bound_types,
                    const char* py_name,
                    const char* doc) {
  auto key = std::type_index(typeid(PlanT));
  auto it = bound_types.find(key);
  if (it != bound_types.end()) {
    m.attr(py_name) = it->second;
    return;
  }

  py::object cls = py::class_<PlanT, std::shared_ptr<PlanT>>(m, py_name, py::module_local())
                       .def("run", &PlanT::run, doc);
  bound_types.emplace(key, cls);
}

int read_positive_env_or_default(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0') || parsed <= 0L ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

int resolve_sm_count_with_cap(int device_id) {
  int queried = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  int cap = read_positive_env_or_default("AISP_NVFP4_GROUP_GEMM_MAX_SM_COUNT", queried);
  return std::max(1, std::min(queried, cap));
}

dim3 resolve_cluster_fallback_shape(dim3 cluster_shape) {
  int fallback_m = read_positive_env_or_default(
      "AISP_NVFP4_GROUP_GEMM_CLUSTER_FALLBACK_M",
      static_cast<int>(cluster_shape.x));
  int fallback_n = read_positive_env_or_default(
      "AISP_NVFP4_GROUP_GEMM_CLUSTER_FALLBACK_N",
      static_cast<int>(cluster_shape.y));
  return dim3(static_cast<uint32_t>(fallback_m), static_cast<uint32_t>(fallback_n), 1);
}

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementC = cutlass::half_t;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::nv_float4_t<ElementInput>;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<ElementInput>;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = ElementC;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape = cute::Shape<int32_t, int32_t, cute::_1>;

template <typename KernelSchedule_, typename MmaTileShape_>
struct MMA2SMConfig {
  using MmaTileShape = MmaTileShape_;
  using KernelSchedule = KernelSchedule_;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

template <typename Config>
using CollectiveEpilogue2SMT = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    typename Config::MmaTileShape,
    ClusterShape,
    cute::Shape<cute::_128, cute::_64>,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutC*,
    AlignmentC,
    ElementD,
    LayoutD*,
    AlignmentD,
    typename Config::EpilogueSchedule>::CollectiveOp;

template <typename Config>
using CollectiveMainloop2SMT = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutA*,
    AlignmentA,
    ElementB,
    LayoutB*,
    AlignmentB,
    ElementAccumulator,
    typename Config::MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue2SMT<Config>::SharedStorage))>,
    typename Config::KernelSchedule>::CollectiveOp;

template <typename Config>
using GemmKernel2SMT = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop2SMT<Config>,
    CollectiveEpilogue2SMT<Config>>;

template <typename Config>
using Gemm2SMT = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SMT<Config>>;

using MMA2SMBlockScaledConfig = MMA2SMConfig<
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmBlockScaledSm100,
    cute::Shape<cute::_256, cute::_256, cute::_256>>;
using MMA2SMNvf4Config = MMA2SMConfig<
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
    cute::Shape<cute::_256, cute::_256, cute::_256>>;
using MMA2SMBlockScaledN128Config = MMA2SMConfig<
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmBlockScaledSm100,
    cute::Shape<cute::_256, cute::_128, cute::_256>>;
using MMA2SMNvf4N128Config = MMA2SMConfig<
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
    cute::Shape<cute::_256, cute::_128, cute::_256>>;

using Gemm2SMBlockScaled = Gemm2SMT<MMA2SMBlockScaledConfig>;
using Gemm2SMNvf4 = Gemm2SMT<MMA2SMNvf4Config>;
using Gemm2SMBlockScaledN128 = Gemm2SMT<MMA2SMBlockScaledN128Config>;
using Gemm2SMNvf4N128 = Gemm2SMT<MMA2SMNvf4N128Config>;

template <typename GemmT>
struct GemmTraits {
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
  using StrideA = typename GemmT::GemmKernel::InternalStrideA;
  using StrideB = typename GemmT::GemmKernel::InternalStrideB;
  using StrideC = typename GemmT::GemmKernel::InternalStrideC;
  using StrideD = typename GemmT::GemmKernel::InternalStrideD;
  using LayoutSFA = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename GemmT::GemmKernel::ElementSF;
  using ElementD = typename GemmT::EpilogueOutputOp::ElementOutput;
};

template <typename T>
torch::Tensor pack_to_cuda_u8_tensor(const std::vector<T>& host, int64_t count) {
  TORCH_CHECK(static_cast<int64_t>(host.size()) == count, "host vector size mismatch");
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  torch::Tensor out = torch::empty({count, static_cast<int64_t>(sizeof(T))}, opts);
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaMemcpyAsync(
      out.data_ptr(),
      host.data(),
      static_cast<size_t>(count) * sizeof(T),
      cudaMemcpyHostToDevice,
      stream.stream());
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed for metadata tensor");
  return out;
}

template <typename GemmT>
std::vector<torch::Tensor> build_metadata_impl(
    torch::Tensor problem_sizes_mnkl_cpu,
    int64_t cluster_m,
    int64_t cluster_n,
    int64_t raster_order,
    int64_t max_swizzle_size) {
  using Traits = GemmTraits<GemmT>;
  TORCH_CHECK(problem_sizes_mnkl_cpu.device().is_cpu(), "problem_sizes must be a CPU tensor");
  TORCH_CHECK(problem_sizes_mnkl_cpu.scalar_type() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(problem_sizes_mnkl_cpu.dim() == 2 && problem_sizes_mnkl_cpu.size(1) == 4,
              "problem_sizes must have shape (G, 4)");
  TORCH_CHECK(cluster_m > 0 && cluster_n > 0, "cluster_m/cluster_n must be > 0");

  const int64_t groups = problem_sizes_mnkl_cpu.size(0);
  const int device_id = c10::cuda::current_device();
  c10::cuda::CUDAGuard guard(device_id);
  auto acc = problem_sizes_mnkl_cpu.accessor<int32_t, 2>();

  std::vector<typename Traits::UnderlyingProblemShape> shapes_host;
  std::vector<typename Traits::StrideA> stride_a_host;
  std::vector<typename Traits::StrideB> stride_b_host;
  std::vector<typename Traits::StrideC> stride_c_host;
  std::vector<typename Traits::StrideD> stride_d_host;
  std::vector<typename Traits::LayoutSFA> layout_sfa_host;
  std::vector<typename Traits::LayoutSFB> layout_sfb_host;

  shapes_host.reserve(groups);
  stride_a_host.reserve(groups);
  stride_b_host.reserve(groups);
  stride_c_host.reserve(groups);
  stride_d_host.reserve(groups);
  layout_sfa_host.reserve(groups);
  layout_sfb_host.reserve(groups);

  for (int64_t i = 0; i < groups; ++i) {
    int32_t M = acc[i][0];
    int32_t N = acc[i][1];
    int32_t K = acc[i][2];
    int32_t L = acc[i][3];
    TORCH_CHECK(L == 1, "Only L=1 is supported (got L=", L, ")");

    shapes_host.push_back(cute::make_shape(M, N, K));
    stride_a_host.push_back(cutlass::make_cute_packed_stride(typename Traits::StrideA{}, {M, K, 1}));
    stride_b_host.push_back(cutlass::make_cute_packed_stride(typename Traits::StrideB{}, {N, K, 1}));
    stride_c_host.push_back(cutlass::make_cute_packed_stride(typename Traits::StrideC{}, {M, N, 1}));
    stride_d_host.push_back(cutlass::make_cute_packed_stride(typename Traits::StrideD{}, {M, N, 1}));

    layout_sfa_host.push_back(
        Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1)));
    layout_sfb_host.push_back(
        Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1)));
  }

  torch::Tensor problem_shapes_u8 = pack_to_cuda_u8_tensor(shapes_host, groups);
  torch::Tensor stride_a_u8 = pack_to_cuda_u8_tensor(stride_a_host, groups);
  torch::Tensor stride_b_u8 = pack_to_cuda_u8_tensor(stride_b_host, groups);
  torch::Tensor stride_c_u8 = pack_to_cuda_u8_tensor(stride_c_host, groups);
  torch::Tensor stride_d_u8 = pack_to_cuda_u8_tensor(stride_d_host, groups);
  torch::Tensor layout_sfa_u8 = pack_to_cuda_u8_tensor(layout_sfa_host, groups);
  torch::Tensor layout_sfb_u8 = pack_to_cuda_u8_tensor(layout_sfb_host, groups);

  auto opts_cuda_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor dummy_ptrs = torch::zeros({groups}, opts_cuda_i64);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = resolve_sm_count_with_cap(hw_info.device_id);
  hw_info.cluster_shape = dim3(static_cast<uint32_t>(cluster_m), static_cast<uint32_t>(cluster_n), 1);
  hw_info.cluster_shape_fallback = resolve_cluster_fallback_shape(hw_info.cluster_shape);

  typename GemmT::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = static_cast<decltype(scheduler.raster_order)>(raster_order);
  scheduler.max_swizzle_size = static_cast<decltype(scheduler.max_swizzle_size)>(max_swizzle_size);

  typename GemmT::Arguments args_ref;
  decltype(args_ref.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  auto ptr_problem = reinterpret_cast<typename Traits::UnderlyingProblemShape*>(problem_shapes_u8.data_ptr<uint8_t>());
  auto ptr_stride_a = reinterpret_cast<typename Traits::StrideA*>(stride_a_u8.data_ptr<uint8_t>());
  auto ptr_stride_b = reinterpret_cast<typename Traits::StrideB*>(stride_b_u8.data_ptr<uint8_t>());
  auto ptr_stride_c = reinterpret_cast<typename Traits::StrideC*>(stride_c_u8.data_ptr<uint8_t>());
  auto ptr_stride_d = reinterpret_cast<typename Traits::StrideD*>(stride_d_u8.data_ptr<uint8_t>());
  auto ptr_layout_sfa = reinterpret_cast<typename Traits::LayoutSFA*>(layout_sfa_u8.data_ptr<uint8_t>());
  auto ptr_layout_sfb = reinterpret_cast<typename Traits::LayoutSFB*>(layout_sfb_u8.data_ptr<uint8_t>());

  auto ptr_a = reinterpret_cast<typename GemmT::ElementA const**>(dummy_ptrs.data_ptr<int64_t>());
  auto ptr_b = reinterpret_cast<typename GemmT::ElementB const**>(dummy_ptrs.data_ptr<int64_t>());
  auto ptr_sfa = reinterpret_cast<typename Traits::ElementSF const**>(dummy_ptrs.data_ptr<int64_t>());
  auto ptr_sfb = reinterpret_cast<typename Traits::ElementSF const**>(dummy_ptrs.data_ptr<int64_t>());
  auto ptr_c = reinterpret_cast<typename GemmT::ElementC const**>(dummy_ptrs.data_ptr<int64_t>());
  auto ptr_d = reinterpret_cast<typename Traits::ElementD**>(dummy_ptrs.data_ptr<int64_t>());

  auto ptr_problem_host = reinterpret_cast<typename Traits::UnderlyingProblemShape*>(shapes_host.data());

  typename GemmT::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {static_cast<int32_t>(groups), ptr_problem, ptr_problem_host},
      {ptr_a, ptr_stride_a, ptr_b, ptr_stride_b, ptr_sfa, ptr_layout_sfa, ptr_sfb, ptr_layout_sfb},
      {fusion_args, ptr_c, ptr_stride_c, ptr_d, ptr_stride_d},
      hw_info,
      scheduler};

  size_t workspace_bytes = GemmT::get_workspace_size(args);
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_bytes)}, opts_u8);

  return {
      problem_shapes_u8,
      stride_a_u8,
      stride_b_u8,
      stride_c_u8,
      stride_d_u8,
      layout_sfa_u8,
      layout_sfb_u8,
      workspace,
  };
}

template <typename GemmT>
class GemmPlanT {
 public:
  GemmPlanT(
      torch::Tensor problem_shapes_u8,
      torch::Tensor stride_a_u8,
      torch::Tensor stride_b_u8,
      torch::Tensor stride_c_u8,
      torch::Tensor stride_d_u8,
      torch::Tensor layout_sfa_u8,
      torch::Tensor layout_sfb_u8,
      torch::Tensor workspace_u8,
      torch::Tensor ptr_a_i64,
      torch::Tensor ptr_b_i64,
      torch::Tensor ptr_sfa_i64,
      torch::Tensor ptr_sfb_i64,
      torch::Tensor ptr_c_i64,
      torch::Tensor ptr_d_i64,
      double alpha,
      double beta,
      int64_t raster_order,
      int64_t cluster_m,
      int64_t cluster_n,
      int64_t max_swizzle_size,
      bool use_pdl)
      : problem_shapes_u8_(std::move(problem_shapes_u8)),
        stride_a_u8_(std::move(stride_a_u8)),
        stride_b_u8_(std::move(stride_b_u8)),
        stride_c_u8_(std::move(stride_c_u8)),
        stride_d_u8_(std::move(stride_d_u8)),
        layout_sfa_u8_(std::move(layout_sfa_u8)),
        layout_sfb_u8_(std::move(layout_sfb_u8)),
        workspace_u8_(std::move(workspace_u8)),
        ptr_a_i64_(std::move(ptr_a_i64)),
        ptr_b_i64_(std::move(ptr_b_i64)),
        ptr_sfa_i64_(std::move(ptr_sfa_i64)),
        ptr_sfb_i64_(std::move(ptr_sfb_i64)),
        ptr_c_i64_(std::move(ptr_c_i64)),
        ptr_d_i64_(std::move(ptr_d_i64)),
        use_pdl_(use_pdl) {
    using Traits = GemmTraits<GemmT>;

    TORCH_CHECK(problem_shapes_u8_.is_cuda(), "problem_shapes must be CUDA");
    TORCH_CHECK(ptr_a_i64_.is_cuda(), "ptr_a must be CUDA");
    TORCH_CHECK(ptr_a_i64_.scalar_type() == torch::kInt64, "ptr tensors must be int64");
    TORCH_CHECK(ptr_b_i64_.scalar_type() == torch::kInt64, "ptr tensors must be int64");
    TORCH_CHECK(ptr_sfa_i64_.scalar_type() == torch::kInt64, "ptr tensors must be int64");
    TORCH_CHECK(ptr_sfb_i64_.scalar_type() == torch::kInt64, "ptr tensors must be int64");
    TORCH_CHECK(ptr_c_i64_.scalar_type() == torch::kInt64, "ptr tensors must be int64");
    TORCH_CHECK(ptr_d_i64_.scalar_type() == torch::kInt64, "ptr tensors must be int64");

    const int64_t groups = ptr_a_i64_.numel();
    TORCH_CHECK(ptr_b_i64_.numel() == groups, "ptr_b size mismatch");
    TORCH_CHECK(ptr_sfa_i64_.numel() == groups, "ptr_sfa size mismatch");
    TORCH_CHECK(ptr_sfb_i64_.numel() == groups, "ptr_sfb size mismatch");
    TORCH_CHECK(ptr_c_i64_.numel() == groups, "ptr_c size mismatch");
    TORCH_CHECK(ptr_d_i64_.numel() == groups, "ptr_d size mismatch");

    c10::cuda::CUDAGuard guard(ptr_a_i64_.get_device());

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = ptr_a_i64_.get_device();
    hw_info.sm_count = resolve_sm_count_with_cap(hw_info.device_id);
    hw_info.cluster_shape = dim3(static_cast<uint32_t>(cluster_m), static_cast<uint32_t>(cluster_n), 1);
    hw_info.cluster_shape_fallback = resolve_cluster_fallback_shape(hw_info.cluster_shape);

    typename GemmT::GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = static_cast<decltype(scheduler.raster_order)>(raster_order);
    scheduler.max_swizzle_size = static_cast<decltype(scheduler.max_swizzle_size)>(max_swizzle_size);

    typename GemmT::Arguments args_ref;
    decltype(args_ref.epilogue.thread) fusion_args;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha = static_cast<float>(alpha);
    fusion_args.beta = static_cast<float>(beta);
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    auto ptr_problem = reinterpret_cast<typename Traits::UnderlyingProblemShape*>(problem_shapes_u8_.data_ptr<uint8_t>());
    auto ptr_stride_a = reinterpret_cast<typename Traits::StrideA*>(stride_a_u8_.data_ptr<uint8_t>());
    auto ptr_stride_b = reinterpret_cast<typename Traits::StrideB*>(stride_b_u8_.data_ptr<uint8_t>());
    auto ptr_stride_c = reinterpret_cast<typename Traits::StrideC*>(stride_c_u8_.data_ptr<uint8_t>());
    auto ptr_stride_d = reinterpret_cast<typename Traits::StrideD*>(stride_d_u8_.data_ptr<uint8_t>());
    auto ptr_layout_sfa = reinterpret_cast<typename Traits::LayoutSFA*>(layout_sfa_u8_.data_ptr<uint8_t>());
    auto ptr_layout_sfb = reinterpret_cast<typename Traits::LayoutSFB*>(layout_sfb_u8_.data_ptr<uint8_t>());

    auto ptr_a = reinterpret_cast<typename GemmT::ElementA const**>(ptr_a_i64_.data_ptr<int64_t>());
    auto ptr_b = reinterpret_cast<typename GemmT::ElementB const**>(ptr_b_i64_.data_ptr<int64_t>());
    auto ptr_sfa = reinterpret_cast<typename Traits::ElementSF const**>(ptr_sfa_i64_.data_ptr<int64_t>());
    auto ptr_sfb = reinterpret_cast<typename Traits::ElementSF const**>(ptr_sfb_i64_.data_ptr<int64_t>());
    auto ptr_c = reinterpret_cast<typename GemmT::ElementC const**>(ptr_c_i64_.data_ptr<int64_t>());
    auto ptr_d = reinterpret_cast<typename Traits::ElementD**>(ptr_d_i64_.data_ptr<int64_t>());

    // Preserve host descriptors so CUTLASS grouped scheduling can use host-aware planning.
    problem_shapes_host_u8_ = problem_shapes_u8_.to(torch::kCPU);
    auto ptr_problem_host =
        reinterpret_cast<typename Traits::UnderlyingProblemShape*>(problem_shapes_host_u8_.data_ptr<uint8_t>());

    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {static_cast<int32_t>(groups), ptr_problem, ptr_problem_host},
        {ptr_a, ptr_stride_a, ptr_b, ptr_stride_b, ptr_sfa, ptr_layout_sfa, ptr_sfb, ptr_layout_sfb},
        {fusion_args, ptr_c, ptr_stride_c, ptr_d, ptr_stride_d},
        hw_info,
        scheduler};

    size_t required = GemmT::get_workspace_size(args);
    TORCH_CHECK(static_cast<size_t>(workspace_u8_.numel()) >= required,
                "workspace too small: have=", workspace_u8_.numel(), " need=", required);

    TORCH_CHECK(gemm_.can_implement(args) == cutlass::Status::kSuccess, "CUTLASS can_implement() failed");
    TORCH_CHECK(gemm_.initialize(args, workspace_u8_.data_ptr()) == cutlass::Status::kSuccess,
                "CUTLASS initialize() failed");
  }

  void run() {
    c10::cuda::CUDAGuard guard(ptr_a_i64_.get_device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cuda_stream = stream.stream();
    TORCH_CHECK(
        gemm_.run(cuda_stream, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ use_pdl_) ==
            cutlass::Status::kSuccess,
        "CUTLASS run() failed");
  }

 private:
  GemmT gemm_;
  torch::Tensor problem_shapes_u8_;
  torch::Tensor problem_shapes_host_u8_;
  torch::Tensor stride_a_u8_;
  torch::Tensor stride_b_u8_;
  torch::Tensor stride_c_u8_;
  torch::Tensor stride_d_u8_;
  torch::Tensor layout_sfa_u8_;
  torch::Tensor layout_sfb_u8_;
  torch::Tensor workspace_u8_;
  torch::Tensor ptr_a_i64_;
  torch::Tensor ptr_b_i64_;
  torch::Tensor ptr_sfa_i64_;
  torch::Tensor ptr_sfb_i64_;
  torch::Tensor ptr_c_i64_;
  torch::Tensor ptr_d_i64_;
  bool use_pdl_;
};

using GemmPlan2SMBlockScaled = GemmPlanT<Gemm2SMBlockScaled>;
using GemmPlan2SMDynS1A1 = GemmPlanT<Gemm2SMNvf4>;
using GemmPlan2SMDynS1A2 = GemmPlan2SMBlockScaled;
using GemmPlan2SMDynS2A1 = GemmPlanT<Gemm2SMNvf4N128>;
using GemmPlan2SMDynS2A2 = GemmPlanT<Gemm2SMBlockScaledN128>;

std::vector<torch::Tensor> build_metadata_2sm_dyn_s1a1(
    torch::Tensor problem_sizes_mnkl_cpu,
    int64_t cluster_m,
    int64_t cluster_n,
    int64_t raster_order,
    int64_t max_swizzle_size) {
  return build_metadata_impl<Gemm2SMNvf4>(
      std::move(problem_sizes_mnkl_cpu), cluster_m, cluster_n, raster_order, max_swizzle_size);
}

std::vector<torch::Tensor> build_metadata_2sm_dyn_s1a2(
    torch::Tensor problem_sizes_mnkl_cpu,
    int64_t cluster_m,
    int64_t cluster_n,
    int64_t raster_order,
    int64_t max_swizzle_size) {
  return build_metadata_impl<Gemm2SMBlockScaled>(
      std::move(problem_sizes_mnkl_cpu), cluster_m, cluster_n, raster_order, max_swizzle_size);
}

std::vector<torch::Tensor> build_metadata_2sm_dyn_s2a1(
    torch::Tensor problem_sizes_mnkl_cpu,
    int64_t cluster_m,
    int64_t cluster_n,
    int64_t raster_order,
    int64_t max_swizzle_size) {
  return build_metadata_impl<Gemm2SMNvf4N128>(
      std::move(problem_sizes_mnkl_cpu), cluster_m, cluster_n, raster_order, max_swizzle_size);
}

std::vector<torch::Tensor> build_metadata_2sm_dyn_s2a2(
    torch::Tensor problem_sizes_mnkl_cpu,
    int64_t cluster_m,
    int64_t cluster_n,
    int64_t raster_order,
    int64_t max_swizzle_size) {
  return build_metadata_impl<Gemm2SMBlockScaledN128>(
      std::move(problem_sizes_mnkl_cpu), cluster_m, cluster_n, raster_order, max_swizzle_size);
}

#else

class GemmPlan2SMBlockScaled {
 public:
  GemmPlan2SMBlockScaled(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, bool) {
    TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined (requires SM100 build)");
  }
  void run() {
    TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined (requires SM100 build)");
  }
};

using GemmPlan2SMDynS1A1 = GemmPlan2SMBlockScaled;
using GemmPlan2SMDynS1A2 = GemmPlan2SMBlockScaled;
using GemmPlan2SMDynS2A1 = GemmPlan2SMBlockScaled;
using GemmPlan2SMDynS2A2 = GemmPlan2SMBlockScaled;

std::vector<torch::Tensor> build_metadata_2sm_dyn_s1a1(torch::Tensor, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined (requires SM100 build)");
}
std::vector<torch::Tensor> build_metadata_2sm_dyn_s1a2(torch::Tensor, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined (requires SM100 build)");
}
std::vector<torch::Tensor> build_metadata_2sm_dyn_s2a1(torch::Tensor, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined (requires SM100 build)");
}
std::vector<torch::Tensor> build_metadata_2sm_dyn_s2a2(torch::Tensor, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined (requires SM100 build)");
}
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

template <typename PlanT>
void bind_create_plan(py::module_& m, const char* fn_name, const char* doc) {
  m.def(
      fn_name,
      [](torch::Tensor problem_shapes_u8,
         torch::Tensor stride_a_u8,
         torch::Tensor stride_b_u8,
         torch::Tensor stride_c_u8,
         torch::Tensor stride_d_u8,
         torch::Tensor layout_sfa_u8,
         torch::Tensor layout_sfb_u8,
         torch::Tensor workspace_u8,
         torch::Tensor ptr_a_i64,
         torch::Tensor ptr_b_i64,
         torch::Tensor ptr_sfa_i64,
         torch::Tensor ptr_sfb_i64,
         torch::Tensor ptr_c_i64,
         torch::Tensor ptr_d_i64,
         double alpha,
         double beta,
         int64_t raster_order,
         int64_t cluster_m,
         int64_t cluster_n,
         int64_t max_swizzle_size,
         bool use_pdl) {
        return std::make_shared<PlanT>(
            problem_shapes_u8,
            stride_a_u8,
            stride_b_u8,
            stride_c_u8,
            stride_d_u8,
            layout_sfa_u8,
            layout_sfb_u8,
            workspace_u8,
            ptr_a_i64,
            ptr_b_i64,
            ptr_sfa_i64,
            ptr_sfb_i64,
            ptr_c_i64,
            ptr_d_i64,
            alpha,
            beta,
            raster_order,
            cluster_m,
            cluster_n,
            max_swizzle_size,
            use_pdl);
      },
      doc,
      py::arg("problem_shapes_u8"),
      py::arg("stride_a_u8"),
      py::arg("stride_b_u8"),
      py::arg("stride_c_u8"),
      py::arg("stride_d_u8"),
      py::arg("layout_sfa_u8"),
      py::arg("layout_sfb_u8"),
      py::arg("workspace_u8"),
      py::arg("ptr_a_i64"),
      py::arg("ptr_b_i64"),
      py::arg("ptr_sfa_i64"),
      py::arg("ptr_sfb_i64"),
      py::arg("ptr_c_i64"),
      py::arg("ptr_d_i64"),
      py::arg("alpha") = 1.0,
      py::arg("beta") = 0.0,
      py::arg("raster_order") = 0,
      py::arg("cluster_m") = 2,
      py::arg("cluster_n") = 1,
      py::arg("max_swizzle_size") = 0,
      py::arg("use_pdl") = false);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  std::unordered_map<std::type_index, py::object> bound_types;

  bind_plan_type<GemmPlan2SMDynS1A1>(
      m,
      bound_types,
      "GemmPlan2SMDynS1A1",
      "Run pre-initialized grouped GEMM plan (2SM NVF4 policy, 256x256x256 tile)");
  bind_plan_type<GemmPlan2SMDynS1A2>(
      m,
      bound_types,
      "GemmPlan2SMDynS1A2",
      "Run pre-initialized grouped GEMM plan (2SM blockscaled generic policy, 256x256x256 tile)");
  bind_plan_type<GemmPlan2SMDynS2A1>(
      m,
      bound_types,
      "GemmPlan2SMDynS2A1",
      "Run pre-initialized grouped GEMM plan (2SM NVF4 policy, 256x128x256 tile)");
  bind_plan_type<GemmPlan2SMDynS2A2>(
      m,
      bound_types,
      "GemmPlan2SMDynS2A2",
      "Run pre-initialized grouped GEMM plan (2SM blockscaled generic policy, 256x128x256 tile)");

  m.def(
      "build_metadata_2sm_dyn_s1a1",
      &build_metadata_2sm_dyn_s1a1,
      "Build metadata for dynamic policy variant s1a1 (2SM NVF4, 256x256x256)",
      py::arg("problem_sizes_mnkl_cpu"),
      py::arg("cluster_m") = 2,
      py::arg("cluster_n") = 1,
      py::arg("raster_order") = 0,
      py::arg("max_swizzle_size") = 0);

  m.def(
      "build_metadata_2sm_dyn_s1a2",
      &build_metadata_2sm_dyn_s1a2,
      "Build metadata for dynamic policy variant s1a2 (2SM blockscaled generic, 256x256x256)",
      py::arg("problem_sizes_mnkl_cpu"),
      py::arg("cluster_m") = 2,
      py::arg("cluster_n") = 1,
      py::arg("raster_order") = 0,
      py::arg("max_swizzle_size") = 0);

  m.def(
      "build_metadata_2sm_dyn_s2a1",
      &build_metadata_2sm_dyn_s2a1,
      "Build metadata for dynamic policy variant s2a1 (2SM NVF4, 256x128x256)",
      py::arg("problem_sizes_mnkl_cpu"),
      py::arg("cluster_m") = 2,
      py::arg("cluster_n") = 1,
      py::arg("raster_order") = 0,
      py::arg("max_swizzle_size") = 0);

  m.def(
      "build_metadata_2sm_dyn_s2a2",
      &build_metadata_2sm_dyn_s2a2,
      "Build metadata for dynamic policy variant s2a2 (2SM blockscaled generic, 256x128x256)",
      py::arg("problem_sizes_mnkl_cpu"),
      py::arg("cluster_m") = 2,
      py::arg("cluster_n") = 1,
      py::arg("raster_order") = 0,
      py::arg("max_swizzle_size") = 0);

  bind_create_plan<GemmPlan2SMDynS1A1>(m, "create_plan_2sm_dyn_s1a1", "Create grouped plan for dynamic policy variant s1a1");
  bind_create_plan<GemmPlan2SMDynS1A2>(m, "create_plan_2sm_dyn_s1a2", "Create grouped plan for dynamic policy variant s1a2");
  bind_create_plan<GemmPlan2SMDynS2A1>(m, "create_plan_2sm_dyn_s2a1", "Create grouped plan for dynamic policy variant s2a1");
  bind_create_plan<GemmPlan2SMDynS2A2>(m, "create_plan_2sm_dyn_s2a2", "Create grouped plan for dynamic policy variant s2a2");
}
