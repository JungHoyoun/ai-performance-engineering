#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_api.h>

#include <cstdint>

namespace {

__device__ __forceinline__ int8_t fp4_e2m1_to_intq(uint8_t nibble) {
    // Signed symmetric lookup in quantized integer space for E2M1 values.
    constexpr int8_t kLut[16] = {
        0, 1, 2, 3,
        4, 6, 8, 12,
        0, -1, -2, -3,
        -4, -6, -8, -12,
    };
    return kLut[nibble & 0x0F];
}

__host__ __device__ __forceinline__ size_t shared_bytes_for_tile(
    int block_m,
    int block_n,
    int kpack_tile,
    int scale_tile
) {
    return
        static_cast<size_t>(block_m) * static_cast<size_t>(kpack_tile) +
        static_cast<size_t>(block_n) * static_cast<size_t>(kpack_tile) +
        static_cast<size_t>(block_m + block_n) * static_cast<size_t>(scale_tile) * sizeof(half) +
        16;
}

__global__ void nvfp4_group_gemm_kernel_grouped_pure_custom(
    const uint64_t* __restrict__ a_ptrs,
    const uint64_t* __restrict__ b_ptrs,
    const uint64_t* __restrict__ sfa_ptrs,
    const uint64_t* __restrict__ sfb_ptrs,
    const uint64_t* __restrict__ c_ptrs,
    const int32_t* __restrict__ m_sizes,
    const int32_t* __restrict__ n_sizes,
    const int32_t* __restrict__ k_halves,
    const int32_t* __restrict__ k_scales,
    int block_m,
    int block_n,
    int kpack_tile,
    int scale_tile
) {
    const int group_idx = static_cast<int>(blockIdx.z);
    const uint8_t* const a_packed = reinterpret_cast<const uint8_t*>(a_ptrs[group_idx]);
    const uint8_t* const b_packed = reinterpret_cast<const uint8_t*>(b_ptrs[group_idx]);
    const half* const sfa_half = reinterpret_cast<const half*>(sfa_ptrs[group_idx]);
    const half* const sfb_half = reinterpret_cast<const half*>(sfb_ptrs[group_idx]);
    half* const c_out = reinterpret_cast<half*>(c_ptrs[group_idx]);

    const int m_size = m_sizes[group_idx];
    const int n_size = n_sizes[group_idx];
    const int k_half = k_halves[group_idx];
    const int k_scale = k_scales[group_idx];

    const int local_n = static_cast<int>(threadIdx.x);
    const int local_m = static_cast<int>(threadIdx.y);
    const int global_m = static_cast<int>(blockIdx.y) * block_m + local_m;
    const int global_n = static_cast<int>(blockIdx.x) * block_n + local_n;

    const int linear_tid = local_m * block_n + local_n;
    const int threads = block_m * block_n;

    if (static_cast<int>(blockIdx.y) * block_m >= m_size || static_cast<int>(blockIdx.x) * block_n >= n_size) {
        return;
    }

    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_ptr = smem_raw;

    uint8_t* const sh_a = reinterpret_cast<uint8_t*>(smem_ptr);
    smem_ptr += static_cast<size_t>(block_m) * static_cast<size_t>(kpack_tile);

    uint8_t* const sh_b = reinterpret_cast<uint8_t*>(smem_ptr);
    smem_ptr += static_cast<size_t>(block_n) * static_cast<size_t>(kpack_tile);

    uintptr_t aligned_ptr = reinterpret_cast<uintptr_t>(smem_ptr);
    aligned_ptr = (aligned_ptr + alignof(half) - 1U) & ~(static_cast<uintptr_t>(alignof(half) - 1U));

    half* const sh_sa = reinterpret_cast<half*>(aligned_ptr);
    aligned_ptr += static_cast<size_t>(block_m) * static_cast<size_t>(scale_tile) * sizeof(half);

    half* const sh_sb = reinterpret_cast<half*>(aligned_ptr);

    float acc = 0.0f;

    for (int kp_base = 0; kp_base < k_half; kp_base += kpack_tile) {
        const int valid_kpack = ((k_half - kp_base) < kpack_tile) ? (k_half - kp_base) : kpack_tile;
        const int valid_scale = (valid_kpack + 7) >> 3;

        for (int idx = linear_tid; idx < block_m * kpack_tile; idx += threads) {
            const int row = idx / kpack_tile;
            const int kk = idx - row * kpack_tile;
            const int gm = static_cast<int>(blockIdx.y) * block_m + row;
            const int gk = kp_base + kk;
            uint8_t value = 0;
            if (gm < m_size && kk < valid_kpack) {
                value = a_packed[static_cast<size_t>(gm) * static_cast<size_t>(k_half) + static_cast<size_t>(gk)];
            }
            sh_a[idx] = value;
        }

        for (int idx = linear_tid; idx < block_n * kpack_tile; idx += threads) {
            const int row = idx / kpack_tile;
            const int kk = idx - row * kpack_tile;
            const int gn = static_cast<int>(blockIdx.x) * block_n + row;
            const int gk = kp_base + kk;
            uint8_t value = 0;
            if (gn < n_size && kk < valid_kpack) {
                value = b_packed[static_cast<size_t>(gn) * static_cast<size_t>(k_half) + static_cast<size_t>(gk)];
            }
            sh_b[idx] = value;
        }

        for (int idx = linear_tid; idx < block_m * scale_tile; idx += threads) {
            const int row = idx / scale_tile;
            const int s = idx - row * scale_tile;
            const int gm = static_cast<int>(blockIdx.y) * block_m + row;
            const int gs = (kp_base >> 3) + s;
            half value = __float2half_rn(0.0f);
            if (gm < m_size && s < valid_scale && gs < k_scale) {
                value = sfa_half[static_cast<size_t>(gm) * static_cast<size_t>(k_scale) + static_cast<size_t>(gs)];
            }
            sh_sa[idx] = value;
        }

        for (int idx = linear_tid; idx < block_n * scale_tile; idx += threads) {
            const int row = idx / scale_tile;
            const int s = idx - row * scale_tile;
            const int gn = static_cast<int>(blockIdx.x) * block_n + row;
            const int gs = (kp_base >> 3) + s;
            half value = __float2half_rn(0.0f);
            if (gn < n_size && s < valid_scale && gs < k_scale) {
                value = sfb_half[static_cast<size_t>(gn) * static_cast<size_t>(k_scale) + static_cast<size_t>(gs)];
            }
            sh_sb[idx] = value;
        }

        __syncthreads();

        if (global_m < m_size && global_n < n_size) {
            const uint8_t* const a_row = sh_a + static_cast<size_t>(local_m) * static_cast<size_t>(kpack_tile);
            const uint8_t* const b_row = sh_b + static_cast<size_t>(local_n) * static_cast<size_t>(kpack_tile);
            const half* const sa_row = sh_sa + static_cast<size_t>(local_m) * static_cast<size_t>(scale_tile);
            const half* const sb_row = sh_sb + static_cast<size_t>(local_n) * static_cast<size_t>(scale_tile);

#pragma unroll
            for (int s = 0; s < 16; ++s) {
                if (s >= valid_scale) {
                    break;
                }
                const float scale = __half2float(sa_row[s]) * __half2float(sb_row[s]);
                const int kk_base = s << 3;
                const int remaining = valid_kpack - kk_base;
                const int active = remaining > 8 ? 8 : remaining;

#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t >= active) {
                        break;
                    }
                    const uint8_t pa = a_row[kk_base + t];
                    const uint8_t pb = b_row[kk_base + t];

                    const int a0 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>(pa & 0x0F)));
                    const int a1 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>((pa >> 4) & 0x0F)));
                    const int b0 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>(pb & 0x0F)));
                    const int b1 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>((pb >> 4) & 0x0F)));

                    acc += scale * (0.25f * static_cast<float>((a0 * b0) + (a1 * b1)));
                }
            }
        }

        __syncthreads();
    }

    if (global_m < m_size && global_n < n_size) {
        c_out[static_cast<size_t>(global_m) * static_cast<size_t>(n_size) + static_cast<size_t>(global_n)] =
            __float2half_rn(acc);
    }
}

}  // namespace

void nvfp4_group_gemm_forward_grouped_cuda(
    torch::Tensor a_ptrs,
    torch::Tensor b_ptrs,
    torch::Tensor sfa_ptrs,
    torch::Tensor sfb_ptrs,
    torch::Tensor c_ptrs,
    torch::Tensor m_sizes,
    torch::Tensor n_sizes,
    torch::Tensor k_halves,
    torch::Tensor k_scales,
    int max_m_size,
    int max_n_size,
    int block_m,
    int block_n,
    int kpack_tile
) {
    TORCH_CHECK(a_ptrs.is_cuda(), "a_ptrs must be CUDA tensor");
    TORCH_CHECK(b_ptrs.is_cuda(), "b_ptrs must be CUDA tensor");
    TORCH_CHECK(sfa_ptrs.is_cuda(), "sfa_ptrs must be CUDA tensor");
    TORCH_CHECK(sfb_ptrs.is_cuda(), "sfb_ptrs must be CUDA tensor");
    TORCH_CHECK(c_ptrs.is_cuda(), "c_ptrs must be CUDA tensor");
    TORCH_CHECK(m_sizes.is_cuda(), "m_sizes must be CUDA tensor");
    TORCH_CHECK(n_sizes.is_cuda(), "n_sizes must be CUDA tensor");
    TORCH_CHECK(k_halves.is_cuda(), "k_halves must be CUDA tensor");
    TORCH_CHECK(k_scales.is_cuda(), "k_scales must be CUDA tensor");

    TORCH_CHECK(a_ptrs.scalar_type() == torch::kInt64, "a_ptrs must be torch.int64");
    TORCH_CHECK(b_ptrs.scalar_type() == torch::kInt64, "b_ptrs must be torch.int64");
    TORCH_CHECK(sfa_ptrs.scalar_type() == torch::kInt64, "sfa_ptrs must be torch.int64");
    TORCH_CHECK(sfb_ptrs.scalar_type() == torch::kInt64, "sfb_ptrs must be torch.int64");
    TORCH_CHECK(c_ptrs.scalar_type() == torch::kInt64, "c_ptrs must be torch.int64");
    TORCH_CHECK(m_sizes.scalar_type() == torch::kInt, "m_sizes must be torch.int32");
    TORCH_CHECK(n_sizes.scalar_type() == torch::kInt, "n_sizes must be torch.int32");
    TORCH_CHECK(k_halves.scalar_type() == torch::kInt, "k_halves must be torch.int32");
    TORCH_CHECK(k_scales.scalar_type() == torch::kInt, "k_scales must be torch.int32");

    TORCH_CHECK(a_ptrs.dim() == 1, "a_ptrs must be 1D");
    TORCH_CHECK(b_ptrs.dim() == 1, "b_ptrs must be 1D");
    TORCH_CHECK(sfa_ptrs.dim() == 1, "sfa_ptrs must be 1D");
    TORCH_CHECK(sfb_ptrs.dim() == 1, "sfb_ptrs must be 1D");
    TORCH_CHECK(c_ptrs.dim() == 1, "c_ptrs must be 1D");
    TORCH_CHECK(m_sizes.dim() == 1, "m_sizes must be 1D");
    TORCH_CHECK(n_sizes.dim() == 1, "n_sizes must be 1D");
    TORCH_CHECK(k_halves.dim() == 1, "k_halves must be 1D");
    TORCH_CHECK(k_scales.dim() == 1, "k_scales must be 1D");

    TORCH_CHECK(a_ptrs.is_contiguous(), "a_ptrs must be contiguous");
    TORCH_CHECK(b_ptrs.is_contiguous(), "b_ptrs must be contiguous");
    TORCH_CHECK(sfa_ptrs.is_contiguous(), "sfa_ptrs must be contiguous");
    TORCH_CHECK(sfb_ptrs.is_contiguous(), "sfb_ptrs must be contiguous");
    TORCH_CHECK(c_ptrs.is_contiguous(), "c_ptrs must be contiguous");
    TORCH_CHECK(m_sizes.is_contiguous(), "m_sizes must be contiguous");
    TORCH_CHECK(n_sizes.is_contiguous(), "n_sizes must be contiguous");
    TORCH_CHECK(k_halves.is_contiguous(), "k_halves must be contiguous");
    TORCH_CHECK(k_scales.is_contiguous(), "k_scales must be contiguous");

    const int groups = static_cast<int>(m_sizes.numel());
    TORCH_CHECK(groups > 0, "grouped call requires at least one group");
    TORCH_CHECK(b_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(sfa_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(sfb_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(c_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(n_sizes.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(k_halves.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(k_scales.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");

    TORCH_CHECK(max_m_size > 0, "max_m_size must be > 0");
    TORCH_CHECK(max_n_size > 0, "max_n_size must be > 0");
    TORCH_CHECK(block_m > 0 && block_n > 0, "block sizes must be > 0");
    TORCH_CHECK(block_m * block_n <= 1024, "block_m * block_n must be <= 1024");
    TORCH_CHECK(kpack_tile > 0, "kpack_tile must be > 0");
    TORCH_CHECK((kpack_tile % 8) == 0, "kpack_tile must be divisible by 8");

    const int scale_tile = (kpack_tile + 7) >> 3;

    const dim3 block(static_cast<unsigned int>(block_n), static_cast<unsigned int>(block_m), 1);
    const dim3 grid(
        static_cast<unsigned int>((max_n_size + block_n - 1) / block_n),
        static_cast<unsigned int>((max_m_size + block_m - 1) / block_m),
        static_cast<unsigned int>(groups)
    );

    const size_t shared_bytes = shared_bytes_for_tile(block_m, block_n, kpack_tile, scale_tile);

    nvfp4_group_gemm_kernel_grouped_pure_custom<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint64_t*>(a_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(b_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(sfa_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(sfb_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(c_ptrs.data_ptr<int64_t>()),
        m_sizes.data_ptr<int32_t>(),
        n_sizes.data_ptr<int32_t>(),
        k_halves.data_ptr<int32_t>(),
        k_scales.data_ptr<int32_t>(),
        block_m,
        block_n,
        kpack_tile,
        scale_tile
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_group_gemm_forward_grouped_fp16_cublas_cuda(
    torch::Tensor a_half_ptrs_cpu,
    torch::Tensor b_half_ptrs_cpu,
    torch::Tensor c_ptrs_cpu,
    torch::Tensor m_sizes_cpu,
    int n_size,
    int k_size
) {
    TORCH_CHECK(!a_half_ptrs_cpu.is_cuda(), "a_half_ptrs_cpu must be CPU tensor");
    TORCH_CHECK(!b_half_ptrs_cpu.is_cuda(), "b_half_ptrs_cpu must be CPU tensor");
    TORCH_CHECK(!c_ptrs_cpu.is_cuda(), "c_ptrs_cpu must be CPU tensor");
    TORCH_CHECK(!m_sizes_cpu.is_cuda(), "m_sizes_cpu must be CPU tensor");

    TORCH_CHECK(a_half_ptrs_cpu.scalar_type() == torch::kInt64, "a_half_ptrs_cpu must be torch.int64");
    TORCH_CHECK(b_half_ptrs_cpu.scalar_type() == torch::kInt64, "b_half_ptrs_cpu must be torch.int64");
    TORCH_CHECK(c_ptrs_cpu.scalar_type() == torch::kInt64, "c_ptrs_cpu must be torch.int64");
    TORCH_CHECK(m_sizes_cpu.scalar_type() == torch::kInt, "m_sizes_cpu must be torch.int32");

    TORCH_CHECK(a_half_ptrs_cpu.dim() == 1, "a_half_ptrs_cpu must be 1D");
    TORCH_CHECK(b_half_ptrs_cpu.dim() == 1, "b_half_ptrs_cpu must be 1D");
    TORCH_CHECK(c_ptrs_cpu.dim() == 1, "c_ptrs_cpu must be 1D");
    TORCH_CHECK(m_sizes_cpu.dim() == 1, "m_sizes_cpu must be 1D");

    TORCH_CHECK(a_half_ptrs_cpu.is_contiguous(), "a_half_ptrs_cpu must be contiguous");
    TORCH_CHECK(b_half_ptrs_cpu.is_contiguous(), "b_half_ptrs_cpu must be contiguous");
    TORCH_CHECK(c_ptrs_cpu.is_contiguous(), "c_ptrs_cpu must be contiguous");
    TORCH_CHECK(m_sizes_cpu.is_contiguous(), "m_sizes_cpu must be contiguous");

    const int groups = static_cast<int>(m_sizes_cpu.numel());
    TORCH_CHECK(groups > 0, "grouped call requires at least one group");
    TORCH_CHECK(a_half_ptrs_cpu.numel() == m_sizes_cpu.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(b_half_ptrs_cpu.numel() == m_sizes_cpu.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(c_ptrs_cpu.numel() == m_sizes_cpu.numel(), "all grouped tensors must have matching lengths");
    TORCH_CHECK(n_size > 0, "n_size must be > 0");
    TORCH_CHECK(k_size > 0, "k_size must be > 0");

    const int64_t* a_ptrs_raw = a_half_ptrs_cpu.data_ptr<int64_t>();
    const int64_t* b_ptrs_raw = b_half_ptrs_cpu.data_ptr<int64_t>();
    const int64_t* c_ptrs_raw = c_ptrs_cpu.data_ptr<int64_t>();
    const int32_t* m_sizes_raw = m_sizes_cpu.data_ptr<int32_t>();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const int m_size = static_cast<int>(m_sizes_raw[g]);
        TORCH_CHECK(m_size > 0, "m_sizes must contain positive values");

        // Row-major mapping:
        //   C_row[M,N] = A_row[M,K] @ B_row[N,K]^T
        // to column-major GEMM:
        //   C_col[N,M] = B_row[N,K] @ A_row[M,K]^T
        // with operands interpreted as column-major (KxN) and (KxM):
        //   op(A)=T => (NxK), op(B)=N => (KxM).
        const void* b_row = reinterpret_cast<const void*>(static_cast<uintptr_t>(b_ptrs_raw[g]));
        const void* a_row = reinterpret_cast<const void*>(static_cast<uintptr_t>(a_ptrs_raw[g]));
        void* c_col = reinterpret_cast<void*>(static_cast<uintptr_t>(c_ptrs_raw[g]));

        TORCH_CUDABLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            n_size,          // m (column-major)
            m_size,          // n (column-major)
            k_size,          // k
            &alpha,
            b_row,
            CUDA_R_16F,
            k_size,          // lda for (KxN) with op(T)
            a_row,
            CUDA_R_16F,
            k_size,          // ldb
            &beta,
            c_col,
            CUDA_R_16F,
            n_size,          // ldc
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_group_gemm_forward_grouped_cuda", &nvfp4_group_gemm_forward_grouped_cuda);
    m.def("nvfp4_group_gemm_forward_grouped_fp16_cublas_cuda", &nvfp4_group_gemm_forward_grouped_fp16_cublas_cuda);
}
