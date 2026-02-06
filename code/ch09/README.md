# Chapter 9 - Arithmetic Intensity & Kernel Fusion

## Summary
Explores how to move workloads along the roofline: raise arithmetic intensity with tiling, fuse memory-bound kernels, and deploy CUTLASS/Triton/inline-PTX paths built for Blackwell tensor cores.

## Learning Goals
- Separate compute-bound vs memory-bound behaviors and adjust kernels accordingly.
- Design micro-tiling schedules that balance register pressure with data reuse.
- Leverage CUTLASS and Triton for rapid iteration while keeping custom CUDA fallbacks.
- Fuse reduction-heavy kernels (e.g., norm + activation) to eliminate redundant memory trips.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_compute_bound.py`, `optimized_compute_bound.py`, `baseline_memory_bound.py`, `optimized_memory_bound.py` | Reference kernels that isolate compute vs bandwidth ceilings and demonstrate tuning strategies. |
| `baseline_micro_tiling_matmul.cu`, `baseline_micro_tiling_matmul.py`, `optimized_micro_tiling_matmul.cu`, `optimized_micro_tiling_matmul.py` | Micro-tiling matmuls with explicit register blocking and cp.async prefetch. |
| `baseline_cutlass_gemm.cu`, `baseline_cutlass_gemm.py`, `optimized_cutlass_gemm.cu`, `optimized_cutlass_gemm.py` | Library GEMM baselines for comparing hand-tuned kernels against vendor libraries. |
| `baseline_cublaslt_gemm.cu`, `baseline_cublaslt_gemm.py`, `optimized_cublaslt_gemm.cu`, `optimized_cublaslt_gemm.py`, `tcgen05_pipelined.cu` | cuBLASLt-driven matmuls and tcgen05 pipeline kernels showcasing tcgen05 lowering and occupancy tuning. |
| `baseline_cute_dsl_nvfp4_gemm.cu`, `baseline_cute_dsl_nvfp4_gemm.py`, `optimized_cute_dsl_nvfp4_gemm.cu`, `optimized_cute_dsl_nvfp4_gemm.py` | CuTe-DSL-inspired NVFP4 GEMM pair (competition shapes) with baseline vs TMA warp-specialized schedule. |
| `baseline_cutlass_gemm_fp4_all_concepts.cu`, `baseline_cutlass_gemm_fp4_all_concepts.py`, `optimized_cutlass_gemm_fp4_all_concepts.cu`, `optimized_cutlass_gemm_fp4_all_concepts.py` | CUTLASS NVFP4 GEMM pair that isolates schedule choice: auto scheduling vs explicit `KernelTmaWarpSpecialized1SmNvf4Sm100`. |
| `baseline_fused_l2norm.cu`, `baseline_fused_l2norm.py`, `optimized_fused_l2norm.cu`, `optimized_fused_l2norm.py`, `fusedL2Norm/` | Fusion examples that merge L2 norm + scaling while staying numerically stable. |
| `baseline_triton.py`, `optimized_triton.py` | Triton counterparts for quick prototyping and verifying compiler-generated PTX on Blackwell. |
| `baseline_tcgen05_tma_pipeline.py`, `optimized_tcgen05_tma_pipeline.py`, `two_stage_pipeline.cu` | Producer/consumer pipelines emphasizing staged TMA loads and inline PTX hooks. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness hooks plus regression thresholds for every example. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch09/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch09
python -m cli.aisp bench run --targets ch09 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python baseline_compute_bound.py --summaries` reports much higher arithmetic intensity than `baseline_memory_bound.py`, matching the roofline plots.
- `python optimized_cublaslt_gemm.py --sizes 4096 4096 8192` improves throughput relative to `baseline_cublaslt_gemm.py` on the same device.
- `python compare.py --examples fused_l2norm` confirms numerically identical outputs before and after fusion.

## Notes
- `inline_ptx_example.cu` demonstrates how to wrap tcgen05 intrinsics safely with architecture guards.
- `baseline_cute_dsl_nvfp4_gemm.cu` / `optimized_cute_dsl_nvfp4_gemm.cu` are inspired by the CuTe DSL NVFP4 kernel walkthrough: https://obolensky.xyz/blog/nvfp4_gemm_kernel_explanation/
- `requirements.txt` includes Triton nightly pinning so the kernels track PyTorch 2.10-dev features.
