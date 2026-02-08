# NVFP4 Group GEMM (B200 Competition Shapes)

This lab mirrors the input shapes from the GPU MODE `nvfp4_group_gemm` leaderboard and is intended
to let us iterate on a strong contender using the AISP harness (clock locking, correctness checks,
Nsight traces), then port the resulting kernel strategy back into a Popcorn `submission.py`.

## Targets

List targets:
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
```

Run the suite (baseline vs optimized) with profiling:
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --update-expectations
```

Notes:
- The baseline uses a straightforward CuTe DSL implementation (close to the official starter).
- The CuTe DSL optimized variants remove per-call allocations by caching metadata tensors and
  tensormap buffers in `setup()`, matching the competition evaluation style where `custom_kernel()`
  is called repeatedly with identical shapes.
- The CUTLASS optimized variants (`optimized_*_cutlass.py`) use a single-launch, device-scheduled
  grouped GEMM kernel adapted from `third_party/cutlass/examples/75_blackwell_grouped_gemm/`.
  Tuning knobs (optional):
  - `AISP_NVFP4_GROUP_GEMM_CLUSTER_M`, `AISP_NVFP4_GROUP_GEMM_CLUSTER_N`
  - `AISP_NVFP4_GROUP_GEMM_RASTER_ORDER`
  - `AISP_NVFP4_GROUP_GEMM_USE_PDL`
