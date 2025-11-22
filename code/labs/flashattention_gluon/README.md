# Lab - FlashAttention Gluon

## Summary
Contrast a naive unfused attention (matmul + softmax + matmul) with a fused, warp-specialized FlashAttention kernel. The optimized path prefers a Gluon/Triton kernel; if Gluon is unavailable, it falls back to the flash-attn fused kernel (warp-specialized on Blackwell).

## Workloads
- `labs.flashattention_gluon.baseline_flashattention_gluon`: unfused attention, math softmax path.
- `labs.flashattention_gluon.optimized_flashattention_gluon`: fused Gluon/flash-attn kernel.

## Running
```bash
cd ai-performance-engineering
python tools/cli/benchmark_cli.py list-targets --chapter labs/flashattention_gluon
python tools/cli/benchmark_cli.py run --targets labs/flashattention_gluon:baseline_flashattention_gluon --profile minimal
python tools/cli/benchmark_cli.py run --targets labs/flashattention_gluon:optimized_flashattention_gluon --profile minimal
```

## Requirements
- NVIDIA GPU with CUDA.
- Gluon or flash-attn installed (setup.sh installs flash-attn; Gluon install is attempted there as well).

## What to inspect
- NVTX ranges `flashattention_baseline_unfused` vs `flashattention_optimized_<provider>`.
- Provider metric indicates whether Gluon or flash-attn was used.
- Expect fused path to show fewer kernels and higher throughput due to warp specialization and on-chip softmax.

## Baseline vs optimized checklist (Blackwell)
Use this when converting a naive FlashAttention to a warp-specialized, TMA-driven kernel on Blackwell:
- Pipeline: baseline interleaves loads/MMAs/softmax in one warp group; optimized uses warp-specialized partitions (load/TMA, QK MMA, softmax subtiles, PV MMA) with M-barriers.
- Layouts: baseline relies on compiler defaults; optimized assigns explicit block/MMA/swizzle layouts and places conversions deliberately.
- Softmax: baseline does a single softmax after MMAs; optimized subtiles softmax along M so two softmax partitions overlap with MMAs.
- Sync: baseline leans on implicit barriers; optimized orders M-barrier acquires/releases so non-critical partitions wait first.
- Memory: baseline lets compiler place operands; optimized parks tensors (e.g., PV operand) in tensor memory with explicit lifetimes and handoffs.
- Loads: baseline issues per-warp small loads; optimized uses TMA bulk loads so one warp saturates memory while others compute.
- Critical path: baseline contends MMAs vs softmax; optimized schedules so softmax dominates while MMAs/loads are overlapped.
- Autotune: baseline leaves schedule to heuristics; optimized encodes human-chosen partition counts, subtiles, and barrier ordering in source/meta-programming helpers.
- Masking: baseline handles masking in the monolithic loop; optimized folds masking into softmax partitions to overlap with MMAs.
