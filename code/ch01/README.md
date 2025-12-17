# Chapter 1 - Performance Fundamentals

## Summary
Establishes the baseline benchmarking discipline with a simple training-loop goodput benchmark and a small CUDA GEMM case study. The goal is to ground later optimizations in repeatable measurement, equivalent workloads, and verifiable outputs.

## Learning Goals
- Profile a minimal PyTorch training loop with the shared harness and reason about throughput vs latency.
- Apply basic optimizations (FP16 and batch fusion) without changing the algorithmic workload.
- Compare hand-written GEMM kernels in batched vs. strided forms to understand arithmetic intensity.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_performance.py`, `optimized_performance.py`, `optimized_performance_fp16.py`, `optimized_performance_fusion.py` | Goodput-focused training loop suite: FP32 baseline (microbatches + gradient accumulation), FP16-only, fusion-only (FP32), and combined FP16+fusion. |
| `baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu` | CUDA GEMM variants (single, batched, strided) used to illustrate launch amortization and memory coalescing. |
| `compare.py`, `workload_config.py`, `arch_config.py`, `expectations_b200.json` | Harness entrypoint, workload shapes, architecture overrides, and stored expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch01/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python compare.py` reports optimized_performance achieving >=2x tokens/sec vs the baseline on default microbatch sizes.
- Running `make && ./baseline_gemm_sm100` vs `./optimized_gemm_batched_sm100` shows a substantial drop in launch count and total runtime.

## Notes
- `requirements.txt` pins lightweight extras (Typer, tabulate) used by helper scripts.
- `Makefile` builds the CUDA GEMM binaries with SM-specific suffixes for quick diffing.
