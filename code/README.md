# AI Systems Performance Engineering - Code and Benchmark Verifications

[![O'Reilly Book](../img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

## Summary
Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

## Benchmark Verification & Validity Protections

This section describes the current benchmark verification system (correctness + workload equivalence) and the harness validity/anti-cheat protections.

This content used to live in `docs/implementation_status.md` and is now maintained in `README.md`.

### TODO

- No open items right now.

### Coverage

| Area | Coverage |
| --- | --- |
| Interface standardization & migration | 11/11 (100%) |
| Anti-cheat protections | 32/32 (100%) |
| Harness & CLI integration | 15/15 (100%) |
| Triton benchmarking best practices | 17/17 (100%) |
| **Total** | **75/75 (100%)** |

### Verification Model

#### Post-timing verification
Verification is **post-timing**: benchmarks run once for timing (warmup + measured iterations), then the harness compares outputs from those same runs.

#### Required benchmark interface
Benchmarks are required to provide explicit verification metadata:
- `get_input_signature()` (workload equivalence)
- `get_verify_output()` (output tensor(s) to compare)
- `get_output_tolerance()` (rtol/atol policy)
- `validate_result()` (sanity checks like NaN/Inf)

The harness is **fail-fast**: missing methods are treated as errors, not auto-inferred.

#### Verification payload mixin
`core/benchmark/verification_mixin.py` provides `VerificationPayloadMixin` and `_set_verification_payload()` so benchmarks can register verification inputs/outputs without hand-rolling boilerplate. The recommended starting point is `templates/benchmark_compliant.py`.

#### Output & overfitting checks
`core/benchmark/verify_runner.py` compares baseline vs optimized outputs and enforces:
- dtype-aware tolerances (via `core/benchmark/verification.py`)
- golden output caching (`GoldenOutputCache`)
- fresh-input and jitter checks (to detect cached/constant outputs)
- workload invariants + signature matching

#### Enforcement phases
Verification can run in phases (`DETECT`, `QUARANTINE`, `GATE`) using `core/benchmark/verification.py` + `core/benchmark/quarantine.py` and the CLI `--verify-phase` option.

### CUDA-L1 Reward-Hacking Protections

Reward-hacking cases identified in the CUDA-L1 paper are covered by the harness:

| Case | Protection |
| --- | --- |
| Improper timing measurement | Full device sync + `StreamAuditor` |
| Lazy evaluation | `force_tensor_evaluation()` |
| Hyperparameter manipulation | `InputSignature` + signature matching |
| Result caching | Fresh-input check |
| Mathematical short-circuit | Workload invariant check |
| Pre-allocated tensors | `MemoryAllocationTracker` |
| Direct shape matching | Signature validation |
| Pre-computed parameters | `check_setup_precomputation()` |

### Validity / Anti-Cheat Protections (Implementation Map)

#### Timing & measurement correctness

| Protection | Primary implementation |
| --- | --- |
| Full device sync | `core/harness/benchmark_harness.py` (`full_device_sync=True`) |
| Adaptive iterations | `core/harness/benchmark_harness.py` (`adaptive_iterations=True`) |
| Event timing cross-validation | `core/harness/benchmark_harness.py` (`cross_validate_timing=True`) |
| Warmup buffer isolation | `core/harness/benchmark_harness.py` (`isolate_warmup_cache=True`) |
| L2 cache clearing | `core/harness/benchmark_harness.py` (`clear_l2_cache`) |
| GPU clock locking | `core/harness/benchmark_harness.py` (`lock_gpu_clocks()`) |
| GC disabled during timing | `core/harness/validity_checks.py` (`gc_disabled()`) |
| Config immutability | `core/harness/benchmark_harness.py` (`enforce_config_immutability=True`) |
| Memory pool reset | `core/harness/benchmark_harness.py` (`reset_memory_pool=True`) |

#### Streams, graphs, and CUDA-specific protections

| Protection | Primary implementation |
| --- | --- |
| Stream auditing + sync completeness | `core/harness/validity_checks.py` (`StreamAuditor`, `audit_streams()`, `check_stream_sync_completeness()`) |
| Graph capture cheat detection | `core/harness/validity_checks.py` (`GraphCaptureCheatDetector`, `check_graph_capture_integrity()`) |
| Setup precomputation detection | `core/harness/validity_checks.py` (`check_setup_precomputation()`) |
| Force tensor evaluation | `core/harness/validity_checks.py` (`force_tensor_evaluation()`) |
| CUDA verify header | `core/common/headers/cuda_verify.cuh` (`VERIFY_CHECKSUM`) |
| CUDA binary symbol inspection | `core/benchmark/cuda_binary_benchmark.py` (`check_perf_binary_clean()`) |

#### Workload & output protections

| Protection | Primary implementation |
| --- | --- |
| Signature matching | `core/benchmark/verify_runner.py` (`_verify_signatures_match()`) |
| Workload invariant checks | `core/benchmark/verify_runner.py` (`_check_workload_invariants()`) |
| Fresh-input check | `core/benchmark/verify_runner.py` (`_run_fresh_input_check()`) |
| Jitter check | `core/benchmark/verify_runner.py` (`_run_jitter_check()`) |
| Golden output caching | `core/benchmark/verify_runner.py` (`GoldenOutputCache`) |
| Seed mutation detection | `core/benchmark/verification.py` (`detect_seed_mutation()`) |
| Input-output aliasing check | `core/benchmark/verify_runner.py` (`_check_input_output_aliasing()`) |
| Skip-flag detection | `core/benchmark/quarantine.py` (`detect_skip_flags()`) |

#### Environment & distributed protections

| Protection | Primary implementation |
| --- | --- |
| Environment validation | `core/harness/validity_checks.py` (`validate_environment()`) |
| GPU state capture (power/thermals) | `core/harness/validity_checks.py` (`capture_gpu_state()`) |
| Compile cache clearing | `core/harness/validity_checks.py` (`clear_compile_cache()`) |
| Distributed topology verification | `core/benchmark/verify_runner.py` + `core/harness/validity_checks.py` (`verify_distributed()`, `gather_rank_outputs()`, `verify_distributed_outputs()`) |

Note: `validate_environment()` treats virtualization (hypervisor present) as invalid; benchmarks are supported only on bare metal.

### CLI / CI Entry Points

| Task | Command |
| --- | --- |
| Audit verification compliance | `python -m cli.aisp bench audit --all` |
| Verify baseline/optimized correctness | `python -m cli.aisp bench verify -t ch12:graph_bandwidth` |
| Generate verification report | `python -m cli.aisp bench verify-report --gpu H100` |
| Generate quarantine report | `python -m cli.aisp bench quarantine-report --format markdown` |
| Print theoretical peaks | `python -m cli.aisp bench theoretical-peak --gpu H100` |

### Files Reference

| Category | File |
| --- | --- |
| Core data models | `core/benchmark/verification.py` |
| Verification mixin | `core/benchmark/verification_mixin.py` |
| Verify runner | `core/benchmark/verify_runner.py` |
| Quarantine manager | `core/benchmark/quarantine.py` |
| Benchmark contract | `core/benchmark/contract.py` |
| Benchmark harness | `core/harness/benchmark_harness.py` |
| Validity checks | `core/harness/validity_checks.py` |
| L2 cache utils | `core/harness/l2_cache_utils.py` |
| Verification reports | `core/analysis/reporting/verification_report.py` |
| CLI commands | `core/benchmark/bench_commands.py` |
| Audit script | `core/scripts/audit_verification_compliance.py` |
| Migration script | `core/scripts/migrate_verification_methods.py` |
| Pair validation | `core/scripts/validate_benchmark_pairs.py` |
| CI compliance check | `core/scripts/ci/check_verification_compliance.py` |

## Learning Goals
- Understand how the chapters, labs, and shared tooling fit together.
- Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.
- Run the benchmark harness directly or through the Typer CLI for automated artifact capture.
- Validate peak hardware characteristics before grading optimizations against stored expectations.

## Directory Layout
| Path | Description |
| --- | --- |
| `ch01` - `ch20` | One directory per chapter with baseline/optimized benchmarks, workload configs, and `compare.py` harness entrypoints. |
| `labs/` | Deep-dive labs for matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more. |
| `core/benchmark/`, `profiling/`, `core/`, `optimization/`, `analysis/` | Shared harness, logging, workload metadata, profiling, and optimization utilities used by every chapter. |
| `python -m cli.aisp bench` | Typer-based CLI for running and profiling targets with reproducible artifacts. |
| `docs/` + `core/scripts/` | Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`). |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_latest.txt
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.
- Use `python core/harness/run_benchmarks.py --targets ch*` for automated regression suites.
- `python core/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.

## Validation Checklist
- `pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.
- `python core/benchmark/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.
- `python -m cli.aisp bench verify -t ch12:graph_bandwidth` validates baseline/optimized correctness.
- `python -m cli.aisp bench audit --all` checks verification compliance (signatures, outputs, workload metadata).

## Docs
- `docs/api-reference.md` (CLI/MCP/Dashboard/Python API overview)
- `docs/benchmark_harness_guide.md` (harness architecture and run modes)
- `docs/perf_intake_and_triage.md` (standard intake bundle for investigations)

## Notes
- `core/scripts/profile_all_workloads.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.
- `benchmark_profiles/` and `artifacts/` hold run outputs; clean them via `python cleanup.py` when rotating hardware.
