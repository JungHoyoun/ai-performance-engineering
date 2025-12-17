# ADR 0001: Harness Organization and Configuration Roadmap

## Status
Accepted (2025-12-17)

## Context
The benchmarking harness is production-grade and feature rich, but several modules are large:

- `core/harness/benchmark_harness.py` is the core execution engine and includes config, execution paths, and profiling integration.
- `core/harness/run_benchmarks.py` and `core/harness/run_all_benchmarks.py` combine orchestration, reporting, and environment management.
- `BenchmarkConfig` has grown large to support many cross-cutting concerns (timing, profiling, distributed launch, validation, artifact capture).

These “monolithic” shapes are common in harnesses where:
- correctness and reproducibility depend on strict coordination across many subsystems, and
- changes must minimize churn for chapter examples that import or reference harness APIs.

## Decision
1. Keep the current file layout short-term to preserve stability and chapter/book alignment.
2. Prefer incremental extraction into small, composable modules behind stable interfaces rather than a single “big bang” refactor.
3. Treat configuration decomposition as a compatibility-sensitive change: introduce grouped config objects without breaking existing config semantics.

## Consequences
- Short-term: key modules remain large, but correctness and ergonomics remain stable.
- Medium-term: refactoring becomes safer because extraction is staged and validated by the existing harness audit + test suite.

## Follow-ups (Roadmap)
- Split `BenchmarkConfig` into focused groups (e.g., `TimingConfig`, `ProfilingConfig`, `DistributedConfig`, `ValidationConfig`) while keeping a compatibility layer for existing call sites.
- Extract `run_benchmarks.py` orchestration into cohesive modules (e.g., runner, result collector, report renderer) while keeping the CLI entrypoints stable.
- Replace process-wide “quick wins” globals with an explicit, idempotent configurator owned by `BenchmarkHarness` initialization.
- Centralize GPU state operations behind a `GPUStateManager` (single place for reset, cache clear, and telemetry snapshots).

