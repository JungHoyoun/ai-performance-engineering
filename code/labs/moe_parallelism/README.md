# Lab - MoE Parallelism Planner

## Summary
Scenario planning tool for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules.

## Learning Goals
- Quantify memory budgets for experts, routers, and KV caches before deploying models.
- Explore different grouping strategies (hashing, topology-aware) and their throughput impact.
- Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.
- Simulate pipeline schedules to identify bottlenecks before touching production systems.

## Directory Layout
| Path | Description |
| --- | --- |
| `run_lab.py`, `scenarios.py`, `plan.py` | Tool entry point + canonical scenario definitions and sizing model. |
| `benchmarking.py` | Optional harness wrapper for ad-hoc validation and integration. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_parallelism
python -m cli.aisp bench run --targets labs/moe_parallelism --profile minimal
```
- Targets follow the `labs/moe_parallelism:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_parallelism:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp tools moe-parallelism -- --scenario memory_budget` runs a single scenario via the tool registry.
- `python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200` runs a larger cluster scenario.
- `python labs/moe_parallelism/run_lab.py --scenario deepseek_gb200` runs the planner directly (without aisp).

## Notes
- Baseline vs optimized here are *planning* scenarios (different designs), not comparable performance benchmarks.
- `plan.py` centralizes scenario definitions so you only update one file when adding a new topology.
