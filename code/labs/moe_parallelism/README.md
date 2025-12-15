# Lab - MoE Parallelism Planner

## Summary
Provides scenario planning for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules.

This lab is a planning tool (CPU-side analysis). The "baseline" vs "optimized" plans are intentionally different designs, so they are not exposed as `aisp bench run` baseline/optimized benchmark pairs.

## Learning Goals
- Quantify memory budgets for experts, routers, and KV caches before deploying models.
- Explore different grouping strategies (hashing, topology-aware) and their throughput impact.
- Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.
- Simulate pipeline schedules to identify bottlenecks before touching production systems.

## Directory Layout
| Path | Description |
| --- | --- |
| `plan.py` | Core sizing model + report formatting. |
| `scenarios.py` | Scenario pairs (baseline vs optimized) used by the tool. |
| `run_lab.py` | Tool entrypoint that prints scenario reports + comparisons. |
| `compare_pairs.py` | Helper for quick side-by-side comparisons. |

## Running the Tool
```bash
python -m cli.aisp tools moe-parallelism -- --scenario memory_budget
python -m cli.aisp tools moe-parallelism -- --scenario pipeline_schedule
python -m cli.aisp tools moe-parallelism -- --scenario deepseek_gb200
```
- Omit `--scenario` to run all scenarios in `labs/moe_parallelism/scenarios.py`.

## Validation Checklist
- `python -m cli.aisp tools moe-parallelism -- --scenario memory_budget` runs and prints baseline/optimized reports.

## Notes
- `plan.py` centralizes the sizing model so scenario edits stay small and readable.
