"""Compare baseline/optimized plan metrics for key pairs (tool helper)."""

from __future__ import annotations

from typing import List

from labs.moe_parallelism.plan import PlanEvaluator, format_report  # noqa: E402
from labs.moe_parallelism.scenarios import get_scenario_pairs  # noqa: E402

SCENARIOS: List[str] = ["gpt_gb200", "deepseek_gb200"]

METRIC_FIELDS = [
    ("step_ms", "estimated_step_ms"),
    ("throughput_tokens_s", "throughput_tokens_per_s"),
    ("bubble", "bubble_fraction"),
    ("params_gb", "params_gb"),
    ("optimizer_gb", "optimizer_gb"),
    ("grads_gb", "grad_gb"),
    ("activations_gb", "activation_gb"),
    ("margin_gb", "memory_margin_gb"),
]


def _load(scenario_name: str, variant: str):
    scenarios = get_scenario_pairs()
    scenario = scenarios[scenario_name]
    evaluator = PlanEvaluator(scenario.cluster, scenario.model)
    plan = scenario.baseline if variant == "baseline" else scenario.optimized
    return evaluator.analyze(plan)


def _fmt(val: float, key: str) -> str:
    if key in {"step_ms", "throughput_tokens_s"}:
        return f"{val:,.0f}"
    if "gb" in key:
        return f"{val:,.1f}"
    if key == "bubble":
        return f"{val*100:.1f}%"
    return f"{val:.3f}"


def compare_pair(scenario_name: str) -> None:
    base_report = _load(scenario_name, "baseline")
    opt_report = _load(scenario_name, "optimized")
    print(f"\n=== {scenario_name} ===")
    print(format_report(base_report))
    print("---")
    print(format_report(opt_report))
    print("\nMetric comparison (baseline -> optimized):")
    rows = []
    for label, attr in METRIC_FIELDS:
        bval = getattr(base_report, attr)
        oval = getattr(opt_report, attr)
        rows.append((label, bval, oval, (oval / bval) if bval else 0.0))
    header = f"{'metric':<18}{'baseline':>15}{'optimized':>15}{'opt/base':>12}"
    print(header)
    print("-" * len(header))
    for label, bval, oval, ratio in rows:
        print(f"{label:<18}{_fmt(bval, label):>15}{_fmt(oval, label):>15}{ratio:>12.2f}")


def main() -> None:
    for scenario in SCENARIOS:
        compare_pair(scenario)


if __name__ == "__main__":
    main()
