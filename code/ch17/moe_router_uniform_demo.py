"""moe_router_uniform_demo.py - Chapter 17 MoE uniform routing demo (non-benchmark).

Demonstrates a topology-agnostic router that assigns each token to a random
expert uniformly (no locality awareness). This is a demo/tool, not a comparable
baseline-vs-optimized benchmark pair.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 17 MoE uniform router demo")
    parser.add_argument("--tokens", type=int, default=4096, help="Number of tokens to route.")
    parser.add_argument("--num-experts", type=int, default=16, help="Total expert count.")
    parser.add_argument("--seed", type=int, default=42, help="Python RNG seed.")
    args = parser.parse_args()

    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")

    rng = random.Random(int(args.seed))
    experts = list(range(int(args.num_experts)))
    assignments = [rng.choice(experts) for _ in range(int(args.tokens))]
    counts = Counter(assignments)

    print("Uniform router demo")
    print(f"tokens:      {args.tokens}")
    print(f"num_experts: {args.num_experts}")
    print(f"seed:        {args.seed}")
    print("expert_load:")
    for expert_id in range(int(args.num_experts)):
        print(f"  {expert_id:>3}: {counts.get(expert_id, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
