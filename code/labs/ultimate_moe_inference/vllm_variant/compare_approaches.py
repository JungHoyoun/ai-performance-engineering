"""Compare our implementation vs vLLM.

Side-by-side comparison showing:
1. Both use similar optimization techniques
2. Educational value of our from-scratch approach
3. Performance characteristics
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


@dataclass
class ComparisonResult:
    """Result of comparison."""
    
    approach: str
    latency_ms: float
    tokens_per_sec: float
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    memory_gb: Optional[float] = None


def compare_approaches(
    model_name: str = "openai/gpt-oss-20b",
    tensor_parallel: int = 1,
    output_path: Optional[Path] = None,
) -> Dict[str, ComparisonResult]:
    """Run comparison between our implementation and vLLM.
    
    Args:
        model_name: Model to benchmark
        tensor_parallel: Number of GPUs
        output_path: Path to save results
        
    Returns:
        Dictionary mapping approach to results
    """
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    results: Dict[str, ComparisonResult] = {}
    
    print("=" * 70)
    print("Implementation Comparison: Our Approach vs vLLM")
    print("=" * 70)
    print()
    print("This comparison demonstrates that:")
    print("1. vLLM uses the same optimization techniques we teach")
    print("2. Our from-scratch approach provides educational transparency")
    print("3. Both achieve similar performance characteristics")
    print()
    
    # Our baseline
    print("[1/4] Running our baseline implementation...")
    try:
        from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
        
        config = InferenceConfig(
            model_name=model_name,
            tensor_parallel=tensor_parallel,
            benchmark_iterations=5,
        )
        benchmark = BaselineUltimateInference(config)
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
        result = harness.benchmark(benchmark)
        
        results["our_baseline"] = ComparisonResult(
            approach="Our Baseline (Ch1-6 only)",
            latency_ms=result.timing.mean_ms if result.timing else 0,
            tokens_per_sec=benchmark.last_metrics.tokens_per_sec if benchmark.last_metrics else 0,
            ttft_ms=benchmark.last_metrics.ttft_ms if benchmark.last_metrics else None,
        )
        
        del benchmark
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
    
    # Our optimized
    print("[2/4] Running our optimized implementation...")
    try:
        from optimized_ultimate_inference import OptimizedUltimateInference, OptimizedConfig
        
        config_path = Path(__file__).parent.parent / "configs" / "single_gpu.yaml"
        config = OptimizedConfig.from_yaml(config_path)
        config.tensor_parallel = tensor_parallel
        
        benchmark = OptimizedUltimateInference(config)
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
        result = harness.benchmark(benchmark)
        
        results["our_optimized"] = ComparisonResult(
            approach="Our Optimized (All chapters)",
            latency_ms=result.timing.mean_ms if result.timing else 0,
            tokens_per_sec=benchmark.last_metrics.tokens_per_sec if benchmark.last_metrics else 0,
            ttft_ms=benchmark.last_metrics.ttft_ms if benchmark.last_metrics else None,
            memory_gb=benchmark.last_metrics.peak_memory_gb if benchmark.last_metrics else None,
        )
        
        del benchmark
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
    
    # vLLM baseline
    print("[3/4] Running vLLM baseline...")
    try:
        from baseline_vllm import BaselineVLLMBenchmark
        
        benchmark = BaselineVLLMBenchmark(
            model_name=model_name,
            tensor_parallel=tensor_parallel,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
        result = harness.benchmark(benchmark)
        
        results["vllm_baseline"] = ComparisonResult(
            approach="vLLM Baseline",
            latency_ms=result.timing.mean_ms if result.timing else 0,
            tokens_per_sec=benchmark.last_metrics.tokens_per_sec if benchmark.last_metrics else 0,
            ttft_ms=benchmark.last_metrics.ttft_ms if benchmark.last_metrics else None,
        )
        
        del benchmark
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error (vLLM not available?): {e}")
    
    # vLLM optimized
    print("[4/4] Running vLLM optimized...")
    try:
        from optimized_vllm import OptimizedVLLMBenchmark
        
        benchmark = OptimizedVLLMBenchmark(
            model_name=model_name,
            tensor_parallel=tensor_parallel,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
        result = harness.benchmark(benchmark)
        
        results["vllm_optimized"] = ComparisonResult(
            approach="vLLM Optimized",
            latency_ms=result.timing.mean_ms if result.timing else 0,
            tokens_per_sec=benchmark.last_metrics.tokens_per_sec if benchmark.last_metrics else 0,
            ttft_ms=benchmark.last_metrics.ttft_ms if benchmark.last_metrics else None,
        )
        
        del benchmark
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error (vLLM not available?): {e}")
    
    # Print comparison
    print()
    print("=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print()
    print(f"{'Approach':<30} {'Latency (ms)':<15} {'Tok/s':<12} {'TTFT (ms)':<12}")
    print("-" * 70)
    
    for name, r in results.items():
        ttft = f"{r.ttft_ms:.2f}" if r.ttft_ms else "N/A"
        print(f"{r.approach:<30} {r.latency_ms:<15.2f} {r.tokens_per_sec:<12.1f} {ttft:<12}")
    
    print()
    print("=" * 70)
    print("Optimization Technique Mapping")
    print("=" * 70)
    print()
    print("| Our Chapter | Technique           | vLLM Equivalent           |")
    print("|-------------|---------------------|---------------------------|")
    print("| Ch10        | FlashAttention      | FlashAttention/PagedAttn  |")
    print("| Ch12        | CUDA Graphs         | CUDA Graphs               |")
    print("| Ch13        | FP8 KV Cache        | FP8 KV Cache              |")
    print("| Ch16        | PagedAttention      | PagedAttention (core)     |")
    print("| Ch16        | Prefix Caching      | Automatic Prefix Caching  |")
    print("| Ch17        | Continuous Batching | Continuous Batching       |")
    print("| Ch18        | Speculative Decode  | Speculative Decoding      |")
    print()
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare Our Implementation vs vLLM")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    
    compare_approaches(
        model_name=args.model,
        tensor_parallel=args.tensor_parallel,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


