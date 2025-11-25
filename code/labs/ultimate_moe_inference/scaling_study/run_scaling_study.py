"""GPU Scaling Study - Benchmark across 1, 2, 4, 8 GPUs.

Measures scaling efficiency and speedup as GPU count increases.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig


@dataclass
class ScalingResult:
    """Result for a single GPU configuration."""
    
    num_gpus: int
    baseline_ms: float
    optimized_ms: float
    speedup: float
    tokens_per_sec: float
    scaling_efficiency: float  # vs linear scaling
    
    # Detailed metrics
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    peak_memory_gb: Optional[float] = None


def run_scaling_study(
    model_size: str = "20b",
    max_gpus: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Dict[int, ScalingResult]:
    """Run benchmark across 1, 2, 4, 8 GPUs and collect metrics.
    
    Args:
        model_size: Model size ("20b" or "120b")
        max_gpus: Maximum GPUs to test (None = all available)
        output_path: Path to save results JSON
        
    Returns:
        Dictionary mapping GPU count to ScalingResult
    """
    from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
    from optimized_ultimate_inference import OptimizedUltimateInference, OptimizedConfig
    
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if max_gpus:
        available_gpus = min(available_gpus, max_gpus)
    
    # GPU counts to test
    gpu_counts = [g for g in [1, 2, 4, 8] if g <= available_gpus]
    
    if not gpu_counts:
        print("No GPUs available for testing")
        return {}
    
    results: Dict[int, ScalingResult] = {}
    single_gpu_baseline: Optional[float] = None
    
    for num_gpus in gpu_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_gpus} GPU(s)")
        print(f"{'='*60}")
        
        # Select model based on GPU count
        if num_gpus >= 4 and model_size == "120b":
            model_name = "openai/gpt-oss-120b"
        else:
            model_name = "openai/gpt-oss-20b"
        
        # Baseline config
        baseline_config = InferenceConfig(
            model_name=model_name,
            tensor_parallel=num_gpus,
            benchmark_iterations=5,
        )
        
        # Run baseline
        print(f"\nRunning baseline (no optimizations)...")
        try:
            baseline = BaselineUltimateInference(baseline_config)
            harness = BenchmarkHarness(
                mode=BenchmarkMode.CUSTOM,
                config=baseline.get_config(),
            )
            baseline_result = harness.benchmark(baseline)
            baseline_ms = baseline_result.timing.mean_ms if baseline_result.timing else 0.0
        except Exception as e:
            print(f"  Baseline failed: {e}")
            baseline_ms = 0.0
        
        # Optimized config
        config_path = Path(__file__).parent.parent / "configs" / f"multi_gpu_{num_gpus}.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "configs" / "single_gpu.yaml"
        
        # Run optimized
        print(f"\nRunning optimized (all optimizations)...")
        try:
            optimized_config = OptimizedConfig.from_yaml(config_path)
            optimized_config.tensor_parallel = num_gpus
            optimized = OptimizedUltimateInference(optimized_config)
            harness = BenchmarkHarness(
                mode=BenchmarkMode.CUSTOM,
                config=optimized.get_config(),
            )
            optimized_result = harness.benchmark(optimized)
            optimized_ms = optimized_result.timing.mean_ms if optimized_result.timing else 0.0
            
            # Get detailed metrics
            metrics = getattr(optimized, 'last_metrics', None)
            ttft_ms = metrics.ttft_ms if metrics else None
            tpot_ms = metrics.tpot_ms if metrics else None
            tokens_per_sec = metrics.tokens_per_sec if metrics else 0.0
            peak_memory_gb = metrics.peak_memory_gb if metrics else None
        except Exception as e:
            print(f"  Optimized failed: {e}")
            optimized_ms = 0.0
            ttft_ms = None
            tpot_ms = None
            tokens_per_sec = 0.0
            peak_memory_gb = None
        
        # Calculate metrics
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
        
        # Store single GPU baseline for scaling efficiency
        if num_gpus == 1:
            single_gpu_baseline = optimized_ms
        
        # Scaling efficiency (how close to linear scaling)
        if single_gpu_baseline and optimized_ms > 0:
            ideal_time = single_gpu_baseline / num_gpus
            scaling_efficiency = ideal_time / optimized_ms
        else:
            scaling_efficiency = 1.0
        
        result = ScalingResult(
            num_gpus=num_gpus,
            baseline_ms=baseline_ms,
            optimized_ms=optimized_ms,
            speedup=speedup,
            tokens_per_sec=tokens_per_sec,
            scaling_efficiency=scaling_efficiency,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            peak_memory_gb=peak_memory_gb,
        )
        
        results[num_gpus] = result
        
        # Print summary
        print(f"\nResults for {num_gpus} GPU(s):")
        print(f"  Baseline: {baseline_ms:.2f} ms")
        print(f"  Optimized: {optimized_ms:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Tokens/sec: {tokens_per_sec:.1f}")
        print(f"  Scaling efficiency: {scaling_efficiency:.1%}")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("GPU Scaling Study Summary")
    print("=" * 70)
    print(f"{'GPUs':<6} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10} {'Tok/s':<12} {'Efficiency':<10}")
    print("-" * 70)
    for num_gpus, result in sorted(results.items()):
        print(
            f"{result.num_gpus:<6} "
            f"{result.baseline_ms:<12.2f} "
            f"{result.optimized_ms:<12.2f} "
            f"{result.speedup:<10.2f}x "
            f"{result.tokens_per_sec:<12.1f} "
            f"{result.scaling_efficiency:<10.1%}"
        )
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPU Scaling Study")
    parser.add_argument("--model-size", choices=["20b", "120b"], default="20b")
    parser.add_argument("--max-gpus", type=int, help="Maximum GPUs to test")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()
    
    run_scaling_study(
        model_size=args.model_size,
        max_gpus=args.max_gpus,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


