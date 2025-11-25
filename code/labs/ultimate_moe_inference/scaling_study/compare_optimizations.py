"""Compare optimization layers - show contribution of each chapter's techniques."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass 
class LayerResult:
    """Result for a layer configuration."""
    
    layer_name: str
    chapters: List[int]
    latency_ms: float
    speedup_vs_baseline: float
    incremental_speedup: float  # vs previous layer
    tokens_per_sec: float


def compare_optimization_layers(
    config_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, LayerResult]:
    """Run benchmark with cumulative optimization layers.
    
    Shows the contribution of each chapter's techniques by running
    with progressively more optimizations enabled.
    
    Args:
        config_path: Path to configuration file
        output_path: Path to save results JSON
        
    Returns:
        Dictionary mapping layer name to LayerResult
    """
    from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
    from optimized_ultimate_inference import OptimizedUltimateInference, OptimizedConfig
    from optimization_layers import get_layers_up_to
    
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "single_gpu.yaml"
    
    results: Dict[str, LayerResult] = {}
    baseline_ms: Optional[float] = None
    prev_ms: Optional[float] = None
    
    # Layer configurations
    layer_configs = [
        ("baseline", [], "No optimizations"),
        ("ch1-6", [1], "Basics: NVTX, NUMA, TF32"),
        ("ch7-8", [1, 2], "+ Memory: Coalescing, occupancy"),
        ("ch9-10", [1, 2, 3], "+ Pipelining: FlashAttention, buffering"),
        ("ch11-12", [1, 2, 3, 4], "+ Concurrency: Streams, graphs"),
        ("ch13-14", [1, 2, 3, 4, 5], "+ PyTorch: FP8, torch.compile"),
        ("ch15-20", [1, 2, 3, 4, 5, 6], "+ Advanced: Speculative, PagedAttn"),
    ]
    
    print("=" * 70)
    print("Layer-by-Layer Optimization Comparison")
    print("=" * 70)
    
    for layer_name, layer_nums, description in layer_configs:
        print(f"\n[{layer_name}] {description}")
        
        if layer_name == "baseline":
            # Run baseline
            baseline_config = InferenceConfig()
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    data = yaml.safe_load(f)
                baseline_config = InferenceConfig.from_yaml(config_path)
            
            try:
                benchmark = BaselineUltimateInference(baseline_config)
                harness = BenchmarkHarness(
                    mode=BenchmarkMode.CUSTOM,
                    config=benchmark.get_config(),
                )
                result = harness.benchmark(benchmark)
                latency_ms = result.timing.mean_ms if result.timing else 0.0
                tokens_per_sec = 0.0
                if hasattr(benchmark, 'last_metrics') and benchmark.last_metrics:
                    tokens_per_sec = benchmark.last_metrics.tokens_per_sec
            except Exception as e:
                print(f"  Error: {e}")
                latency_ms = 1000.0
                tokens_per_sec = 0.0
            
            baseline_ms = latency_ms
            prev_ms = latency_ms
            
            results[layer_name] = LayerResult(
                layer_name=layer_name,
                chapters=[],
                latency_ms=latency_ms,
                speedup_vs_baseline=1.0,
                incremental_speedup=1.0,
                tokens_per_sec=tokens_per_sec,
            )
        else:
            # Run with specific layers enabled
            # For now, we run the full optimized benchmark and note which layers are active
            # A more sophisticated implementation would selectively enable layers
            
            try:
                optimized_config = OptimizedConfig.from_yaml(config_path)
                
                # Disable optimizations for layers not included
                if 4 not in layer_nums:
                    optimized_config.use_cuda_graphs = False
                if 5 not in layer_nums:
                    optimized_config.use_torch_compile = False
                    optimized_config.use_fp8_kv_cache = False
                if 6 not in layer_nums:
                    optimized_config.use_speculative_decode = False
                    optimized_config.use_paged_attention = False
                if 3 not in layer_nums:
                    optimized_config.use_flash_attention = False
                
                benchmark = OptimizedUltimateInference(optimized_config)
                harness = BenchmarkHarness(
                    mode=BenchmarkMode.CUSTOM,
                    config=benchmark.get_config(),
                )
                result = harness.benchmark(benchmark)
                latency_ms = result.timing.mean_ms if result.timing else 0.0
                tokens_per_sec = 0.0
                if hasattr(benchmark, 'last_metrics') and benchmark.last_metrics:
                    tokens_per_sec = benchmark.last_metrics.tokens_per_sec
            except Exception as e:
                print(f"  Error: {e}")
                latency_ms = prev_ms or 1000.0
                tokens_per_sec = 0.0
            
            # Calculate speedups
            speedup_vs_baseline = baseline_ms / latency_ms if latency_ms > 0 else 0.0
            incremental_speedup = prev_ms / latency_ms if latency_ms > 0 and prev_ms else 1.0
            
            # Map layer nums to chapter ranges
            chapter_map = {
                1: [1, 2, 3, 4, 5, 6],
                2: [7, 8],
                3: [9, 10],
                4: [11, 12],
                5: [13, 14],
                6: [15, 16, 17, 18, 19, 20],
            }
            chapters = []
            for ln in layer_nums:
                chapters.extend(chapter_map.get(ln, []))
            
            results[layer_name] = LayerResult(
                layer_name=layer_name,
                chapters=chapters,
                latency_ms=latency_ms,
                speedup_vs_baseline=speedup_vs_baseline,
                incremental_speedup=incremental_speedup,
                tokens_per_sec=tokens_per_sec,
            )
            
            prev_ms = latency_ms
        
        # Print result
        r = results[layer_name]
        print(f"  Latency: {r.latency_ms:.2f} ms")
        print(f"  Speedup vs baseline: {r.speedup_vs_baseline:.2f}x")
        print(f"  Incremental speedup: {r.incremental_speedup:.2f}x")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: Cumulative Optimization Impact")
    print("=" * 70)
    print(f"{'Layer':<12} {'Latency (ms)':<15} {'vs Baseline':<12} {'Incremental':<12}")
    print("-" * 70)
    for name, result in results.items():
        print(
            f"{result.layer_name:<12} "
            f"{result.latency_ms:<15.2f} "
            f"{result.speedup_vs_baseline:<12.2f}x "
            f"{result.incremental_speedup:<12.2f}x"
        )
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Layer-by-Layer Optimization Comparison")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()
    
    compare_optimization_layers(
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


