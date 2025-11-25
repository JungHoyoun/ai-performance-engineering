"""Precision Sweep - Compare MXFP4 vs FP8 vs BF16."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PrecisionResult:
    """Result for a precision configuration."""
    
    precision: str
    latency_ms: float
    tokens_per_sec: float
    memory_gb: float
    
    # Quality metrics (optional)
    perplexity_delta: Optional[float] = None


def run_precision_sweep(
    config_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, PrecisionResult]:
    """Compare precision modes on same workload.
    
    Tests: MXFP4 (native), FP8 (Transformer Engine), BF16 (baseline)
    
    Args:
        config_path: Path to base configuration file
        output_path: Path to save results JSON
        
    Returns:
        Dictionary mapping precision to PrecisionResult
    """
    from optimized_ultimate_inference import OptimizedUltimateInference, OptimizedConfig
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    import torch
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "single_gpu.yaml"
    
    precisions = ["mxfp4", "fp8", "bf16"]
    results: Dict[str, PrecisionResult] = {}
    
    print("=" * 70)
    print("Precision Comparison Sweep")
    print("=" * 70)
    
    for precision in precisions:
        print(f"\nTesting precision: {precision}")
        
        try:
            # Load and modify config
            config = OptimizedConfig.from_yaml(config_path)
            config.precision = precision
            
            # Adjust FP8 settings
            if precision == "fp8":
                config.use_fp8_kv_cache = True
            else:
                config.use_fp8_kv_cache = False
            
            # Run benchmark
            benchmark = OptimizedUltimateInference(config)
            harness = BenchmarkHarness(
                mode=BenchmarkMode.CUSTOM,
                config=benchmark.get_config(),
            )
            result = harness.benchmark(benchmark)
            
            latency_ms = result.timing.mean_ms if result.timing else 0.0
            
            # Get detailed metrics
            metrics = getattr(benchmark, 'last_metrics', None)
            tokens_per_sec = metrics.tokens_per_sec if metrics else 0.0
            memory_gb = metrics.peak_memory_gb if metrics else 0.0
            
            results[precision] = PrecisionResult(
                precision=precision,
                latency_ms=latency_ms,
                tokens_per_sec=tokens_per_sec,
                memory_gb=memory_gb,
            )
            
            print(f"  Latency: {latency_ms:.2f} ms")
            print(f"  Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"  Memory: {memory_gb:.2f} GB")
            
            # Clean up
            del benchmark
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error with {precision}: {e}")
            results[precision] = PrecisionResult(
                precision=precision,
                latency_ms=0.0,
                tokens_per_sec=0.0,
                memory_gb=0.0,
            )
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Precision Comparison Summary")
    print("=" * 70)
    print(f"{'Precision':<12} {'Latency (ms)':<15} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-" * 70)
    for precision, r in results.items():
        print(f"{r.precision:<12} {r.latency_ms:<15.2f} {r.tokens_per_sec:<15.1f} {r.memory_gb:<12.2f}")
    print("=" * 70)
    
    # Calculate relative performance
    if "bf16" in results and results["bf16"].tokens_per_sec > 0:
        bf16_tps = results["bf16"].tokens_per_sec
        print("\nRelative to BF16:")
        for precision, r in results.items():
            if r.tokens_per_sec > 0:
                ratio = r.tokens_per_sec / bf16_tps
                print(f"  {precision}: {ratio:.2f}x throughput")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Precision Sweep")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()
    
    run_precision_sweep(
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


