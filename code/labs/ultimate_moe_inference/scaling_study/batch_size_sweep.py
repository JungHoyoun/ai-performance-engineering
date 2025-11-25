"""Batch Size Sweep - Find optimal throughput vs latency tradeoffs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BatchSizeResult:
    """Result for a batch size configuration."""
    
    batch_size: int
    latency_ms: float
    tokens_per_sec: float
    memory_gb: float
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None


def run_batch_size_sweep(
    config_path: Optional[Path] = None,
    batch_sizes: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> Dict[int, BatchSizeResult]:
    """Sweep batch sizes to find throughput vs latency tradeoffs.
    
    Args:
        config_path: Path to base configuration file
        batch_sizes: List of batch sizes to test
        output_path: Path to save results JSON
        
    Returns:
        Dictionary mapping batch size to BatchSizeResult
    """
    from optimized_ultimate_inference import OptimizedUltimateInference, OptimizedConfig
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    import torch
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "single_gpu.yaml"
    
    results: Dict[int, BatchSizeResult] = {}
    
    print("=" * 70)
    print("Batch Size Sweep")
    print("=" * 70)
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        try:
            # Load and modify config
            config = OptimizedConfig.from_yaml(config_path)
            config.batch_size = batch_size
            
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
            ttft_ms = metrics.ttft_ms if metrics else None
            tpot_ms = metrics.tpot_ms if metrics else None
            
            results[batch_size] = BatchSizeResult(
                batch_size=batch_size,
                latency_ms=latency_ms,
                tokens_per_sec=tokens_per_sec,
                memory_gb=memory_gb,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
            )
            
            print(f"  Latency: {latency_ms:.2f} ms")
            print(f"  Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"  Memory: {memory_gb:.2f} GB")
            
            # Clean up
            del benchmark
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch size {batch_size}")
                break
            raise
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Batch Size Sweep Summary")
    print("=" * 70)
    print(f"{'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-" * 70)
    for bs, r in sorted(results.items()):
        print(f"{r.batch_size:<8} {r.latency_ms:<15.2f} {r.tokens_per_sec:<15.1f} {r.memory_gb:<12.2f}")
    print("=" * 70)
    
    # Find optimal configurations
    if results:
        best_latency = min(results.values(), key=lambda x: x.latency_ms)
        best_throughput = max(results.values(), key=lambda x: x.tokens_per_sec)
        
        print(f"\nBest for latency: batch_size={best_latency.batch_size} ({best_latency.latency_ms:.2f} ms)")
        print(f"Best for throughput: batch_size={best_throughput.batch_size} ({best_throughput.tokens_per_sec:.1f} tok/s)")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch Size Sweep")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()
    
    run_batch_size_sweep(
        config_path=args.config,
        batch_sizes=args.batch_sizes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


