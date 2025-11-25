"""Roofline Analysis for Ultimate MoE Inference.

Auto-detects if workload is memory or compute bound using NCU metrics
and generates roofline plots.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch


@dataclass
class RooflineMetrics:
    """Metrics for roofline analysis."""
    
    # Measured values
    achieved_flops: float  # FLOP/s
    achieved_bandwidth: float  # Bytes/s
    arithmetic_intensity: float  # FLOP/Byte
    
    # Peak values (hardware)
    peak_flops: float
    peak_bandwidth: float
    ridge_point: float  # AI where compute = memory
    
    # Classification
    bound: str  # "compute" or "memory"
    achieved_compute_pct: float
    achieved_bandwidth_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "achieved_flops": self.achieved_flops,
            "achieved_bandwidth": self.achieved_bandwidth,
            "arithmetic_intensity": self.arithmetic_intensity,
            "peak_flops": self.peak_flops,
            "peak_bandwidth": self.peak_bandwidth,
            "ridge_point": self.ridge_point,
            "bound": self.bound,
            "achieved_compute_pct": self.achieved_compute_pct,
            "achieved_bandwidth_pct": self.achieved_bandwidth_pct,
        }


# B200 specifications
B200_SPECS = {
    "peak_fp16_tflops": 2250.0,  # 2.25 PFLOPS FP8
    "peak_fp32_tflops": 1125.0,
    "peak_tensor_tflops": 4500.0,  # FP8 Tensor Core
    "peak_hbm_bandwidth_tb_s": 8.0,  # 8 TB/s HBM3e
}


class RooflineAnalyzer:
    """Analyze workload using roofline model.
    
    Uses NCU metrics to determine if a workload is memory or compute bound
    and calculates efficiency relative to hardware peaks.
    
    Example:
        analyzer = RooflineAnalyzer()
        metrics = analyzer.analyze_from_ncu("kernel.ncu-rep")
        print(f"Workload is {metrics.bound}-bound")
        print(f"Compute efficiency: {metrics.achieved_compute_pct:.1f}%")
    """
    
    def __init__(self, gpu_specs: Optional[Dict[str, float]] = None):
        """Initialize roofline analyzer.
        
        Args:
            gpu_specs: GPU specifications dict (default: B200)
        """
        self.specs = gpu_specs or B200_SPECS
        
        # Convert to base units
        self.peak_flops = self.specs["peak_tensor_tflops"] * 1e12  # FLOP/s
        self.peak_bandwidth = self.specs["peak_hbm_bandwidth_tb_s"] * 1e12  # B/s
        
        # Ridge point: where rooflines intersect
        self.ridge_point = self.peak_flops / self.peak_bandwidth
    
    def analyze_from_ncu(self, ncu_report_path: Path) -> RooflineMetrics:
        """Analyze roofline from NCU report.
        
        Args:
            ncu_report_path: Path to .ncu-rep file
            
        Returns:
            RooflineMetrics with analysis results
        """
        # Extract metrics from NCU report
        metrics = self._extract_ncu_metrics(ncu_report_path)
        
        return self._compute_roofline(metrics)
    
    def analyze_from_metrics(
        self,
        flops: float,
        dram_bytes: float,
        duration_s: float,
    ) -> RooflineMetrics:
        """Analyze roofline from raw metrics.
        
        Args:
            flops: Total floating point operations
            dram_bytes: Total DRAM bytes transferred
            duration_s: Kernel duration in seconds
            
        Returns:
            RooflineMetrics with analysis results
        """
        achieved_flops = flops / duration_s
        achieved_bandwidth = dram_bytes / duration_s
        
        metrics = {
            "achieved_flops": achieved_flops,
            "achieved_bandwidth": achieved_bandwidth,
        }
        
        return self._compute_roofline(metrics)
    
    def _extract_ncu_metrics(self, ncu_report_path: Path) -> Dict[str, float]:
        """Extract metrics from NCU report."""
        try:
            # Try using ncu to export metrics
            result = subprocess.run(
                [
                    "ncu", "--import", str(ncu_report_path),
                    "--csv", "--page", "raw",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                raise ValueError("No data in NCU report")
            
            # Look for key metrics
            metrics = {}
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 2:
                    metric_name = parts[0].strip('"')
                    try:
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except (ValueError, IndexError):
                        continue
            
            # Calculate derived metrics
            flops = metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0) * 2
            dram_read = metrics.get("dram__bytes_read.sum", 0)
            dram_write = metrics.get("dram__bytes_write.sum", 0)
            duration = metrics.get("gpu__time_duration.sum", 1) / 1e9  # ns to s
            
            return {
                "achieved_flops": flops / duration if duration > 0 else 0,
                "achieved_bandwidth": (dram_read + dram_write) / duration if duration > 0 else 0,
            }
            
        except Exception as e:
            print(f"Warning: Could not extract NCU metrics: {e}")
            # Return placeholder metrics
            return {
                "achieved_flops": 0,
                "achieved_bandwidth": 0,
            }
    
    def _compute_roofline(self, metrics: Dict[str, float]) -> RooflineMetrics:
        """Compute roofline classification from metrics."""
        achieved_flops = metrics.get("achieved_flops", 0)
        achieved_bandwidth = metrics.get("achieved_bandwidth", 1)
        
        # Arithmetic intensity
        ai = achieved_flops / achieved_bandwidth if achieved_bandwidth > 0 else 0
        
        # Classification
        if ai > self.ridge_point:
            bound = "compute"
        else:
            bound = "memory"
        
        # Efficiency
        achieved_compute_pct = 100 * achieved_flops / self.peak_flops
        achieved_bandwidth_pct = 100 * achieved_bandwidth / self.peak_bandwidth
        
        return RooflineMetrics(
            achieved_flops=achieved_flops,
            achieved_bandwidth=achieved_bandwidth,
            arithmetic_intensity=ai,
            peak_flops=self.peak_flops,
            peak_bandwidth=self.peak_bandwidth,
            ridge_point=self.ridge_point,
            bound=bound,
            achieved_compute_pct=achieved_compute_pct,
            achieved_bandwidth_pct=achieved_bandwidth_pct,
        )
    
    def print_analysis(self, metrics: RooflineMetrics) -> None:
        """Print roofline analysis summary."""
        print("\n" + "=" * 60)
        print("Roofline Analysis")
        print("=" * 60)
        print(f"\nWorkload Classification: {metrics.bound.upper()}-BOUND")
        print()
        print("Achieved Performance:")
        print(f"  FLOP/s:      {metrics.achieved_flops / 1e12:.2f} TFLOP/s")
        print(f"  Bandwidth:   {metrics.achieved_bandwidth / 1e12:.2f} TB/s")
        print(f"  Arith. Int.: {metrics.arithmetic_intensity:.2f} FLOP/Byte")
        print()
        print("Hardware Peaks (B200):")
        print(f"  Peak FLOP/s:     {metrics.peak_flops / 1e12:.2f} TFLOP/s")
        print(f"  Peak Bandwidth:  {metrics.peak_bandwidth / 1e12:.2f} TB/s")
        print(f"  Ridge Point:     {metrics.ridge_point:.2f} FLOP/Byte")
        print()
        print("Efficiency:")
        print(f"  Compute:   {metrics.achieved_compute_pct:.1f}% of peak")
        print(f"  Bandwidth: {metrics.achieved_bandwidth_pct:.1f}% of peak")
        print()
        
        if metrics.bound == "memory":
            print("Optimization Focus: Memory bandwidth")
            print("  - Increase arithmetic intensity")
            print("  - Fuse operations to reduce memory traffic")
            print("  - Use lower precision (FP8, FP4)")
        else:
            print("Optimization Focus: Compute throughput")
            print("  - Maximize Tensor Core utilization")
            print("  - Increase occupancy")
            print("  - Reduce instruction overhead")
        
        print("=" * 60)


def run_roofline_analysis(
    benchmark_name: str = "optimized_ultimate_inference",
    output_path: Optional[Path] = None,
) -> RooflineMetrics:
    """Run roofline analysis on a benchmark.
    
    Args:
        benchmark_name: Benchmark to analyze
        output_path: Path to save results
        
    Returns:
        RooflineMetrics
    """
    analyzer = RooflineAnalyzer()
    
    # Look for existing NCU report
    artifacts_dir = Path("artifacts/nsight")
    ncu_report = artifacts_dir / f"{benchmark_name}.ncu-rep"
    
    if ncu_report.exists():
        print(f"Analyzing existing NCU report: {ncu_report}")
        metrics = analyzer.analyze_from_ncu(ncu_report)
    else:
        print(f"No NCU report found. Generating placeholder analysis...")
        # Placeholder for LLM inference (typically memory-bound)
        metrics = analyzer.analyze_from_metrics(
            flops=1e15,  # 1 PFLOP
            dram_bytes=1e12,  # 1 TB
            duration_s=1.0,
        )
    
    analyzer.print_analysis(metrics)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Roofline Analysis")
    parser.add_argument("--benchmark", default="optimized_ultimate_inference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    
    run_roofline_analysis(args.benchmark, args.output)


