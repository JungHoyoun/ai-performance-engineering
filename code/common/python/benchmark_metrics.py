"""Benchmark metrics helpers for domain-specific performance analysis.

This module provides easy-to-use helpers for computing performance metrics
that help understand WHY optimizations work and HOW to improve performance.

Each chapter has specific metrics that matter. Use these helpers in your
benchmark's get_custom_metrics() method.

Usage:
    from common.python.benchmark_metrics import (
        compute_memory_metrics,
        compute_compute_metrics,
        compute_stream_metrics,
        ...
    )
    
    def get_custom_metrics(self) -> Optional[dict]:
        return compute_memory_metrics(
            bytes_transferred=self.N * 4,
            elapsed_ms=self._last_elapsed_ms,
        )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any


# =============================================================================
# Hardware Specifications (Blackwell B200 defaults)
# =============================================================================

@dataclass(frozen=True)
class HardwareSpecs:
    """Hardware specifications for computing theoretical peaks."""
    name: str
    hbm_bandwidth_gbps: float      # HBM3e bandwidth in GB/s
    pcie_bandwidth_gbps: float     # PCIe Gen5 x16 bandwidth in GB/s
    nvlink_bandwidth_gbps: float   # NVLink per-link bandwidth in GB/s
    fp32_tflops: float             # Peak FP32 TFLOPS
    fp16_tflops: float             # Peak FP16 TFLOPS
    fp8_tflops: float              # Peak FP8 TFLOPS
    tensor_tflops: float           # Peak Tensor Core TFLOPS (FP16)
    num_sms: int                   # Number of Streaming Multiprocessors
    shared_mem_per_sm_kb: float    # Shared memory per SM in KB


# Common hardware profiles
BLACKWELL_B200 = HardwareSpecs(
    name="NVIDIA B200",
    hbm_bandwidth_gbps=8000.0,     # ~8 TB/s HBM3e
    pcie_bandwidth_gbps=64.0,      # PCIe Gen5 x16
    nvlink_bandwidth_gbps=900.0,   # NVLink5 per link
    fp32_tflops=80.0,              # Non-tensor FP32
    fp16_tflops=160.0,             # Non-tensor FP16
    fp8_tflops=2500.0,             # FP8 Tensor Core sparse
    tensor_tflops=1250.0,          # FP16 Tensor Core
    num_sms=148,
    shared_mem_per_sm_kb=228.0,
)

HOPPER_H100 = HardwareSpecs(
    name="NVIDIA H100",
    hbm_bandwidth_gbps=3350.0,
    pcie_bandwidth_gbps=64.0,
    nvlink_bandwidth_gbps=450.0,
    fp32_tflops=67.0,
    fp16_tflops=134.0,
    fp8_tflops=1979.0,
    tensor_tflops=989.0,
    num_sms=132,
    shared_mem_per_sm_kb=228.0,
)

# Default to Blackwell
DEFAULT_SPECS = BLACKWELL_B200


def detect_hardware_specs() -> HardwareSpecs:
    """Detect current hardware and return appropriate specs."""
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 10:  # Blackwell
                return BLACKWELL_B200
            elif props.major == 9:  # Hopper
                return HOPPER_H100
    return DEFAULT_SPECS


# =============================================================================
# Chapter 2: Memory Transfer Metrics
# =============================================================================

def compute_memory_transfer_metrics(
    bytes_transferred: float,
    elapsed_ms: float,
    transfer_type: str = "pcie",  # "pcie", "nvlink", "hbm"
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute metrics for memory transfer benchmarks (ch2).
    
    Args:
        bytes_transferred: Number of bytes moved
        elapsed_ms: Time elapsed in milliseconds
        transfer_type: Type of transfer ("pcie", "nvlink", "hbm")
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with bandwidth metrics and efficiency percentages
    """
    specs = specs or detect_hardware_specs()
    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    achieved_gbps = (bytes_transferred / 1e9) / elapsed_s
    
    # Get theoretical peak for transfer type
    peak_map = {
        "pcie": specs.pcie_bandwidth_gbps,
        "nvlink": specs.nvlink_bandwidth_gbps,
        "hbm": specs.hbm_bandwidth_gbps,
    }
    theoretical_peak = peak_map.get(transfer_type, specs.pcie_bandwidth_gbps)
    efficiency = (achieved_gbps / theoretical_peak) * 100.0 if theoretical_peak > 0 else 0.0
    
    return {
        "transfer.bytes": bytes_transferred,
        "transfer.achieved_gbps": achieved_gbps,
        "transfer.theoretical_peak_gbps": theoretical_peak,
        "transfer.efficiency_pct": min(efficiency, 100.0),
        "transfer.type": 0.0 if transfer_type == "pcie" else (1.0 if transfer_type == "nvlink" else 2.0),
    }


# =============================================================================
# Chapter 6: Kernel Fundamentals Metrics
# =============================================================================

def compute_kernel_fundamentals_metrics(
    num_elements: int,
    num_iterations: int = 1,
    expected_bank_conflicts_per_warp: float = 0.0,
    expected_divergent_branches: float = 0.0,
) -> Dict[str, float]:
    """Compute metrics for kernel fundamentals benchmarks (ch6).
    
    Args:
        num_elements: Number of elements processed
        num_iterations: Number of kernel iterations
        expected_bank_conflicts_per_warp: Expected bank conflicts (0 = none, 32 = worst)
        expected_divergent_branches: Expected divergent branches per warp
    
    Returns:
        Dict with kernel characteristic metrics
    """
    return {
        "kernel.elements": float(num_elements),
        "kernel.iterations": float(num_iterations),
        "kernel.expected_bank_conflicts_per_warp": expected_bank_conflicts_per_warp,
        "kernel.expected_divergent_branches": expected_divergent_branches,
        # 32-way bank conflict = worst case
        "kernel.bank_conflict_severity": expected_bank_conflicts_per_warp / 32.0,
    }


# =============================================================================
# Chapter 7: Memory Access Pattern Metrics
# =============================================================================

def compute_memory_access_metrics(
    bytes_requested: float,
    bytes_actually_transferred: float,
    num_transactions: int,
    optimal_transactions: int,
) -> Dict[str, float]:
    """Compute metrics for memory access pattern benchmarks (ch7).
    
    Args:
        bytes_requested: Bytes actually needed by the kernel
        bytes_actually_transferred: Bytes moved over the bus (includes waste)
        num_transactions: Actual memory transactions issued
        optimal_transactions: Theoretical minimum transactions needed
    
    Returns:
        Dict with coalescing and efficiency metrics
    """
    efficiency = (bytes_requested / bytes_actually_transferred) * 100.0 if bytes_actually_transferred > 0 else 0.0
    transaction_efficiency = (optimal_transactions / num_transactions) * 100.0 if num_transactions > 0 else 0.0
    
    return {
        "memory.bytes_requested": bytes_requested,
        "memory.bytes_transferred": bytes_actually_transferred,
        "memory.efficiency_pct": min(efficiency, 100.0),
        "memory.transactions_actual": float(num_transactions),
        "memory.transactions_optimal": float(optimal_transactions),
        "memory.transaction_efficiency_pct": min(transaction_efficiency, 100.0),
        # Coalescing: 100% = perfect, <50% = severe coalescing issues
        "memory.coalescing_quality": min(efficiency, 100.0),
    }


# =============================================================================
# Chapter 8: Optimization Technique Metrics
# =============================================================================

def compute_optimization_metrics(
    baseline_ms: float,
    optimized_ms: float,
    technique: str,
    registers_per_thread: int = 0,
    shared_mem_bytes: int = 0,
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute metrics for optimization technique benchmarks (ch8).
    
    Args:
        baseline_ms: Baseline execution time
        optimized_ms: Optimized execution time
        technique: Name of optimization technique
        registers_per_thread: Registers used per thread
        shared_mem_bytes: Shared memory used per block
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with optimization effectiveness metrics
    """
    specs = specs or detect_hardware_specs()
    speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
    improvement_pct = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else 0.0
    
    # Estimate occupancy impact
    max_regs_per_sm = 65536  # Typical for modern GPUs
    max_shared_per_sm = specs.shared_mem_per_sm_kb * 1024
    
    # Simple occupancy estimate (actual calculation is more complex)
    reg_limited_blocks = max_regs_per_sm // (registers_per_thread * 256) if registers_per_thread > 0 else 32
    smem_limited_blocks = int(max_shared_per_sm / shared_mem_bytes) if shared_mem_bytes > 0 else 32
    estimated_blocks_per_sm = min(reg_limited_blocks, smem_limited_blocks, 32)
    
    return {
        "optimization.baseline_ms": baseline_ms,
        "optimization.optimized_ms": optimized_ms,
        "optimization.speedup": speedup,
        "optimization.improvement_pct": improvement_pct,
        "optimization.registers_per_thread": float(registers_per_thread),
        "optimization.shared_mem_bytes": float(shared_mem_bytes),
        "optimization.estimated_blocks_per_sm": float(estimated_blocks_per_sm),
    }


# =============================================================================
# Chapter 9: Compute-Bound Metrics (Roofline)
# =============================================================================

def compute_roofline_metrics(
    total_flops: float,
    total_bytes: float,
    elapsed_ms: float,
    precision: str = "fp16",  # "fp32", "fp16", "fp8", "tensor"
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute roofline analysis metrics for compute-bound benchmarks (ch9).
    
    Args:
        total_flops: Total floating-point operations
        total_bytes: Total bytes moved to/from memory
        elapsed_ms: Execution time in milliseconds
        precision: Precision type for peak calculation
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with roofline position and efficiency metrics
    """
    specs = specs or detect_hardware_specs()
    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    
    # Achieved performance
    achieved_tflops = (total_flops / 1e12) / elapsed_s
    achieved_gbps = (total_bytes / 1e9) / elapsed_s
    
    # Arithmetic intensity (FLOPS per byte)
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0
    
    # Peak performance for precision type
    peak_map = {
        "fp32": specs.fp32_tflops,
        "fp16": specs.fp16_tflops,
        "fp8": specs.fp8_tflops,
        "tensor": specs.tensor_tflops,
    }
    peak_tflops = peak_map.get(precision, specs.fp16_tflops)
    
    # Ridge point: where memory and compute rooflines meet
    # ridge_point = peak_tflops / (memory_bandwidth_TB_per_s)
    ridge_point = (peak_tflops * 1000.0) / specs.hbm_bandwidth_gbps
    
    # Memory-bound performance ceiling at this arithmetic intensity
    memory_ceiling_tflops = (achieved_gbps / 1000.0) * arithmetic_intensity
    
    # Classification
    is_compute_bound = arithmetic_intensity > ridge_point
    
    # Efficiency relative to the appropriate ceiling
    if is_compute_bound:
        efficiency = (achieved_tflops / peak_tflops) * 100.0
    else:
        efficiency = (achieved_gbps / specs.hbm_bandwidth_gbps) * 100.0
    
    return {
        "roofline.achieved_tflops": achieved_tflops,
        "roofline.achieved_gbps": achieved_gbps,
        "roofline.arithmetic_intensity": arithmetic_intensity,
        "roofline.ridge_point": ridge_point,
        "roofline.peak_tflops": peak_tflops,
        "roofline.peak_gbps": specs.hbm_bandwidth_gbps,
        "roofline.memory_ceiling_tflops": memory_ceiling_tflops,
        "roofline.is_compute_bound": 1.0 if is_compute_bound else 0.0,
        "roofline.efficiency_pct": min(efficiency, 100.0),
    }


# =============================================================================
# Chapter 11: CUDA Stream Metrics
# =============================================================================

def compute_stream_metrics(
    sequential_time_ms: float,
    overlapped_time_ms: float,
    num_streams: int,
    num_operations: int,
) -> Dict[str, float]:
    """Compute metrics for CUDA stream benchmarks (ch11).
    
    Args:
        sequential_time_ms: Time when running sequentially
        overlapped_time_ms: Time with stream overlap
        num_streams: Number of CUDA streams used
        num_operations: Number of independent operations
    
    Returns:
        Dict with stream overlap efficiency metrics
    """
    # Overlap efficiency: how much time was saved
    time_saved_ms = sequential_time_ms - overlapped_time_ms
    overlap_efficiency = (time_saved_ms / sequential_time_ms) * 100.0 if sequential_time_ms > 0 else 0.0
    
    # Theoretical max speedup with perfect overlap
    theoretical_speedup = num_operations  # Perfect parallelism
    actual_speedup = sequential_time_ms / overlapped_time_ms if overlapped_time_ms > 0 else 1.0
    parallelism_efficiency = (actual_speedup / theoretical_speedup) * 100.0 if theoretical_speedup > 0 else 0.0
    
    return {
        "stream.sequential_ms": sequential_time_ms,
        "stream.overlapped_ms": overlapped_time_ms,
        "stream.time_saved_ms": time_saved_ms,
        "stream.overlap_efficiency_pct": overlap_efficiency,
        "stream.num_streams": float(num_streams),
        "stream.num_operations": float(num_operations),
        "stream.theoretical_speedup": float(theoretical_speedup),
        "stream.actual_speedup": actual_speedup,
        "stream.parallelism_efficiency_pct": min(parallelism_efficiency, 100.0),
    }


# =============================================================================
# Chapter 12: CUDA Graph Metrics
# =============================================================================

def compute_graph_metrics(
    baseline_launch_overhead_us: float,
    graph_launch_overhead_us: float,
    num_nodes: int,
    num_iterations: int,
) -> Dict[str, float]:
    """Compute metrics for CUDA graph benchmarks (ch12).
    
    Args:
        baseline_launch_overhead_us: Per-launch overhead without graphs (microseconds)
        graph_launch_overhead_us: Per-launch overhead with graphs (microseconds)
        num_nodes: Number of nodes in the graph
        num_iterations: Number of graph replays
    
    Returns:
        Dict with graph optimization metrics
    """
    overhead_reduction_us = baseline_launch_overhead_us - graph_launch_overhead_us
    overhead_reduction_pct = (overhead_reduction_us / baseline_launch_overhead_us) * 100.0 if baseline_launch_overhead_us > 0 else 0.0
    total_overhead_saved_us = overhead_reduction_us * num_iterations
    
    return {
        "graph.baseline_launch_us": baseline_launch_overhead_us,
        "graph.graph_launch_us": graph_launch_overhead_us,
        "graph.overhead_reduction_us": overhead_reduction_us,
        "graph.overhead_reduction_pct": overhead_reduction_pct,
        "graph.num_nodes": float(num_nodes),
        "graph.num_iterations": float(num_iterations),
        "graph.total_overhead_saved_us": total_overhead_saved_us,
    }


# =============================================================================
# Chapter 13+: Precision and Training Metrics
# =============================================================================

def compute_precision_metrics(
    fp32_time_ms: float,
    reduced_precision_time_ms: float,
    precision_type: str,  # "fp16", "bf16", "fp8", "fp4"
    accuracy_delta: float = 0.0,  # Accuracy loss (if measured)
) -> Dict[str, float]:
    """Compute metrics for precision optimization benchmarks (ch13, ch19).
    
    Args:
        fp32_time_ms: Baseline FP32 execution time
        reduced_precision_time_ms: Reduced precision execution time
        precision_type: Type of reduced precision used
        accuracy_delta: Change in accuracy (negative = loss)
    
    Returns:
        Dict with precision tradeoff metrics
    """
    speedup = fp32_time_ms / reduced_precision_time_ms if reduced_precision_time_ms > 0 else 0.0
    
    # Memory reduction factors
    memory_reduction = {
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 4.0,
        "fp4": 8.0,
    }
    reduction_factor = memory_reduction.get(precision_type, 1.0)
    
    # Theoretical speedup based on memory bandwidth
    theoretical_speedup = reduction_factor  # Simplified: assumes memory-bound
    speedup_efficiency = (speedup / theoretical_speedup) * 100.0 if theoretical_speedup > 0 else 0.0
    
    return {
        "precision.fp32_ms": fp32_time_ms,
        "precision.reduced_ms": reduced_precision_time_ms,
        "precision.speedup": speedup,
        "precision.memory_reduction_factor": reduction_factor,
        "precision.theoretical_speedup": theoretical_speedup,
        "precision.speedup_efficiency_pct": speedup_efficiency,
        "precision.accuracy_delta": accuracy_delta,
    }


# =============================================================================
# Chapter 15+: Inference Metrics
# =============================================================================

def compute_inference_metrics(
    ttft_ms: float,
    tpot_ms: float,
    total_tokens: int,
    total_requests: int,
    batch_size: int,
    max_batch_size: int,
) -> Dict[str, float]:
    """Compute metrics for inference benchmarks (ch15-ch18).
    
    Args:
        ttft_ms: Time to first token (milliseconds)
        tpot_ms: Time per output token (milliseconds)
        total_tokens: Total tokens generated
        total_requests: Total requests processed
        batch_size: Actual batch size used
        max_batch_size: Maximum supported batch size
    
    Returns:
        Dict with inference performance metrics
    """
    tokens_per_second = (total_tokens / (ttft_ms + tpot_ms * total_tokens)) * 1000.0 if total_tokens > 0 else 0.0
    batch_utilization = (batch_size / max_batch_size) * 100.0 if max_batch_size > 0 else 0.0
    
    return {
        "inference.ttft_ms": ttft_ms,
        "inference.tpot_ms": tpot_ms,
        "inference.total_tokens": float(total_tokens),
        "inference.total_requests": float(total_requests),
        "inference.tokens_per_second": tokens_per_second,
        "inference.batch_size": float(batch_size),
        "inference.max_batch_size": float(max_batch_size),
        "inference.batch_utilization_pct": batch_utilization,
    }


# =============================================================================
# Chapter 18: Speculative Decoding Metrics
# =============================================================================

def compute_speculative_decoding_metrics(
    draft_tokens: int,
    accepted_tokens: int,
    draft_time_ms: float,
    verify_time_ms: float,
    num_rounds: int,
) -> Dict[str, float]:
    """Compute metrics for speculative decoding benchmarks (ch18).
    
    Args:
        draft_tokens: Total draft tokens generated
        accepted_tokens: Total tokens accepted after verification
        draft_time_ms: Total time spent in draft phase
        verify_time_ms: Total time spent in verification phase
        num_rounds: Number of draft-verify rounds
    
    Returns:
        Dict with speculative decoding efficiency metrics
    """
    acceptance_rate = (accepted_tokens / draft_tokens) * 100.0 if draft_tokens > 0 else 0.0
    avg_accepted_per_round = accepted_tokens / num_rounds if num_rounds > 0 else 0.0
    draft_verify_ratio = draft_time_ms / verify_time_ms if verify_time_ms > 0 else 0.0
    
    # Tokens wasted on rejected drafts
    rejected_tokens = draft_tokens - accepted_tokens
    waste_pct = (rejected_tokens / draft_tokens) * 100.0 if draft_tokens > 0 else 0.0
    
    return {
        "speculative.draft_tokens": float(draft_tokens),
        "speculative.accepted_tokens": float(accepted_tokens),
        "speculative.rejected_tokens": float(rejected_tokens),
        "speculative.acceptance_rate_pct": acceptance_rate,
        "speculative.waste_pct": waste_pct,
        "speculative.avg_accepted_per_round": avg_accepted_per_round,
        "speculative.draft_time_ms": draft_time_ms,
        "speculative.verify_time_ms": verify_time_ms,
        "speculative.draft_verify_ratio": draft_verify_ratio,
        "speculative.num_rounds": float(num_rounds),
    }


# =============================================================================
# Generic Helper
# =============================================================================

def compute_speedup_metrics(
    baseline_ms: float,
    optimized_ms: float,
    name: str = "",
) -> Dict[str, float]:
    """Compute basic speedup metrics between baseline and optimized.
    
    Args:
        baseline_ms: Baseline execution time
        optimized_ms: Optimized execution time
        name: Optional name prefix for metrics
    
    Returns:
        Dict with speedup and improvement metrics
    """
    prefix = f"{name}." if name else ""
    speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
    improvement_pct = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else 0.0
    
    return {
        f"{prefix}baseline_ms": baseline_ms,
        f"{prefix}optimized_ms": optimized_ms,
        f"{prefix}speedup": speedup,
        f"{prefix}improvement_pct": improvement_pct,
    }


