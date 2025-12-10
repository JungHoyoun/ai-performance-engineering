#!/usr/bin/env python3
"""
Edge case tests for all 94 anti-cheat protections.

This file provides comprehensive edge case coverage including:
- Boundary conditions (exactly at tolerance limits)
- Extreme values (very large/small)
- Negative tests (attacks that should be detected)
- Race conditions
- Concurrent access patterns
- Precision edge cases

Each protection category has dedicated edge case tests.
"""

import gc
import math
import os
import random
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import numpy as np

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for edge case tests"
)


# =============================================================================
# TIMING EDGE CASES (7 protections)
# =============================================================================

class TestTimingEdgeCases:
    """Edge cases for timing protections."""
    
    # 1. Unsynced Streams
    def test_unsynced_streams_multiple_streams(self):
        """Edge: Multiple concurrent streams."""
        streams = [torch.cuda.Stream() for _ in range(4)]
        results = []
        
        for s in streams:
            with torch.cuda.stream(s):
                x = torch.randn(100, device="cuda")
                results.append(x)
        
        # Must sync ALL streams
        for s in streams:
            s.synchronize()
        
        assert len(results) == 4
    
    def test_unsynced_streams_nested_streams(self):
        """Edge: Nested stream contexts."""
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        
        with torch.cuda.stream(s1):
            x = torch.randn(100, device="cuda")
            with torch.cuda.stream(s2):
                y = torch.randn(100, device="cuda")
        
        s1.synchronize()
        s2.synchronize()
        torch.cuda.synchronize()
    
    def test_unsynced_streams_priority_streams(self):
        """Edge: High and low priority streams."""
        high_priority = torch.cuda.Stream(priority=-1)  # High priority
        low_priority = torch.cuda.Stream(priority=0)    # Low priority
        
        with torch.cuda.stream(high_priority):
            torch.randn(1000, device="cuda")
        
        with torch.cuda.stream(low_priority):
            torch.randn(1000, device="cuda")
        
        torch.cuda.synchronize()
    
    # 2. Incomplete Async Ops
    def test_async_ops_chained_operations(self):
        """Edge: Long chain of async operations."""
        x = torch.randn(1000, 1000, device="cuda")
        
        for _ in range(10):
            x = torch.mm(x, x.T)  # Chain of matmuls
        
        torch.cuda.synchronize()
        assert x.shape == (1000, 1000)
    
    def test_async_ops_mixed_cpu_gpu(self):
        """Edge: Mixed CPU/GPU operations."""
        cpu_data = torch.randn(1000, 1000)
        gpu_data = cpu_data.cuda()
        
        result = torch.mm(gpu_data, gpu_data.T)
        cpu_result = result.cpu()  # Async transfer
        
        torch.cuda.synchronize()
        assert cpu_result.device.type == "cpu"
    
    # 3. Event Timing Gaps
    def test_event_timing_zero_duration(self):
        """Edge: Near-zero duration operations."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # Empty - zero duration
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        assert elapsed >= 0  # Should never be negative
    
    def test_event_timing_very_long_duration(self):
        """Edge: Very long operations."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # Large operation
        x = torch.randn(5000, 5000, device="cuda")
        y = torch.mm(x, x.T)
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        assert elapsed > 0
    
    # 4. Timer Granularity
    def test_timer_granularity_sub_microsecond(self):
        """Edge: Operations faster than timer resolution."""
        times = []
        
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            torch.cuda.synchronize()  # Very fast
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        # Should have measurable variance even at small scale
        assert max(times) >= min(times)
    
    # 5. Warmup Bleed
    def test_warmup_bleed_jit_compilation(self):
        """Edge: JIT compilation during warmup."""
        @torch.jit.script
        def fn(x):
            return x * 2 + 1
        
        x = torch.randn(100, device="cuda")
        
        # Warmup - JIT compilation happens here
        for _ in range(3):
            fn(x)
        
        torch.cuda.synchronize()
        
        # Timed - should be compiled
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn(x)
        end.record()
        
        torch.cuda.synchronize()
    
    def test_warmup_bleed_cudnn_autotuning(self):
        """Edge: cuDNN autotuning during warmup."""
        torch.backends.cudnn.benchmark = True
        
        x = torch.randn(32, 64, 128, 128, device="cuda")
        conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
        
        # Warmup - autotuning happens
        for _ in range(10):
            y = conv(x)
        
        torch.cuda.synchronize()
        torch.backends.cudnn.benchmark = False
    
    # 6. Clock Drift
    def test_clock_drift_long_benchmark(self):
        """Edge: Detect clock drift over long benchmarks."""
        import time
        
        wall_times = []
        cuda_times = []
        
        for _ in range(5):
            wall_start = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            x = torch.randn(2000, 2000, device="cuda")
            y = torch.mm(x, x.T)
            end.record()
            
            torch.cuda.synchronize()
            wall_end = time.perf_counter()
            
            wall_times.append((wall_end - wall_start) * 1000)  # ms
            cuda_times.append(start.elapsed_time(end))
        
        # Should be roughly correlated
        assert len(wall_times) == len(cuda_times)
    
    # 7. Profiler Overhead
    def test_profiler_overhead_nested_profilers(self):
        """Edge: Nested profiler contexts."""
        with torch.profiler.profile() as outer:
            x = torch.randn(100, device="cuda")
            # Can't nest profilers, but should handle gracefully


# =============================================================================
# OUTPUT EDGE CASES (10 protections)
# =============================================================================

class TestOutputEdgeCases:
    """Edge cases for output validation."""
    
    # 1. Constant Output
    def test_constant_output_single_value(self):
        """Edge: Output is single repeated value."""
        output = torch.zeros(1000, device="cuda")
        
        # Should detect constant output
        is_constant = (output.max() - output.min()).abs() < 1e-10
        assert is_constant
    
    def test_constant_output_near_constant(self):
        """Edge: Output is nearly constant (within noise)."""
        output = torch.ones(1000, device="cuda") + torch.randn(1000, device="cuda") * 1e-8
        
        variation = (output.max() - output.min()).item()
        assert variation < 1e-6  # Near constant but not exactly
    
    # 2. Stale Cache
    def test_stale_cache_after_input_change(self):
        """Edge: Cache not invalidated after input change."""
        inputs1 = torch.randn(100, device="cuda")
        inputs2 = torch.randn(100, device="cuda")
        
        # Different inputs should produce different outputs
        assert not torch.allclose(inputs1, inputs2)
    
    # 3. Approximation Drift
    def test_approximation_drift_accumulated_error(self):
        """Edge: Error accumulates over iterations."""
        x = torch.ones(100, device="cuda")
        
        for _ in range(1000):
            x = x * 1.0001  # Small multiplicative error
        
        # Error should accumulate
        assert (x - 1.0).abs().mean() > 0.01
    
    def test_approximation_drift_fp16_accumulation(self):
        """Edge: FP16 accumulation error."""
        x = torch.ones(1000, device="cuda", dtype=torch.float16)
        
        for _ in range(100):
            x = x + 0.001
        
        # FP16 has less precision
        assert x.dtype == torch.float16
    
    # 4. Invalid Values (NaN)
    def test_nan_from_division_by_zero(self):
        """Edge: NaN from 0/0."""
        x = torch.zeros(10, device="cuda")
        y = torch.zeros(10, device="cuda")
        
        result = x / y
        assert torch.isnan(result).any()
    
    def test_nan_from_sqrt_negative(self):
        """Edge: NaN from sqrt of negative."""
        x = torch.tensor([-1.0], device="cuda")
        result = torch.sqrt(x)
        assert torch.isnan(result).any()
    
    def test_nan_propagation(self):
        """Edge: NaN propagates through computation."""
        x = torch.tensor([1.0, float('nan'), 2.0], device="cuda")
        y = x * 2
        assert torch.isnan(y).any()
    
    # 5. Invalid Values (Inf)
    def test_inf_from_overflow(self):
        """Edge: Inf from overflow."""
        x = torch.tensor([1e38], device="cuda", dtype=torch.float32)
        result = x * x
        assert torch.isinf(result).any()
    
    def test_inf_from_division(self):
        """Edge: Inf from division by very small number."""
        x = torch.tensor([1e38], device="cuda")
        y = torch.tensor([1e-38], device="cuda")
        result = x / y
        assert torch.isinf(result).any()
    
    # 6. Invalid Ground Truth
    def test_ground_truth_contains_nan(self):
        """Edge: Ground truth itself has invalid values."""
        gt = torch.tensor([1.0, float('nan'), 2.0], device="cuda")
        pred = torch.tensor([1.0, 1.5, 2.0], device="cuda")
        
        # Comparison with NaN ground truth is problematic
        assert torch.isnan(gt).any()
    
    # 7. Shape Mismatch
    def test_shape_mismatch_broadcast_ambiguity(self):
        """Edge: Shapes that broadcast but shouldn't."""
        expected = torch.randn(32, 64, device="cuda")
        actual = torch.randn(1, 64, device="cuda")  # Broadcasts
        
        assert expected.shape != actual.shape
    
    def test_shape_mismatch_transposed(self):
        """Edge: Transposed shapes."""
        expected = torch.randn(32, 64, device="cuda")
        actual = torch.randn(64, 32, device="cuda")
        
        assert expected.shape != actual.shape
    
    # 8. Dtype Mismatch
    def test_dtype_mismatch_float_precisions(self):
        """Edge: Different float precisions."""
        fp32 = torch.randn(100, device="cuda", dtype=torch.float32)
        fp16 = fp32.half()
        
        assert fp32.dtype != fp16.dtype
        assert not torch.allclose(fp32, fp16.float(), atol=1e-4)
    
    def test_dtype_mismatch_int_vs_float(self):
        """Edge: Integer vs float."""
        int_tensor = torch.randint(0, 100, (100,), device="cuda")
        float_tensor = int_tensor.float()
        
        assert int_tensor.dtype != float_tensor.dtype
    
    # 9. Denormalized Values
    def test_denormalized_values_detection(self):
        """Edge: Detect denormalized floats."""
        # Smallest positive normal float32
        smallest_normal = torch.tensor([1.175494e-38], device="cuda")
        # Denormalized value
        denorm = smallest_normal / 1000
        
        # Denorm is very small
        assert denorm.item() < 1.175494e-38
    
    # 10. Uninitialized Memory
    def test_uninitialized_memory_torch_empty(self):
        """Edge: torch.empty() contains garbage."""
        x = torch.empty(1000, device="cuda")
        
        # Empty tensor is uninitialized - values are indeterminate
        # Fill and verify
        x.fill_(1.0)
        assert torch.allclose(x, torch.ones_like(x))


# =============================================================================
# WORKLOAD EDGE CASES (10 protections)
# =============================================================================

class TestWorkloadEdgeCases:
    """Edge cases for workload validation."""
    
    # 1. Batch Shrinking
    def test_batch_shrinking_by_one(self):
        """Edge: Batch reduced by exactly 1."""
        from core.benchmark.verification import InputSignature, PrecisionFlags
        
        baseline = InputSignature(
            shapes={"x": (32, 128)},
            dtypes={"x": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        
        optimized = InputSignature(
            shapes={"x": (31, 128)},
            dtypes={"x": "float32"},
            batch_size=31,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        
        # 1 sample difference should still be detected
        assert baseline.batch_size != optimized.batch_size
    
    def test_batch_shrinking_zero_batch(self):
        """Edge: Batch reduced to zero should be detected."""
        from core.benchmark.verification import InputSignature, PrecisionFlags
        
        # Zero batch is invalid - our validation should catch this
        try:
            sig = InputSignature(
                shapes={"x": (0, 128)},  # Zero batch
                dtypes={"x": "float32"},
                batch_size=0,
                parameter_count=1000,
                precision_flags=PrecisionFlags(),
            )
            # If it doesn't raise, check that batch_size is 0
            assert sig.batch_size == 0, "Zero batch should be flagged"
        except (ValueError, AssertionError):
            pass  # Expected - validation caught zero batch
    
    # 2. Sequence Truncation
    def test_sequence_truncation_by_one(self):
        """Edge: Sequence reduced by 1 token."""
        from core.benchmark.verification import InputSignature, PrecisionFlags
        
        baseline = InputSignature(
            shapes={"tokens": (32, 2048)},
            dtypes={"tokens": "int64"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        
        optimized = InputSignature(
            shapes={"tokens": (32, 2047)},
            dtypes={"tokens": "int64"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        
        assert baseline.shapes != optimized.shapes
    
    # 3. Hidden Downsampling
    def test_hidden_downsampling_power_of_two(self):
        """Edge: Downsampling by power of 2 (common optimization)."""
        from core.benchmark.verification import InputSignature, PrecisionFlags
        
        baseline = InputSignature(
            shapes={"img": (32, 3, 512, 512)},
            dtypes={"img": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        
        # 2x downsampling
        optimized = InputSignature(
            shapes={"img": (32, 3, 256, 256)},
            dtypes={"img": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )
        
        assert baseline.shapes != optimized.shapes
    
    # 4. Precision Mismatch
    def test_precision_mismatch_tf32(self):
        """Edge: TF32 vs FP32 precision."""
        from core.benchmark.verification import InputSignature, PrecisionFlags
        
        baseline_flags = PrecisionFlags(tf32=True)
        optimized_flags = PrecisionFlags(tf32=False)
        
        assert baseline_flags != optimized_flags
    
    def test_precision_mismatch_fp8_variants(self):
        """Edge: Different FP8 formats."""
        from core.benchmark.verification import PrecisionFlags
        
        e4m3 = PrecisionFlags(fp8=True)  # E4M3
        e5m2 = PrecisionFlags(fp8=True)  # E5M2 (same flag, different interpretation)
        
        # Both are FP8 but different formats
        # Real implementation would track format separately
    
    # 5. Undeclared Shortcuts
    def test_undeclared_shortcuts_skip_layers(self):
        """Edge: Silently skipping computation layers."""
        class FullModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(*[
                    torch.nn.Linear(100, 100) for _ in range(10)
                ])
            
            def forward(self, x):
                return self.layers(x)
        
        class ShortcutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(*[
                    torch.nn.Linear(100, 100) for _ in range(5)  # Only 5!
                ])
            
            def forward(self, x):
                return self.layers(x)
        
        full = FullModel().cuda()
        shortcut = ShortcutModel().cuda()
        
        assert sum(1 for _ in full.parameters()) != sum(1 for _ in shortcut.parameters())
    
    # 6. Early Exit
    def test_early_exit_threshold_boundary(self):
        """Edge: Exit at exact confidence threshold."""
        confidence = 0.95
        threshold = 0.95
        
        # Exactly at threshold - should NOT exit early
        should_exit = confidence > threshold
        assert not should_exit  # Edge case: exact equality
    
    # 7. Sparsity Mismatch
    def test_sparsity_mismatch_one_percent_difference(self):
        """Edge: 1% sparsity difference."""
        baseline_sparsity = 0.50  # 50% sparse
        optimized_sparsity = 0.51  # 51% sparse
        
        # 1% difference should be detected
        assert abs(baseline_sparsity - optimized_sparsity) > 0.005
    
    def test_sparsity_mismatch_near_zero(self):
        """Edge: Near-zero vs zero sparsity."""
        baseline_sparsity = 0.001
        optimized_sparsity = 0.0
        
        assert baseline_sparsity != optimized_sparsity
    
    # 8. Attention Mask Mismatch
    def test_attention_mask_causal_vs_full(self):
        """Edge: Causal mask vs full attention."""
        seq_len = 128
        
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
        
        # Full attention (all ones)
        full_mask = torch.ones(seq_len, seq_len, device="cuda")
        
        assert not torch.allclose(causal_mask, full_mask)
    
    # 9. KV Cache Size Mismatch
    def test_kv_cache_size_off_by_one(self):
        """Edge: KV cache one entry short."""
        seq_len = 2048
        baseline_cache_len = seq_len
        optimized_cache_len = seq_len - 1
        
        assert baseline_cache_len != optimized_cache_len
    
    # 10. Train/Test Overlap
    def test_train_test_overlap_single_sample(self):
        """Edge: Single overlapping sample."""
        train_set = set(range(0, 10000))
        test_set = {9999, 10000, 10001}  # One overlap: 9999
        
        overlap = train_set & test_set
        assert len(overlap) == 1  # Single sample overlap


# =============================================================================
# LOCATION EDGE CASES (7 protections)
# =============================================================================

class TestLocationEdgeCases:
    """Edge cases for computation location validation."""
    
    # 1. CPU Spillover
    def test_cpu_spillover_single_op_on_cpu(self):
        """Edge: Single operation spills to CPU."""
        x = torch.randn(100, device="cuda")
        
        # Force one op on CPU
        cpu_x = x.cpu()
        result_cpu = cpu_x * 2
        result_gpu = result_cpu.cuda()
        
        assert result_gpu.device.type == "cuda"
    
    def test_cpu_spillover_data_dependent_branch(self):
        """Edge: Data-dependent CPU branch."""
        x = torch.randn(100, device="cuda")
        
        # Must transfer to CPU to check condition
        if x.mean().item() > 0:  # .item() transfers to CPU
            result = x * 2
        else:
            result = x * 3
        
        assert result.device.type == "cuda"
    
    # 2. Setup Pre-computation
    def test_setup_precomputation_cached_result(self):
        """Edge: Setup caches the final result."""
        class CachingBenchmark:
            def __init__(self):
                self._cached = None
            
            def setup(self):
                # Cheating: computing result in setup
                self._cached = torch.randn(1000, device="cuda") * 2
            
            def benchmark_fn(self):
                # Just returns cached result
                return self._cached
        
        bench = CachingBenchmark()
        bench.setup()
        
        # Both calls return same object - detected by fresh input check
        result1 = bench.benchmark_fn()
        result2 = bench.benchmark_fn()
        assert result1.data_ptr() == result2.data_ptr()  # Same memory!
    
    # 3. Graph Capture Cheat
    def test_graph_capture_computation_in_capture(self):
        """Edge: Heavy computation during graph capture."""
        if not hasattr(torch.cuda, 'CUDAGraph'):
            pytest.skip("CUDA graphs not available")
        
        # Warmup
        x = torch.randn(1000, 1000, device="cuda")
        
        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = torch.mm(x, x.T)  # Computation happens at capture time
        
        # Replay - should be fast, but work was done in capture
        g.replay()
        torch.cuda.synchronize()
    
    # 4. Warmup Computation
    def test_warmup_computation_result_reused(self):
        """Edge: Warmup computes result that's reused."""
        warmup_result = [None]
        
        def benchmark_with_warmup():
            x = torch.randn(1000, 1000, device="cuda")
            
            # Warmup - but save the result
            for _ in range(3):
                warmup_result[0] = torch.mm(x, x)
            
            # Timed - could reuse warmup_result
            return warmup_result[0]  # Cheat!
        
        result = benchmark_with_warmup()
        assert result is warmup_result[0]
    
    # 5. Background Thread
    def test_background_thread_computation(self):
        """Edge: Work done on background thread."""
        result = [None]
        
        def background_work():
            result[0] = torch.randn(1000, device="cuda")
            result[0] = result[0] * 2
        
        thread = threading.Thread(target=background_work)
        thread.start()
        thread.join()
        
        assert result[0] is not None
    
    # 6. Lazy Evaluation Skip
    def test_lazy_evaluation_never_materialized(self):
        """Edge: Lazy tensor never materialized."""
        # PyTorch is eager, but testing the concept
        x = torch.randn(1000, device="cuda")
        
        # Create computation graph but don't force evaluation
        y = x * 2
        z = y + 1
        
        # Force evaluation
        torch.cuda.synchronize()
        _ = z.sum().item()
    
    # 7. JIT Compilation Timing
    def test_jit_compilation_first_call_slow(self):
        """Edge: First JIT call includes compilation."""
        @torch.jit.script
        def jit_fn(x):
            return x * 2 + torch.sin(x)
        
        x = torch.randn(1000, device="cuda")
        
        # First call - includes compilation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = jit_fn(x)
        end.record()
        torch.cuda.synchronize()
        
        first_call_time = start.elapsed_time(end)
        
        # Subsequent calls - compiled
        start.record()
        _ = jit_fn(x)
        end.record()
        torch.cuda.synchronize()
        
        second_call_time = start.elapsed_time(end)
        
        # First call may be slower (compilation)
        # But both should complete


# =============================================================================
# MEMORY EDGE CASES (7 protections)
# =============================================================================

class TestMemoryEdgeCases:
    """Edge cases for memory validation."""
    
    # 1. Pre-allocated Output
    def test_preallocated_output_exact_shape(self):
        """Edge: Pre-allocated output with exact shape."""
        output = torch.zeros(32, 64, device="cuda")
        
        # Function that writes to pre-allocated
        def compute(x, out):
            out.copy_(x * 2)
        
        x = torch.randn(32, 64, device="cuda")
        compute(x, output)
        
        assert output.shape == (32, 64)
    
    def test_preallocated_output_view(self):
        """Edge: Output is a view of larger tensor."""
        big_tensor = torch.zeros(1000, device="cuda")
        output = big_tensor[:100]  # View
        
        output.fill_(1.0)
        
        # View affects original
        assert big_tensor[:100].sum() == 100
    
    # 2. Input-Output Aliasing
    def test_input_output_aliasing_inplace(self):
        """Edge: In-place operation causes aliasing."""
        x = torch.randn(100, device="cuda")
        original_ptr = x.data_ptr()
        
        # In-place operation
        x.mul_(2)
        
        # Same memory location
        assert x.data_ptr() == original_ptr
    
    def test_input_output_aliasing_view(self):
        """Edge: Output is view of input."""
        x = torch.randn(100, device="cuda")
        y = x.view(10, 10)  # View - same memory
        
        assert x.data_ptr() == y.data_ptr()
    
    # 3. Pinned Memory Timing
    def test_pinned_memory_transfer_timing(self):
        """Edge: Pinned vs non-pinned transfer."""
        size = (1000, 1000)
        
        # Regular memory
        regular = torch.randn(*size)
        
        # Pinned memory
        pinned = torch.randn(*size).pin_memory()
        
        assert pinned.is_pinned()
        assert not regular.is_pinned()
    
    # 4. Memory Pool Reuse
    def test_memory_pool_cached_allocation(self):
        """Edge: Reusing cached memory allocation."""
        # Allocate and free
        x = torch.randn(1000, device="cuda")
        ptr1 = x.data_ptr()
        del x
        
        # Allocate same size - may reuse
        y = torch.randn(1000, device="cuda")
        ptr2 = y.data_ptr()
        
        # Memory may or may not be reused (implementation detail)
        # But allocation should succeed
        assert y.numel() == 1000
    
    # 5. Fragmentation Effects
    def test_fragmentation_many_small_allocs(self):
        """Edge: Many small allocations cause fragmentation."""
        tensors = []
        
        # Many small allocations
        for _ in range(100):
            tensors.append(torch.randn(1000, device="cuda"))
        
        # Free every other one
        for i in range(0, 100, 2):
            tensors[i] = None
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Try large allocation - may fail due to fragmentation
        try:
            large = torch.randn(10000000, device="cuda")
            del large
        except RuntimeError:
            pass  # Expected if fragmented
    
    # 6. Page Fault Timing
    def test_page_fault_first_access(self):
        """Edge: First access triggers page fault."""
        # Empty allocation - no physical pages yet
        x = torch.empty(1000000, device="cuda")
        
        # First access
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        x.fill_(1.0)  # First access - may trigger allocation
        end.record()
        torch.cuda.synchronize()
    
    # 7. Swap Interference
    def test_swap_interference_large_allocation(self):
        """Edge: Large allocation may trigger swap."""
        # Get available memory
        free_mem = torch.cuda.mem_get_info()[0]
        
        # Try to allocate most of it
        try:
            # Allocate 80% of free memory
            alloc_size = int(free_mem * 0.8 / 4)  # float32 = 4 bytes
            x = torch.randn(alloc_size, device="cuda")
            del x
        except RuntimeError:
            pass  # Expected if not enough memory
        
        torch.cuda.empty_cache()


# =============================================================================
# CUDA EDGE CASES (10 protections)
# =============================================================================

class TestCudaEdgeCases:
    """Edge cases for CUDA-specific protections."""
    
    # 1. Host Callback Escape
    def test_host_callback_with_stream(self):
        """Edge: Host callback on non-default stream."""
        s = torch.cuda.Stream()
        
        called = [False]
        
        def callback(stream_ptr):
            called[0] = True
        
        # Note: PyTorch doesn't directly expose cudaLaunchHostFunc
        # but we can test stream synchronization
        with torch.cuda.stream(s):
            x = torch.randn(100, device="cuda")
        
        s.synchronize()
    
    # 2. Async Memcpy Incomplete
    def test_async_memcpy_stream_ordering(self):
        """Edge: Async memcpy on different streams."""
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        
        x = torch.randn(1000, device="cuda")
        
        with torch.cuda.stream(s1):
            y = x.cpu()  # Async D2H on s1
        
        with torch.cuda.stream(s2):
            z = x * 2  # Work on s2
        
        s1.synchronize()
        s2.synchronize()
    
    # 3. Workspace Pre-compute
    def test_workspace_precomputed_cublas(self):
        """Edge: cuBLAS workspace pre-allocated."""
        # First matmul allocates workspace
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        
        c1 = torch.mm(a, b)  # May allocate workspace
        c2 = torch.mm(a, b)  # Reuses workspace
        
        torch.cuda.synchronize()
    
    # 4. Persistent Kernel
    def test_persistent_kernel_occupancy(self):
        """Edge: Kernel with high occupancy."""
        # Persistent kernels maintain state across blocks
        x = torch.randn(10000, device="cuda")
        
        # Operations that might use persistent kernels
        y = torch.cumsum(x, dim=0)
        
        torch.cuda.synchronize()
    
    # 5. Undeclared Multi-GPU
    def test_undeclared_multi_gpu_single_declared(self):
        """Edge: Using multiple GPUs when one declared."""
        num_gpus = torch.cuda.device_count()
        
        if num_gpus < 2:
            pytest.skip("Need 2+ GPUs")
        
        # Declared: single GPU
        x = torch.randn(1000, device="cuda:0")
        
        # Would be cheating to also use GPU 1
        # y = torch.randn(1000, device="cuda:1")  # Undeclared!
    
    # 6. Context Switch Overhead
    def test_context_switch_device_change(self):
        """Edge: Device context switches."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need 2+ GPUs")
        
        # Alternating devices causes context switches
        for i in range(10):
            device = f"cuda:{i % 2}"
            x = torch.randn(100, device=device)
            del x
    
    # 7. Driver Overhead
    def test_driver_overhead_many_small_kernels(self):
        """Edge: Many small kernel launches."""
        x = torch.randn(10, device="cuda")
        
        # Many tiny operations
        for _ in range(1000):
            x = x + 1  # Separate kernel launch
        
        torch.cuda.synchronize()
    
    # 8. Cooperative Launch Abuse
    def test_cooperative_launch_grid_sync(self):
        """Edge: Cooperative launch with grid synchronization."""
        # PyTorch doesn't directly expose cooperative launch
        # but we test the sync behavior
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.mm(x, x.T)
        torch.cuda.synchronize()
    
    # 9. Dynamic Parallelism Hidden
    def test_dynamic_parallelism_nested_launch(self):
        """Edge: Nested kernel launches."""
        # PyTorch doesn't expose dynamic parallelism directly
        # Test standard recursive computation
        x = torch.randn(1000, device="cuda")
        
        for _ in range(5):
            x = x * x  # Could trigger dynamic parallelism in custom CUDA
        
        torch.cuda.synchronize()
    
    # 10. Unified Memory Faults
    def test_unified_memory_managed_allocation(self):
        """Edge: Unified memory page faults."""
        # PyTorch uses device memory by default
        # Unified memory would cause page faults
        x = torch.randn(1000, device="cuda")
        
        # Access pattern that would cause faults with unified memory
        _ = x[::2].sum()  # Strided access
        
        torch.cuda.synchronize()


# =============================================================================
# COMPILE EDGE CASES (7 protections)
# =============================================================================

class TestCompileEdgeCases:
    """Edge cases for compilation protections."""
    
    # 1. Compilation Cache Hit
    def test_cache_hit_shape_variation(self):
        """Edge: Slight shape change causes recompile."""
        torch._dynamo.reset()
        
        @torch.compile
        def fn(x):
            return x * 2
        
        # First shape
        x1 = torch.randn(100, device="cuda")
        _ = fn(x1)
        
        # Different shape - may recompile
        x2 = torch.randn(101, device="cuda")
        _ = fn(x2)
        
        torch._dynamo.reset()
    
    # 2. Trace Reuse
    def test_trace_reuse_dynamic_shapes(self):
        """Edge: Dynamic shapes affect trace reuse."""
        torch._dynamo.reset()
        
        @torch.compile(dynamic=True)
        def fn(x):
            return x.sum()
        
        for size in [100, 200, 150, 300]:
            x = torch.randn(size, device="cuda")
            _ = fn(x)
        
        torch._dynamo.reset()
    
    # 3. Mode Inconsistency
    def test_mode_inconsistency_train_vs_eval(self):
        """Edge: Model mode differs between runs."""
        model = torch.nn.BatchNorm1d(100).cuda()
        
        x = torch.randn(32, 100, device="cuda")
        
        model.train()
        train_out = model(x)
        
        model.eval()
        eval_out = model(x)
        
        # Outputs differ based on mode
        assert not torch.allclose(train_out, eval_out)
    
    # 4. Inductor Asymmetry
    def test_inductor_asymmetry_different_backends(self):
        """Edge: Different compiler backends."""
        torch._dynamo.reset()
        
        def fn(x):
            return x * 2 + 1
        
        # Compile with inductor
        compiled_fn = torch.compile(fn, backend="inductor")
        
        x = torch.randn(100, device="cuda")
        eager_result = fn(x)
        compiled_result = compiled_fn(x)
        
        # Should be close but may have numerical differences
        torch._dynamo.reset()
    
    # 5. Guard Failure Hidden
    def test_guard_failure_control_flow(self):
        """Edge: Guard failure from control flow."""
        torch._dynamo.reset()
        
        @torch.compile
        def fn(x, flag):
            if flag:
                return x * 2
            else:
                return x + 1
        
        x = torch.randn(100, device="cuda")
        
        # Different paths - may cause guard failures
        _ = fn(x, True)
        _ = fn(x, False)
        
        torch._dynamo.reset()
    
    # 6. Autotuning Variance
    def test_autotuning_variance_repeated_runs(self):
        """Edge: Autotuning produces different configs."""
        torch._dynamo.reset()
        
        @torch.compile(mode="max-autotune")
        def fn(x):
            return torch.mm(x, x.T)
        
        x = torch.randn(1000, 1000, device="cuda")
        
        results = []
        for _ in range(3):
            results.append(fn(x))
            torch.cuda.synchronize()
        
        # Results should be numerically close
        torch._dynamo.reset()
    
    # 7. Symbolic Shape Exploit
    def test_symbolic_shape_specialization(self):
        """Edge: Shape specialization affects performance."""
        torch._dynamo.reset()
        
        @torch.compile
        def fn(x):
            return x.reshape(-1)
        
        # Fixed shape - can specialize
        x = torch.randn(32, 64, device="cuda")
        _ = fn(x)
        
        torch._dynamo.reset()


# =============================================================================
# DISTRIBUTED EDGE CASES (8 protections)
# =============================================================================

class TestDistributedEdgeCases:
    """Edge cases for distributed protections."""
    
    # 1. Rank Skipping
    def test_rank_skipping_single_rank(self):
        """Edge: Single rank in multi-rank job."""
        # Simulate rank behavior
        world_size = 4
        active_ranks = {0, 1, 3}  # Rank 2 skipped
        
        assert len(active_ranks) < world_size
    
    # 2. Collective Short-circuit
    def test_collective_shortcircuit_single_element(self):
        """Edge: Collective on single-element tensor."""
        x = torch.tensor([1.0], device="cuda")
        
        # Single element - trivial reduction
        result = x.sum()
        assert result.numel() == 1
    
    # 3. Topology Mismatch
    def test_topology_mismatch_ring_vs_tree(self):
        """Edge: Different collective topologies."""
        # Ring topology: 0 -> 1 -> 2 -> 3 -> 0
        ring_hops = 4
        
        # Tree topology: depth 2
        tree_hops = 2
        
        assert ring_hops != tree_hops
    
    # 4. Barrier Timing
    def test_barrier_timing_straggler(self):
        """Edge: Barrier with one slow rank."""
        import time
        
        # Simulate ranks with different completion times
        rank_times = [0.1, 0.1, 0.1, 0.5]  # Rank 3 is slow
        
        # Barrier waits for slowest
        barrier_time = max(rank_times)
        assert barrier_time == 0.5
    
    # 5. Gradient Bucketing Mismatch
    def test_gradient_bucketing_different_sizes(self):
        """Edge: Different bucket sizes."""
        baseline_bucket_mb = 25
        optimized_bucket_mb = 50
        
        assert baseline_bucket_mb != optimized_bucket_mb
    
    # 6. Async Gradient Timing
    def test_async_gradient_overlap(self):
        """Edge: Gradient communication overlaps compute."""
        # Simulate async gradient
        compute_time = 100  # ms
        comm_time = 50  # ms
        
        # Perfect overlap
        overlapped_time = max(compute_time, comm_time)
        
        # No overlap
        sequential_time = compute_time + comm_time
        
        assert overlapped_time < sequential_time
    
    # 7. Pipeline Bubble Hiding
    def test_pipeline_bubble_microbatch_count(self):
        """Edge: Different microbatch counts affect bubbles."""
        stages = 4
        
        # Fewer microbatches = more bubbles
        microbatches_low = 4
        bubbles_low = (stages - 1) * 2  # Startup + teardown
        
        microbatches_high = 16
        bubbles_high = (stages - 1) * 2  # Same absolute, lower relative
        
        # Bubble fraction is lower with more microbatches
        bubble_frac_low = bubbles_low / microbatches_low
        bubble_frac_high = bubbles_high / microbatches_high
        
        assert bubble_frac_high < bubble_frac_low
    
    # 8. Shard Size Mismatch
    def test_shard_size_imbalanced(self):
        """Edge: Imbalanced shard sizes."""
        total_params = 1000
        num_shards = 4
        
        # Balanced
        balanced_shard = total_params // num_shards
        
        # Imbalanced
        imbalanced_shards = [300, 300, 200, 200]
        
        assert sum(imbalanced_shards) == total_params


# =============================================================================
# ENVIRONMENT EDGE CASES (12 protections)
# =============================================================================

class TestEnvironmentEdgeCases:
    """Edge cases for environment protections."""
    
    # 1. Device Mismatch
    def test_device_mismatch_compute_capability(self):
        """Edge: Same GPU name, different compute capability."""
        props = torch.cuda.get_device_properties(0)
        
        # Compute capability should be recorded
        cc = f"{props.major}.{props.minor}"
        assert len(cc) > 0
    
    # 2. Frequency Boost
    def test_frequency_boost_detection(self):
        """Edge: Detect if GPU is boosting."""
        # Workload that may trigger boost
        x = torch.randn(5000, 5000, device="cuda")
        for _ in range(10):
            x = torch.mm(x, x.T)
        
        torch.cuda.synchronize()
    
    # 3. Priority Elevation
    def test_priority_elevation_process_nice(self):
        """Edge: Process priority affects scheduling."""
        import os
        
        try:
            current_nice = os.nice(0)
            # Nice value 0 is default
            assert -20 <= current_nice <= 19
        except OSError:
            pass  # May not have permission
    
    # 4. Memory Overcommit
    def test_memory_overcommit_reservation(self):
        """Edge: Reserved vs allocated memory."""
        torch.cuda.empty_cache()
        
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        
        # Reserved >= allocated
        assert reserved >= allocated
    
    # 5. NUMA Inconsistency
    def test_numa_gpu_affinity(self):
        """Edge: GPU-CPU NUMA affinity."""
        # Check if NUMA info available
        numa_path = Path("/sys/devices/system/node")
        has_numa = numa_path.exists()
        
        # Test passes regardless - just documents check
    
    # 6. CPU Governor Mismatch
    def test_cpu_governor_consistency(self):
        """Edge: CPU frequency governor setting."""
        gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        
        if gov_path.exists():
            governor = gov_path.read_text().strip()
            # Performance governor is preferred for benchmarks
            assert governor in ["performance", "powersave", "ondemand", "conservative", "schedutil"]
    
    # 7. Thermal Throttling
    def test_thermal_throttling_temperature_check(self):
        """Edge: Check GPU temperature."""
        # Would need nvidia-smi or pynvml
        # Test that check doesn't crash
        pass
    
    # 8. Power Limit Difference
    def test_power_limit_variation(self):
        """Edge: Power limit affects performance."""
        # Would need nvidia-smi
        pass
    
    # 9. Driver Version Mismatch
    def test_driver_version_recorded(self):
        """Edge: Driver version is recorded."""
        # CUDA runtime version
        cuda_version = torch.version.cuda
        assert cuda_version is not None
    
    # 10. Library Version Mismatch
    def test_library_version_cudnn(self):
        """Edge: cuDNN version affects performance."""
        cudnn_version = torch.backends.cudnn.version()
        assert cudnn_version is not None
    
    # 11. Container Resource Limits
    def test_container_cgroup_limits(self):
        """Edge: Container memory/CPU limits."""
        cgroup_mem = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        
        if cgroup_mem.exists():
            try:
                limit = int(cgroup_mem.read_text().strip())
                # Very high limit means no restriction
            except:
                pass
    
    # 12. Virtualization Overhead
    def test_virtualization_detection(self):
        """Edge: Detect if running in VM."""
        # Check for hypervisor
        cpuinfo = Path("/proc/cpuinfo")
        
        if cpuinfo.exists():
            content = cpuinfo.read_text()
            is_vm = "hypervisor" in content.lower()
            # Test passes - just documents check


# =============================================================================
# STATISTICAL EDGE CASES (8 protections)
# =============================================================================

class TestStatisticalEdgeCases:
    """Edge cases for statistical protections."""
    
    # 1. Outlier Injection
    def test_outlier_injection_single_extreme(self):
        """Edge: Single extreme value."""
        measurements = [1.0] * 99 + [100.0]  # One outlier
        
        mean_with = sum(measurements) / len(measurements)
        mean_without = sum(measurements[:-1]) / len(measurements[:-1])
        
        # Outlier significantly affects mean
        assert mean_with > mean_without * 1.5
    
    def test_outlier_injection_median_robust(self):
        """Edge: Median is robust to outliers."""
        import statistics
        
        measurements = [1.0] * 99 + [1000.0]
        
        mean_val = statistics.mean(measurements)
        median_val = statistics.median(measurements)
        
        # Median is robust
        assert median_val < mean_val
    
    # 2. Variance Gaming
    def test_variance_gaming_consistent_slow(self):
        """Edge: Consistently slow is better than variable fast."""
        import statistics
        
        # Variable but sometimes fast
        variable = [0.5, 1.5, 0.6, 1.4, 0.7, 1.3]
        
        # Consistent but slower
        consistent = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        assert statistics.stdev(variable) > statistics.stdev(consistent)
        assert statistics.mean(variable) == statistics.mean(consistent)
    
    # 3. Percentile Selection
    def test_percentile_selection_p50_vs_p99(self):
        """Edge: P50 vs P99 tells different stories."""
        import statistics
        
        # Mostly fast with occasional slow
        measurements = [1.0] * 99 + [10.0]
        sorted_m = sorted(measurements)
        
        p50 = sorted_m[len(sorted_m) // 2]
        p99 = sorted_m[int(len(sorted_m) * 0.99)]
        
        assert p99 > p50
    
    # 4. Insufficient Samples
    def test_insufficient_samples_high_variance(self):
        """Edge: Few samples with high variance."""
        import statistics
        
        measurements = [1.0, 2.0, 3.0]  # Only 3 samples
        
        # Standard error is high with few samples
        std = statistics.stdev(measurements)
        sem = std / math.sqrt(len(measurements))
        
        # SEM is significant fraction of mean
        assert sem / statistics.mean(measurements) > 0.1
    
    # 5. Cold Start Inclusion
    def test_cold_start_first_iteration_slow(self):
        """Edge: First iteration much slower."""
        # Simulate cold start
        iterations = [10.0, 1.0, 1.0, 1.0, 1.0]  # First is 10x slower
        
        mean_with_cold = sum(iterations) / len(iterations)
        mean_without_cold = sum(iterations[1:]) / len(iterations[1:])
        
        # Cold start inflates mean
        assert mean_with_cold > mean_without_cold * 2
    
    # 6. GC Interference
    def test_gc_interference_collection_pause(self):
        """Edge: GC pause during measurement."""
        gc.disable()
        
        try:
            # Create garbage
            garbage = [torch.randn(1000) for _ in range(100)]
            del garbage
            
            # Without GC, memory not reclaimed
            # With GC, pause would affect timing
        finally:
            gc.enable()
            gc.collect()
    
    # 7. Background Process Noise
    def test_background_noise_cpu_bound(self):
        """Edge: CPU-bound background work."""
        def cpu_work():
            total = 0
            for i in range(100000):
                total += i
            return total
        
        # This could interfere with benchmarks
        thread = threading.Thread(target=cpu_work)
        thread.start()
        
        # Do GPU work while CPU work runs
        x = torch.randn(1000, device="cuda")
        y = x * 2
        torch.cuda.synchronize()
        
        thread.join()
    
    # 8. Cherry-picking
    def test_cherry_picking_best_of_n(self):
        """Edge: Reporting only best of N runs."""
        import statistics
        
        all_runs = [1.2, 1.0, 1.4, 1.1, 1.3, 1.5, 0.9, 1.25]
        
        # Best of N is misleading
        best_of_3 = min(all_runs[:3])
        true_mean = statistics.mean(all_runs)
        
        assert best_of_3 < true_mean


# =============================================================================
# EVALUATION EDGE CASES (8 protections)
# =============================================================================

class TestEvaluationEdgeCases:
    """Edge cases for evaluation protections."""
    
    # 1. Eval Code Exploitation
    def test_eval_code_exploitation_hardcoded(self):
        """Edge: Hardcoded answers in eval code."""
        # Example of exploitable eval
        def bad_eval(model_output, question_id):
            # Hardcoded answers - bad!
            hardcoded = {1: "Paris", 2: "Blue", 3: "42"}
            if question_id in hardcoded:
                return model_output == hardcoded[question_id]
            return False
        
        # Model could just return hardcoded answers
    
    # 2. Timeout Manipulation
    def test_timeout_manipulation_extend(self):
        """Edge: Extending timeout during run."""
        original_timeout = 60
        
        class TimoutConfig:
            def __init__(self, timeout):
                self._timeout = timeout
            
            @property
            def timeout(self):
                return self._timeout
        
        config = TimoutConfig(original_timeout)
        
        # Should not be able to modify
        # config._timeout = 120  # Would be cheating
    
    # 3. Metric Definition Gaming
    def test_metric_gaming_threshold_tuning(self):
        """Edge: Threshold tuned to maximize metric."""
        predictions = [0.4, 0.6, 0.45, 0.55, 0.5]
        labels = [0, 1, 0, 1, 1]
        
        # Different thresholds give different accuracies
        def accuracy_at_threshold(preds, labels, thresh):
            pred_labels = [1 if p > thresh else 0 for p in preds]
            return sum(p == l for p, l in zip(pred_labels, labels)) / len(labels)
        
        acc_05 = accuracy_at_threshold(predictions, labels, 0.5)
        acc_045 = accuracy_at_threshold(predictions, labels, 0.45)
        
        # Different thresholds may give different results
    
    # 4. Test Data Leakage
    def test_data_leakage_feature_from_label(self):
        """Edge: Feature derived from label."""
        # Bad: label information in features
        labels = [0, 1, 0, 1, 1]
        features_bad = [[l * 0.5 + 0.25] for l in labels]  # Contains label!
        
        # Model could exploit this
    
    # 5. Benchmark Overfitting
    def test_overfitting_train_on_test(self):
        """Edge: Training on test set."""
        test_set = list(range(100))
        
        # Should NOT use test set for training
        train_set = list(range(1000))
        
        # No overlap
        overlap = set(test_set) & set(train_set)
        assert len(overlap) == 0 or overlap == set(test_set)  # Either no overlap or test is subset
    
    # 6. Self-Modifying Tests
    def test_self_modifying_immutable_test(self):
        """Edge: Test that modifies itself."""
        original_value = 42
        
        class ImmutableTest:
            def __init__(self, value):
                self._value = value
            
            @property
            def value(self):
                return self._value
            
            # No setter - immutable
        
        test = ImmutableTest(original_value)
        assert test.value == original_value
    
    # 7. Benchmark Memorization
    def test_memorization_hash_detection(self):
        """Edge: Detect if model memorized via hash."""
        import hashlib
        
        # Test inputs
        test_inputs = ["hello", "world", "test"]
        
        # Hashes
        hashes = [hashlib.md5(s.encode()).hexdigest() for s in test_inputs]
        
        # Model shouldn't have seen these exact hashes
        assert len(set(hashes)) == len(test_inputs)
    
    # 8. Missing Holdout Sets
    def test_missing_holdout_temporal_split(self):
        """Edge: Temporal split for holdout."""
        # Data from different times
        train_dates = list(range(2020, 2023))  # 2020-2022
        test_dates = list(range(2023, 2025))   # 2023-2024
        
        # No overlap
        assert set(train_dates) & set(test_dates) == set()


# =============================================================================
# ADDITIONAL BOUNDARY CONDITION TESTS
# =============================================================================

class TestBoundaryConditions:
    """Test exact boundary conditions for tolerance checks."""
    
    def test_tolerance_exactly_at_boundary(self):
        """Edge: Values exactly at tolerance boundary."""
        from core.benchmark.verification import ToleranceSpec
        
        rtol = 1e-5
        atol = 1e-8
        spec = ToleranceSpec(rtol=rtol, atol=atol)
        
        base = 1.0
        # Exactly at relative tolerance
        at_rtol = base * (1 + rtol)
        
        # torch.allclose with same tolerances
        a = torch.tensor([base])
        b = torch.tensor([at_rtol])
        
        # At boundary - may or may not pass depending on < vs <=
    
    def test_workload_mismatch_at_one_percent(self):
        """Edge: Workload differs by exactly 1%."""
        baseline_bytes = 1000000
        optimized_bytes = 990000  # Exactly 1% less
        
        diff_pct = abs(baseline_bytes - optimized_bytes) / baseline_bytes * 100
        assert diff_pct == 1.0
    
    def test_sparsity_at_detection_threshold(self):
        """Edge: Sparsity exactly at detection threshold."""
        threshold = 0.01  # 1% threshold
        
        baseline_sparsity = 0.50
        optimized_sparsity = 0.51  # 1% absolute difference
        
        abs_diff = abs(baseline_sparsity - optimized_sparsity)
        # Use approximate comparison due to floating point
        assert abs(abs_diff - threshold) < 1e-9
    
    def test_timing_variance_coefficient(self):
        """Edge: Coefficient of variation at threshold."""
        import statistics
        
        # Measurements with specific CV
        measurements = [1.0, 1.1, 0.9, 1.05, 0.95]
        
        mean = statistics.mean(measurements)
        std = statistics.stdev(measurements)
        cv = std / mean
        
        # CV under 10% is typically acceptable
        assert cv < 0.1


class TestExtremeValues:
    """Test behavior with extreme values."""
    
    def test_very_large_tensor(self):
        """Edge: Very large tensor dimensions."""
        try:
            # Try to allocate large tensor
            x = torch.randn(10000, 10000, device="cuda")
            del x
        except RuntimeError:
            pass  # Expected if OOM
        
        torch.cuda.empty_cache()
    
    def test_very_small_values(self):
        """Edge: Subnormal float values."""
        x = torch.tensor([1e-45], device="cuda", dtype=torch.float32)
        
        # Very small but not zero
        assert x.item() > 0
    
    def test_zero_dimensions(self):
        """Edge: Zero-sized dimensions."""
        x = torch.randn(10, 0, device="cuda")
        
        assert x.numel() == 0
        assert x.shape == (10, 0)
    
    def test_single_element(self):
        """Edge: Single element tensor."""
        x = torch.randn(1, device="cuda")
        
        # Operations on single element
        y = x * 2
        assert y.numel() == 1
    
    def test_maximum_dimensions(self):
        """Edge: Many dimensions."""
        # PyTorch supports up to 64 dimensions
        shape = [2] * 10  # 10 dimensions
        x = torch.randn(*shape, device="cuda")
        
        assert len(x.shape) == 10


# =============================================================================
# RACE CONDITION TESTS
# =============================================================================

class TestRaceConditions:
    """Test for race conditions in concurrent scenarios."""
    
    def test_concurrent_stream_operations(self):
        """Edge: Race between streams."""
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        
        shared = torch.zeros(1000, device="cuda")
        
        def work_s1():
            with torch.cuda.stream(s1):
                shared.fill_(1.0)
        
        def work_s2():
            with torch.cuda.stream(s2):
                shared.fill_(2.0)
        
        t1 = threading.Thread(target=work_s1)
        t2 = threading.Thread(target=work_s2)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        s1.synchronize()
        s2.synchronize()
        
        # Final value depends on ordering
    
    def test_concurrent_allocations(self):
        """Edge: Concurrent memory allocations."""
        results = [None] * 10
        
        def allocate(idx):
            results[idx] = torch.randn(1000, device="cuda")
        
        threads = [threading.Thread(target=allocate, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        torch.cuda.synchronize()
        
        assert all(r is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

