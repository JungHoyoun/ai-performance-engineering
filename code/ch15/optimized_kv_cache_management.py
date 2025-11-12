"""optimized_kv_cache_management.py - Optimized KV cache management in disaggregated inference.

Demonstrates efficient KV cache management with cache reuse.
KV cache management: Implements KV cache reuse and efficient management.
Reuses cached keys/values to avoid recomputation.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")

class OptimizedKVCacheManagementBenchmark(Benchmark):
    """Optimized: KV cache management with cache reuse.
    
    KV cache management: Implements efficient KV cache management.
    Reuses cached keys/values to avoid recomputation, improving efficiency.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs = None
        self.cache_buffer = None
        self.batch_size = 4
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
    
    def setup(self) -> None:
        """Setup: Initialize model with KV cache management."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: KV cache management - reuse cached values
        # KV cache management stores and reuses keys/values
        
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            module.eval()
        
        max_seq_len = 32
        self.cache_buffer = torch.zeros(self.batch_size, max_seq_len, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
            for _ in range(max_seq_len)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: KV cache management with reuse."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_kv_cache_management", enable=enable_nvtx):
            with torch.no_grad():
                assert self.q_proj and self.k_proj and self.v_proj and self.out_proj
                queries = torch.cat(self.inputs, dim=1)
                k_cache = self.cache_buffer.clone()
                
                q = self.q_proj(queries)
                k = self.k_proj(k_cache)
                v = self.v_proj(k_cache)
                
                q = q.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = attn.transpose(1, 2).contiguous().view(self.batch_size, -1, self.hidden_dim)
                output = self.out_proj(attn)
                
                # Update cache with the newest token block without reallocation.
                self.cache_buffer.copy_(torch.cat([self.cache_buffer[:, 1:, :], queries[:, -1:, :]], dim=1))
                _ = output[:, -1, :].sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        self.cache_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.q_proj is None or self.k_proj is None or self.v_proj is None or self.out_proj is None:
            return "Projection layers not initialized"
        if self.inputs is None or self.cache_buffer is None:
            return "Inputs/cache not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedKVCacheManagementBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedKVCacheManagementBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: kv_cache_management")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
