"""Tests for inference components."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


class TestKVCacheManager:
    """Tests for PagedKVCache."""
    
    def test_allocation(self):
        """Test block allocation."""
        from components.kv_cache_manager import PagedKVCache, KVCacheConfig
        
        config = KVCacheConfig(
            block_size=16,
            num_blocks=100,
            num_layers=4,
            num_heads=8,
            head_dim=64,
        )
        
        cache = PagedKVCache(config, device=torch.device("cpu"))
        
        # Allocate blocks
        blocks = cache.allocate("seq_1", num_tokens=50)
        assert len(blocks) == 4  # ceil(50/16)
        
        # Check usage
        usage = cache.memory_usage()
        assert usage["used_blocks"] == 4
        assert usage["free_blocks"] == 96
    
    def test_free(self):
        """Test block deallocation."""
        from components.kv_cache_manager import PagedKVCache, KVCacheConfig
        
        config = KVCacheConfig(
            block_size=16,
            num_blocks=100,
            num_layers=4,
            num_heads=8,
            head_dim=64,
        )
        
        cache = PagedKVCache(config, device=torch.device("cpu"))
        
        cache.allocate("seq_1", num_tokens=32)
        assert cache.memory_usage()["used_blocks"] == 2
        
        cache.free("seq_1")
        assert cache.memory_usage()["used_blocks"] == 0
        assert cache.memory_usage()["free_blocks"] == 100


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_basic_metrics(self):
        """Test basic metric collection."""
        from components.monitoring import MetricsCollector
        
        collector = MetricsCollector(track_power=False)
        
        collector.start_request("test_req")
        collector.record_first_token("test_req")
        collector.record_decode_step("test_req", 10.0)
        collector.record_decode_step("test_req", 12.0)
        collector.end_request("test_req", prompt_tokens=100, output_tokens=50)
        
        metrics = collector.compute_metrics()
        
        assert metrics.batch_size == 1
        assert metrics.prompt_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.tpot_ms == 11.0  # Average of 10 and 12
    
    def test_percentiles(self):
        """Test percentile computation."""
        from components.monitoring import MetricsCollector
        
        collector = MetricsCollector(track_power=False)
        
        # Add 20 requests with varying TTFT
        for i in range(20):
            req_id = f"req_{i}"
            collector.start_request(req_id)
            collector.record_first_token(req_id)
            collector.end_request(req_id, prompt_tokens=100, output_tokens=10)
        
        metrics = collector.compute_metrics()
        
        # Should have percentiles for 20+ requests
        assert metrics.ttft_p50_ms is not None
        assert metrics.ttft_p90_ms is not None
        assert metrics.ttft_p99_ms is not None


class TestWorkloadLoader:
    """Tests for WorkloadLoader."""
    
    def test_synthetic_workload(self):
        """Test synthetic workload generation."""
        from components.workload_loader import WorkloadLoader, WorkloadConfig, WorkloadType
        
        config = WorkloadConfig(
            workload_type=WorkloadType.SYNTHETIC,
            num_samples=10,
        )
        
        loader = WorkloadLoader(config)
        prompts = loader.get_prompts()
        
        assert len(prompts) == 10
        assert all(p.text for p in prompts)
        assert all(p.source == "synthetic" for p in prompts)
    
    def test_stats(self):
        """Test workload statistics."""
        from components.workload_loader import WorkloadLoader, WorkloadConfig, WorkloadType
        
        config = WorkloadConfig(
            workload_type=WorkloadType.SYNTHETIC,
            num_samples=5,
        )
        
        loader = WorkloadLoader(config)
        stats = loader.get_stats()
        
        assert stats["num_prompts"] == 5
        assert stats["workload_type"] == "synthetic"
        assert "mean_tokens" in stats


class TestPrefixCache:
    """Tests for PrefixCache."""
    
    def test_cache_hit(self):
        """Test cache hit/miss."""
        from components.prefix_cache import PrefixCache
        
        cache = PrefixCache(max_size_gb=0.1)
        
        # Create dummy KV tensors
        kv = [(torch.randn(1, 10, 64), torch.randn(1, 10, 64))]
        
        # First access - miss
        result = cache.get([1, 2, 3])
        assert result is None
        assert cache.misses == 1
        
        # Store
        cache.put([1, 2, 3], kv)
        
        # Second access - hit
        result = cache.get([1, 2, 3])
        assert result is not None
        assert cache.hits == 1
    
    def test_eviction(self):
        """Test LRU eviction."""
        from components.prefix_cache import PrefixCache
        
        # Very small cache
        cache = PrefixCache(max_size_gb=0.0001)
        
        # Add entries until eviction happens
        for i in range(10):
            kv = [(torch.randn(1, 100, 64), torch.randn(1, 100, 64))]
            cache.put([i], kv)
        
        # Should have evicted some entries
        assert len(cache.cache) < 10


class TestGQAOptimizer:
    """Tests for GQAOptimizer."""
    
    def test_expand_kv(self):
        """Test KV expansion."""
        from components.gqa_optimizer import GQAOptimizer
        
        optimizer = GQAOptimizer(num_q_heads=32, num_kv_heads=4)
        
        k = torch.randn(1, 4, 100, 64)  # 4 KV heads
        v = torch.randn(1, 4, 100, 64)
        
        k_exp, v_exp = optimizer.expand_kv(k, v)
        
        assert k_exp.shape == (1, 32, 100, 64)  # Expanded to 32
        assert v_exp.shape == (1, 32, 100, 64)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        from components.gqa_optimizer import GQAOptimizer
        
        # MQA
        mqa = GQAOptimizer(num_q_heads=32, num_kv_heads=1)
        assert mqa.get_compression_ratio() == 32
        
        # GQA
        gqa = GQAOptimizer(num_q_heads=32, num_kv_heads=8)
        assert gqa.get_compression_ratio() == 4
        
        # MHA
        mha = GQAOptimizer(num_q_heads=32, num_kv_heads=32)
        assert mha.get_compression_ratio() == 1


class TestSpeculativeDecoder:
    """Tests for speculative decoding components."""
    
    def test_ngram_speculator(self):
        """Test n-gram speculation."""
        from components.speculative_decoder import NGramSpeculator
        
        speculator = NGramSpeculator(n=3)
        
        # Build cache from sample sequence
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 6, 7]
        speculator.build_cache(tokens)
        
        # Should predict based on n-gram
        assert (1, 2) in speculator.cache
        assert 3 in speculator.cache[(1, 2)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


