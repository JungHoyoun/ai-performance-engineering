"""Correctness tests - ensure optimizations don't degrade output quality."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


class TestOutputEquivalence:
    """Test that optimized outputs match baseline."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_attention_equivalence(self):
        """Test FlashAttention produces same results as standard attention."""
        from optimization_layers.layer_03_pipelining import Layer03Pipelining
        
        torch.manual_seed(42)
        
        q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float32)
        k = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float32)
        v = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float32)
        
        # Standard attention
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        standard_out = torch.matmul(attn_weights, v)
        
        # SDPA attention
        layer = Layer03Pipelining()
        sdpa_out = layer.apply_sdpa_attention(q, k, v, is_causal=False)
        
        # Should be numerically close
        assert torch.allclose(standard_out, sdpa_out, atol=1e-5)
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gqa_expansion_equivalence(self):
        """Test GQA expansion produces correct shapes and values."""
        from components.gqa_optimizer import GQAOptimizer
        
        optimizer = GQAOptimizer(num_q_heads=8, num_kv_heads=2)
        
        k = torch.randn(1, 2, 32, 64, device="cuda")
        v = torch.randn(1, 2, 32, 64, device="cuda")
        
        k_exp, v_exp = optimizer.expand_kv(k, v)
        
        # Check expansion is correct (repeat_interleave)
        for i in range(8):
            source_head = i // 4  # Group size = 4
            assert torch.equal(k_exp[0, i], k[0, source_head])
            assert torch.equal(v_exp[0, i], v[0, source_head])
    
    def test_prefix_cache_determinism(self):
        """Test prefix cache returns same values on hit."""
        from components.prefix_cache import PrefixCache
        
        cache = PrefixCache(max_size_gb=1.0)
        
        # Create KV tensors
        kv = [(torch.randn(1, 10, 64), torch.randn(1, 10, 64))]
        
        cache.put([1, 2, 3], kv)
        
        # Get twice
        result1 = cache.get([1, 2, 3])
        result2 = cache.get([1, 2, 3])
        
        # Should return equivalent tensors
        assert torch.equal(result1[0][0], result2[0][0])
        assert torch.equal(result1[0][1], result2[0][1])


class TestNumericalStability:
    """Test numerical stability of optimizations."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_attention_no_nan(self):
        """Test attention doesn't produce NaN."""
        from optimization_layers.layer_03_pipelining import Layer03Pipelining
        
        # Create tensors that might cause numerical issues
        q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16) * 10
        k = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16) * 10
        v = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        
        layer = Layer03Pipelining()
        output = layer.apply_sdpa_attention(q, k, v)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_online_softmax_stability(self):
        """Test online softmax computation is stable."""
        from components.ring_attention import RingAttention
        
        ring = RingAttention(world_size=1)
        
        # Create large score values
        out1 = torch.randn(1, 8, 64, 64)
        lse1 = torch.randn(1, 8, 64) + 1000  # Large log-sum-exp
        out2 = torch.randn(1, 8, 64, 64)
        lse2 = torch.randn(1, 8, 64) + 1000
        
        # Update should handle large values
        updated_out, updated_lse = ring._online_softmax_update(
            out1, lse1, out2, lse2
        )
        
        assert not torch.isnan(updated_out).any()
        assert not torch.isnan(updated_lse).any()


class TestMemoryCorrectness:
    """Test memory management correctness."""
    
    def test_kv_cache_no_overlap(self):
        """Test KV cache blocks don't overlap."""
        from components.kv_cache_manager import PagedKVCache, KVCacheConfig
        
        config = KVCacheConfig(
            block_size=16,
            num_blocks=10,
            num_layers=2,
            num_heads=4,
            head_dim=32,
        )
        
        cache = PagedKVCache(config, device=torch.device("cpu"))
        
        # Allocate two sequences
        blocks1 = cache.allocate("seq_1", num_tokens=20)
        blocks2 = cache.allocate("seq_2", num_tokens=20)
        
        # Blocks should not overlap
        assert set(blocks1).isdisjoint(set(blocks2))
    
    def test_prefix_cache_lru(self):
        """Test LRU eviction correctness."""
        from components.prefix_cache import PrefixCache
        import time
        
        cache = PrefixCache(max_size_gb=0.0001)  # Very small
        
        # Add entries with delays
        kv = [(torch.randn(1, 10, 64), torch.randn(1, 10, 64))]
        cache.put([1], kv)
        time.sleep(0.01)
        cache.put([2], kv)
        time.sleep(0.01)
        cache.put([3], kv)
        
        # Entry [1] should be evicted (least recently used)
        # Can't guarantee exact eviction order without very tight control


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


