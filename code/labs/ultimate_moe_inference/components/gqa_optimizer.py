"""GQA/MQA Optimization for KV cache efficiency.

Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) share
KV heads across multiple Q heads, reducing KV cache size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAOptimizer:
    """Optimizer for Grouped-Query Attention (GQA).
    
    GQA shares KV heads across multiple Q heads:
    - Standard MHA: num_kv_heads = num_q_heads
    - MQA: num_kv_heads = 1
    - GQA: num_kv_heads < num_q_heads (typically 1/8 or 1/4)
    
    Memory savings:
    - Standard: O(2 * num_heads * head_dim)
    - GQA: O(2 * num_kv_heads * head_dim)
    
    Example:
        optimizer = GQAOptimizer(num_q_heads=32, num_kv_heads=4)
        
        # Expand KV heads to match Q heads (memory-efficient)
        k_expanded, v_expanded = optimizer.expand_kv(k, v)
    """
    
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
    ):
        """Initialize GQA optimizer.
        
        Args:
            num_q_heads: Number of query heads
            num_kv_heads: Number of key/value heads
        """
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        
        self.is_mqa = num_kv_heads == 1
        self.is_gqa = num_kv_heads > 1 and num_kv_heads < num_q_heads
        self.is_mha = num_kv_heads == num_q_heads
    
    def expand_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand KV heads to match Q heads.
        
        Uses repeat_interleave for memory-efficient expansion that
        doesn't copy data when possible.
        
        Args:
            k: Key tensor [batch, num_kv_heads, seq, head_dim]
            v: Value tensor [batch, num_kv_heads, seq, head_dim]
            
        Returns:
            Expanded (k, v) tensors [batch, num_q_heads, seq, head_dim]
        """
        if self.is_mha:
            return k, v
        
        # Repeat KV heads to match Q heads
        k_expanded = k.repeat_interleave(self.group_size, dim=1)
        v_expanded = v.repeat_interleave(self.group_size, dim=1)
        
        return k_expanded, v_expanded
    
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Compute GQA-aware attention.
        
        Expands KV and computes attention in a memory-efficient way.
        
        Args:
            q: Query [batch, num_q_heads, seq_q, head_dim]
            k: Key [batch, num_kv_heads, seq_k, head_dim]
            v: Value [batch, num_kv_heads, seq_k, head_dim]
            is_causal: Apply causal mask
            
        Returns:
            Attention output [batch, num_q_heads, seq_q, head_dim]
        """
        # Expand KV heads
        k_expanded, v_expanded = self.expand_kv(k, v)
        
        # Use SDPA (leverages FlashAttention when available)
        return F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            is_causal=is_causal,
        )
    
    def get_kv_cache_size(
        self,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Calculate KV cache size in bytes.
        
        Args:
            seq_len: Sequence length
            head_dim: Per-head dimension
            num_layers: Number of layers
            dtype: Data type
            
        Returns:
            Size in bytes
        """
        bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        
        # 2 for K and V
        elements = 2 * num_layers * self.num_kv_heads * seq_len * head_dim
        
        return elements * bytes_per_elem
    
    def get_compression_ratio(self) -> float:
        """Get KV cache compression ratio vs standard MHA.
        
        Returns:
            Compression ratio (e.g., 8.0 for GQA with group_size=8)
        """
        return self.group_size
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.is_mqa:
            attn_type = "MQA"
        elif self.is_gqa:
            attn_type = f"GQA (group_size={self.group_size})"
        else:
            attn_type = "MHA"
        
        return f"GQAOptimizer({attn_type}): {self.num_q_heads} Q -> {self.num_kv_heads} KV"


class GQAKVCache:
    """KV cache optimized for GQA/MQA models.
    
    Stores only the required KV heads, not the full Q head count.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize GQA KV cache.
        
        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads (not Q heads!)
            head_dim: Per-head dimension
            max_seq_len: Maximum sequence length
            device: Device
            dtype: Data type
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate cache
        # Shape: [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
        self.cache = torch.zeros(
            num_layers, 2, max_seq_len, num_kv_heads, head_dim,
            device=device, dtype=dtype
        )
        
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full KV.
        
        Args:
            layer_idx: Layer index
            k: New keys [batch, num_kv_heads, new_len, head_dim]
            v: New values [batch, num_kv_heads, new_len, head_dim]
            
        Returns:
            Full cached (k, v)
        """
        batch_size = k.shape[0]
        new_len = k.shape[2]
        
        # Transpose for storage: [batch, heads, seq, dim] -> [seq, heads, dim]
        k_store = k[0].transpose(0, 1)  # [seq, heads, dim]
        v_store = v[0].transpose(0, 1)
        
        # Store
        self.cache[layer_idx, 0, self.seq_len:self.seq_len + new_len] = k_store
        self.cache[layer_idx, 1, self.seq_len:self.seq_len + new_len] = v_store
        
        if layer_idx == self.num_layers - 1:
            self.seq_len += new_len
        
        # Return full cache in expected format
        k_full = self.cache[layer_idx, 0, :self.seq_len].transpose(0, 1).unsqueeze(0)
        v_full = self.cache[layer_idx, 1, :self.seq_len].transpose(0, 1).unsqueeze(0)
        
        return k_full, v_full
    
    def reset(self) -> None:
        """Reset cache."""
        self.seq_len = 0
    
    def memory_usage_gb(self) -> float:
        """Get memory usage in GB."""
        bytes_per_elem = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        total_bytes = self.cache.numel() * bytes_per_elem
        return total_bytes / 1e9


def optimize_for_gqa(
    model: nn.Module,
    num_q_heads: int,
    num_kv_heads: int,
) -> GQAOptimizer:
    """Create GQA optimizer for a model.
    
    Args:
        model: Model to optimize
        num_q_heads: Number of Q heads
        num_kv_heads: Number of KV heads
        
    Returns:
        GQA optimizer instance
    """
    optimizer = GQAOptimizer(num_q_heads, num_kv_heads)
    
    print(f"GQA optimization: {optimizer}")
    print(f"  KV cache compression: {optimizer.get_compression_ratio():.1f}x")
    
    return optimizer


