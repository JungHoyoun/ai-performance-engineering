"""FlashMLA - Multi-head Latent Attention (Ch18).

Implements memory-efficient attention using low-rank KV compression,
as used in DeepSeek-V2 and similar architectures.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashMLA(nn.Module):
    """Multi-head Latent Attention with memory-efficient KV compression.
    
    Key insight: KV cache can be compressed via low-rank projection
    without significant quality loss. This reduces memory by up to 8x.
    
    Architecture:
    - Project KV to low-rank latent space before caching
    - Expand back to full dimension during attention
    - Uses FlashAttention for compute-efficient attention
    
    Memory savings:
    - Standard: O(seq_len * num_heads * head_dim * 2)
    - MLA: O(seq_len * latent_dim)
    
    Example:
        mla = FlashMLA(hidden_dim=4096, num_heads=32, latent_dim=512)
        
        # Compress KV for caching
        latent_kv = mla.compress_kv(k, v)
        
        # Expand and compute attention
        output = mla.forward(q, latent_kv)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        latent_dim: int = 512,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Initialize FlashMLA.
        
        Args:
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            latent_dim: Dimension of compressed latent space
            head_dim: Per-head dimension (default: hidden_dim // num_heads)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.dropout = dropout
        
        # KV compression: project to low-rank latent
        self.kv_compress = nn.Linear(hidden_dim * 2, latent_dim, bias=False)
        
        # KV expansion: project back to full dimension
        self.kv_expand = nn.Linear(latent_dim, hidden_dim * 2, bias=False)
        
        # Q projection (standard)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.kv_compress, self.kv_expand, self.q_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def compress_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compress K and V to latent space for caching.
        
        Args:
            k: Key tensor [batch, seq, hidden_dim]
            v: Value tensor [batch, seq, hidden_dim]
            
        Returns:
            Compressed latent [batch, seq, latent_dim]
        """
        # Concatenate K and V
        kv = torch.cat([k, v], dim=-1)  # [batch, seq, hidden_dim * 2]
        
        # Project to latent
        return self.kv_compress(kv)  # [batch, seq, latent_dim]
    
    def expand_kv(
        self,
        latent_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand latent back to K and V.
        
        Args:
            latent_kv: Compressed latent [batch, seq, latent_dim]
            
        Returns:
            Tuple of (k, v) tensors
        """
        # Project back to full dimension
        kv = self.kv_expand(latent_kv)  # [batch, seq, hidden_dim * 2]
        
        # Split K and V
        k, v = kv.chunk(2, dim=-1)
        
        return k, v
    
    def forward(
        self,
        q: torch.Tensor,
        latent_kv: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Compute attention with compressed KV.
        
        Args:
            q: Query tensor [batch, seq_q, hidden_dim]
            latent_kv: Compressed KV [batch, seq_k, latent_dim]
            is_causal: Apply causal masking
            
        Returns:
            Attention output [batch, seq_q, hidden_dim]
        """
        batch_size, seq_q, _ = q.shape
        seq_k = latent_kv.shape[1]
        
        # Project Q
        q = self.q_proj(q)
        
        # Expand KV from latent
        k, v = self.expand_kv(latent_kv)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention (uses FlashAttention if available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_q, self.hidden_dim)
        
        return self.out_proj(attn_out)
    
    def get_compression_ratio(self) -> float:
        """Get memory compression ratio.
        
        Returns:
            Ratio of standard KV size to MLA size
        """
        standard_size = self.hidden_dim * 2
        mla_size = self.latent_dim
        return standard_size / mla_size


class MLAKVCache:
    """KV cache optimized for MLA (stores compressed latents).
    
    Instead of storing full K and V tensors, stores the compressed
    latent representation, reducing memory by the compression ratio.
    """
    
    def __init__(
        self,
        max_seq_len: int,
        latent_dim: int,
        num_layers: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize MLA KV cache.
        
        Args:
            max_seq_len: Maximum sequence length
            latent_dim: Latent dimension
            num_layers: Number of transformer layers
            device: Device for tensors
            dtype: Data type
        """
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate cache
        # Shape: [num_layers, max_seq_len, latent_dim]
        self.cache = torch.zeros(
            num_layers, max_seq_len, latent_dim,
            device=device, dtype=dtype
        )
        
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        latent_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Update cache with new latent and return full cache.
        
        Args:
            layer_idx: Layer index
            latent_kv: New latent KV [batch, new_seq, latent_dim]
            
        Returns:
            Full cached latent [batch, total_seq, latent_dim]
        """
        new_len = latent_kv.shape[1]
        
        # Store new latent
        self.cache[layer_idx, self.seq_len:self.seq_len + new_len] = latent_kv[0]
        
        if layer_idx == self.num_layers - 1:
            self.seq_len += new_len
        
        # Return full cache up to current position
        return self.cache[layer_idx, :self.seq_len].unsqueeze(0)
    
    def reset(self) -> None:
        """Reset cache."""
        self.seq_len = 0
        self.cache.zero_()
    
    def memory_usage_gb(self) -> float:
        """Get memory usage in GB."""
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        total_bytes = self.cache.numel() * bytes_per_element
        return total_bytes / 1e9


