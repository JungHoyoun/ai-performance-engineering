"""Decode kernel builders shared by the bucketed decode demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODE = "reduce-overhead" if torch.cuda.is_available() else "default"


@dataclass
class DecodeKernel:
    """Light wrapper so callers can introspect the backend type."""

    fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    backend: str

    def __call__(
        self, tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.fn(tokens, kv, mask)


class VLLMDecodeKernel:
    """
    Small PagedAttention-backed decode step.

    Uses vLLM's fused paged attention kernel for efficient decode.
    """

    def __init__(self, hidden: int, max_batch: int = 32, device: str = DEVICE) -> None:
        from vllm.attention.ops.paged_attn import PagedAttention

        self.device = device
        self.hidden = hidden
        self.num_heads = 1
        self.head_size = hidden
        self.kv_cache_dtype = "auto"
        self.scale = 1.0 / math.sqrt(float(self.head_size))

        self.max_batch = max_batch
        # vLLM paged attention requires block_size >= 8 (typically 8, 16, or 32)
        self.block_size = 16
        self.num_blocks = 4
        self.max_seq_len = self.num_blocks * self.block_size  # Max sequence length

        # Preallocate KV cache
        kv_shape = PagedAttention.get_kv_cache_shape(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_kv_heads=self.num_heads,
            head_size=self.head_size,
            cache_dtype_str=self.kv_cache_dtype,
        )
        self.kv_cache = torch.randn(kv_shape, device=self.device, dtype=torch.float16)
        self.key_cache, self.value_cache = PagedAttention.split_kv_cache(
            self.kv_cache, num_kv_heads=self.num_heads, head_size=self.head_size
        )

        # Block tables: each sequence gets assigned blocks
        self.block_tables = torch.arange(
            self.num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0).expand(self.max_batch, -1).contiguous()
        
        # Sequence lengths (tokens per sequence)
        self.seq_lens = torch.full(
            (self.max_batch,), self.block_size, dtype=torch.int32, device=self.device
        )
        
        # K/V scale factors (1.0 = no scaling) - REQUIRED by vLLM API
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        self._paged_attention = PagedAttention.forward_decode

    def ensure_capacity(self, batch: int) -> None:
        if batch <= self.max_batch:
            return
        # Resize block tables / seq_lens for a larger batch
        self.block_tables = torch.arange(
            self.num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0).expand(batch, -1).contiguous()
        
        self.seq_lens = torch.full(
            (batch,), self.block_size, dtype=torch.int32, device=self.device
        )
        self.max_batch = batch

    def __call__(
        self, tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch = tokens.size(0)
        self.ensure_capacity(batch)

        # Query shape: [batch, num_heads, head_size]
        query = tokens.view(batch, self.num_heads, self.head_size).to(torch.float16)
        
        out = self._paged_attention(
            query,
            self.key_cache,
            self.value_cache,
            self.block_tables[:batch],
            self.seq_lens[:batch],
            self.max_seq_len,      # max_seq_len (was missing!)
            self.kv_cache_dtype,
            self.num_heads,
            self.scale,
            None,                  # alibi_slopes
            self.k_scale,          # k_scale (required tensor, not None!)
            self.v_scale,          # v_scale (required tensor, not None!)
        )

        flat = out.view(batch, self.hidden)
        if mask is not None:
            flat = flat.masked_fill(~mask[:, None], float("-inf"))
        return flat

    @property
    def bytes(self) -> int:
        return self.kv_cache.numel() * self.kv_cache.element_size()


def _torch_decode(hidden: int) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """Fallback torch-based decode (for when vLLM unavailable)."""
    def _decode(tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = torch.tanh(tokens + kv)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask[:, None], float("-inf"))
        return attn_scores

    try:
        return torch.compile(_decode, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
    except Exception:
        return _decode


def build_decode_kernel(
    hidden: int,
    *,
    max_batch: int = 32,
    prefer_vllm: bool = True,
    device: str = DEVICE,
) -> DecodeKernel:
    """
    Build a vLLM-backed decode kernel. Raises if vLLM unavailable.
    """
    if prefer_vllm:
        # No fallback - fail fast if vLLM doesn't work
        kernel = VLLMDecodeKernel(hidden=hidden, max_batch=max_batch, device=device)
        return DecodeKernel(fn=kernel, backend="vllm")

    # Only use torch if explicitly requested (prefer_vllm=False)
    torch_kernel = _torch_decode(hidden)
    return DecodeKernel(fn=torch_kernel, backend="torch")
