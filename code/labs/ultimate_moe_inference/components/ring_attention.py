"""Ring Attention for distributed long-sequence inference.

Distributes attention computation across GPUs in a ring topology,
enabling very long context lengths without memory overflow.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist


class RingAttention:
    """Distributed attention across GPUs in a ring.
    
    Each GPU holds a chunk of the KV cache. Query blocks rotate
    around the ring, with each GPU computing partial attention
    against its local KV chunk.
    
    This enables context lengths that would OOM on a single GPU
    by distributing memory across the ring.
    
    Memory: O(seq_len / world_size) per GPU instead of O(seq_len)
    
    Example:
        ring_attn = RingAttention(world_size=8, chunk_size=8192)
        
        # Each GPU has its local KV chunk
        output = ring_attn.forward(q, k_local, v_local, group)
    """
    
    def __init__(
        self,
        world_size: int,
        chunk_size: int = 8192,
        causal: bool = True,
    ):
        """Initialize ring attention.
        
        Args:
            world_size: Number of GPUs in ring
            chunk_size: Tokens per GPU chunk
            causal: Apply causal masking
        """
        self.world_size = world_size
        self.chunk_size = chunk_size
        self.causal = causal
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Compute attention with ring exchange.
        
        Args:
            q: Query tensor [batch, heads, seq_q, head_dim]
            k: Local key tensor [batch, heads, seq_k, head_dim]
            v: Local value tensor [batch, heads, seq_k, head_dim]
            group: Process group for communication
            
        Returns:
            Attention output [batch, heads, seq_q, head_dim]
        """
        if self.world_size == 1 or group is None:
            return self._local_attention(q, k, v)
        
        rank = dist.get_rank(group)
        
        # Initialize accumulators for online softmax
        # out = sum(exp(scores - max_score) * v) / sum(exp(scores - max_score))
        batch, heads, seq_q, head_dim = q.shape
        seq_k = k.shape[2]
        
        # Accumulators
        out = torch.zeros_like(q)
        lse = torch.full((batch, heads, seq_q), float('-inf'), device=q.device)
        
        # Current KV to process
        current_k = k.clone()
        current_v = v.clone()
        
        # Ring exchange buffers
        send_k = torch.empty_like(k)
        send_v = torch.empty_like(v)
        recv_k = torch.empty_like(k)
        recv_v = torch.empty_like(v)
        
        for step in range(self.world_size):
            # Compute attention with current KV chunk
            chunk_rank = (rank - step) % self.world_size
            
            # Apply causal mask if needed
            if self.causal:
                # Only attend to chunks from previous or current position
                if chunk_rank > rank:
                    # This chunk is "future" - skip
                    pass
                else:
                    partial_out, partial_lse = self._attention_with_lse(
                        q, current_k, current_v
                    )
                    out, lse = self._online_softmax_update(
                        out, lse, partial_out, partial_lse
                    )
            else:
                partial_out, partial_lse = self._attention_with_lse(
                    q, current_k, current_v
                )
                out, lse = self._online_softmax_update(
                    out, lse, partial_out, partial_lse
                )
            
            # Ring exchange: send to next, receive from previous
            if step < self.world_size - 1:
                send_k.copy_(current_k)
                send_v.copy_(current_v)
                
                # Send to rank+1, receive from rank-1
                next_rank = (rank + 1) % self.world_size
                prev_rank = (rank - 1) % self.world_size
                
                send_ops = [
                    dist.P2POp(dist.isend, send_k, next_rank, group),
                    dist.P2POp(dist.isend, send_v, next_rank, group),
                    dist.P2POp(dist.irecv, recv_k, prev_rank, group),
                    dist.P2POp(dist.irecv, recv_v, prev_rank, group),
                ]
                
                reqs = dist.batch_isend_irecv(send_ops)
                for req in reqs:
                    req.wait()
                
                current_k.copy_(recv_k)
                current_v.copy_(recv_v)
        
        return out
    
    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Standard local attention without distribution."""
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
        )
    
    def _attention_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention and log-sum-exp for online softmax.
        
        Returns:
            Tuple of (weighted_v, log_sum_exp)
        """
        scale = q.shape[-1] ** -0.5
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq_q, seq_k]
        
        # Log-sum-exp for numerical stability
        max_score = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_score)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        lse = max_score.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
        
        # Weighted values
        attn_weights = exp_scores / sum_exp
        weighted_v = torch.matmul(attn_weights, v)
        
        return weighted_v, lse
    
    def _online_softmax_update(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        new_out: torch.Tensor,
        new_lse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update running output using online softmax.
        
        Combines partial attention outputs from different KV chunks
        while maintaining numerical stability.
        """
        # Compute new max
        max_lse = torch.maximum(lse, new_lse)
        
        # Rescale old and new contributions
        old_scale = torch.exp(lse - max_lse).unsqueeze(-1)
        new_scale = torch.exp(new_lse - max_lse).unsqueeze(-1)
        
        # Update output
        updated_out = old_scale * out + new_scale * new_out
        
        # Update log-sum-exp
        updated_lse = max_lse + torch.log(
            torch.exp(lse - max_lse) + torch.exp(new_lse - max_lse)
        )
        
        # Normalize output
        updated_out = updated_out / (old_scale + new_scale)
        
        return updated_out, updated_lse


def ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Convenience function for ring attention.
    
    Args:
        q: Query tensor [batch, heads, seq_q, head_dim]
        k: Local key tensor [batch, heads, seq_k, head_dim]
        v: Local value tensor [batch, heads, seq_k, head_dim]
        world_size: Number of GPUs
        group: Process group
        causal: Apply causal masking
        
    Returns:
        Attention output
    """
    ring = RingAttention(world_size, causal=causal)
    return ring.forward(q, k, v, group)


