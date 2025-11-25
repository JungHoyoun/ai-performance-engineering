"""Prefix Cache for KV state sharing (Ch16).

Caches KV states for common prompt prefixes to avoid
redundant computation (e.g., shared system prompts).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import torch


@dataclass
class CacheEntry:
    """An entry in the prefix cache."""
    
    prefix_hash: str
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]  # [(K, V) per layer]
    num_tokens: int
    
    # Statistics
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    @property
    def size_bytes(self) -> int:
        """Size of cached KV tensors in bytes."""
        total = 0
        for k, v in self.kv_cache:
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total
    
    def access(self) -> None:
        """Record an access."""
        self.hits += 1
        self.last_accessed = time.time()


class PrefixCache:
    """Cache KV states for common prompt prefixes.
    
    Key features:
    - Hash-based lookup for prefix matching
    - LRU eviction when cache is full
    - Support for hierarchical prefixes
    
    Common use case: System prompts shared across requests
    (e.g., "You are a helpful AI assistant...")
    
    Example:
        cache = PrefixCache(max_size_gb=10.0)
        
        # First request computes and caches
        kv = cache.get_or_compute(prefix_tokens, model)
        
        # Subsequent requests with same prefix hit cache
        kv = cache.get_or_compute(prefix_tokens, model)  # Cache hit!
    """
    
    def __init__(
        self,
        max_size_gb: float = 10.0,
        eviction_policy: str = "lru",
    ):
        """Initialize prefix cache.
        
        Args:
            max_size_gb: Maximum cache size in GB
            eviction_policy: "lru" or "lfu"
        """
        self.max_size_bytes = int(max_size_gb * 1e9)
        self.eviction_policy = eviction_policy
        
        self.cache: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, tokens: List[int]) -> str:
        """Compute hash for token sequence.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Hash string
        """
        token_bytes = bytes(str(tokens), 'utf-8')
        return hashlib.sha256(token_bytes).hexdigest()[:16]
    
    def get(self, prefix_tokens: List[int]) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get cached KV state if available.
        
        Args:
            prefix_tokens: Prefix token IDs
            
        Returns:
            Cached KV tensors or None
        """
        prefix_hash = self._compute_hash(prefix_tokens)
        
        if prefix_hash in self.cache:
            entry = self.cache[prefix_hash]
            entry.access()
            self.hits += 1
            
            # Clone tensors to avoid modification
            return [(k.clone(), v.clone()) for k, v in entry.kv_cache]
        
        self.misses += 1
        return None
    
    def put(
        self,
        prefix_tokens: List[int],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Store KV state in cache.
        
        Args:
            prefix_tokens: Prefix token IDs
            kv_cache: KV tensors per layer
        """
        prefix_hash = self._compute_hash(prefix_tokens)
        
        # Create entry
        entry = CacheEntry(
            prefix_hash=prefix_hash,
            kv_cache=[(k.clone(), v.clone()) for k, v in kv_cache],
            num_tokens=len(prefix_tokens),
        )
        
        # Check if we need to evict
        while self.current_size_bytes + entry.size_bytes > self.max_size_bytes:
            if not self._evict_one():
                break
        
        # Store entry
        self.cache[prefix_hash] = entry
        self.current_size_bytes += entry.size_bytes
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy.
        
        Returns:
            True if an entry was evicted
        """
        if not self.cache:
            return False
        
        # Find entry to evict
        if self.eviction_policy == "lfu":
            # Least frequently used
            victim_hash = min(self.cache.keys(), key=lambda h: self.cache[h].hits)
        else:
            # LRU
            victim_hash = min(self.cache.keys(), key=lambda h: self.cache[h].last_accessed)
        
        # Remove entry
        entry = self.cache.pop(victim_hash)
        self.current_size_bytes -= entry.size_bytes
        
        return True
    
    def get_or_compute(
        self,
        prefix_tokens: List[int],
        model: Any,
        device: Optional[torch.device] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV or compute and cache.
        
        Args:
            prefix_tokens: Prefix token IDs
            model: Model to compute KV
            device: Device for computation
            
        Returns:
            KV tensors per layer
        """
        # Try cache first
        cached = self.get(prefix_tokens)
        if cached is not None:
            return cached
        
        # Compute KV
        device = device or next(model.parameters()).device
        input_ids = torch.tensor([prefix_tokens], device=device)
        
        with torch.no_grad():
            outputs = model(
                input_ids,
                use_cache=True,
                return_dict=True,
            )
        
        # Extract KV cache
        past_key_values = outputs.past_key_values
        
        if past_key_values is None:
            return []
        
        # Convert to list of tuples
        kv_cache = [(kv[0], kv[1]) for kv in past_key_values]
        
        # Store in cache
        self.put(prefix_tokens, kv_cache)
        
        return kv_cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_requests = self.hits + self.misses
        
        return {
            "num_entries": len(self.cache),
            "size_gb": self.current_size_bytes / 1e9,
            "max_size_gb": self.max_size_bytes / 1e9,
            "utilization_pct": 100 * self.current_size_bytes / self.max_size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": 100 * self.hits / max(total_requests, 1),
        }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
    
    def __str__(self) -> str:
        """Return string representation."""
        stats = self.get_stats()
        return (
            f"PrefixCache(entries={stats['num_entries']}, "
            f"size={stats['size_gb']:.2f}GB, "
            f"hit_rate={stats['hit_rate_pct']:.1f}%)"
        )


