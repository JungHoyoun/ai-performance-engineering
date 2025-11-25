"""Speculative Decoding for accelerated inference (Ch18).

Implements draft-and-verify pattern:
- Draft model generates k tokens speculatively
- Target model verifies in parallel
- Accepted tokens avoid redundant computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SpeculativeOutput:
    """Output from speculative decoding."""
    
    tokens: torch.Tensor
    num_accepted: int
    num_generated: int
    acceptance_rate: float


class NGramSpeculator:
    """N-gram based speculation (no draft model needed).
    
    Uses patterns from the context to predict likely continuations.
    Useful when loading a separate draft model is not feasible.
    """
    
    def __init__(self, n: int = 3, max_cache_size: int = 10000):
        """Initialize n-gram speculator.
        
        Args:
            n: N-gram size
            max_cache_size: Maximum entries in cache
        """
        self.n = n
        self.max_cache_size = max_cache_size
        self.cache: Dict[Tuple[int, ...], List[int]] = {}
    
    def build_cache(self, token_ids: List[int]) -> None:
        """Build n-gram cache from token sequence.
        
        Args:
            token_ids: Sequence of token IDs
        """
        for i in range(len(token_ids) - self.n):
            key = tuple(token_ids[i:i + self.n - 1])
            next_token = token_ids[i + self.n - 1]
            
            if key not in self.cache:
                self.cache[key] = []
            
            if next_token not in self.cache[key]:
                self.cache[key].append(next_token)
        
        # Prune if too large
        if len(self.cache) > self.max_cache_size:
            # Keep most recently added
            self.cache = dict(list(self.cache.items())[-self.max_cache_size:])
    
    def predict(self, context: torch.Tensor, k: int) -> torch.Tensor:
        """Predict k tokens using n-gram lookup.
        
        Args:
            context: Context tokens [1, seq_len]
            k: Number of tokens to predict
            
        Returns:
            Predicted tokens [1, k]
        """
        device = context.device
        current = context[0].tolist()
        predictions = []
        
        for _ in range(k):
            key = tuple(current[-(self.n - 1):])
            
            if key in self.cache and self.cache[key]:
                next_token = self.cache[key][0]
            else:
                # No prediction available
                next_token = 0
            
            predictions.append(next_token)
            current.append(next_token)
        
        return torch.tensor([predictions], device=device)


class DraftModelSpeculator:
    """Draft model based speculation.
    
    Uses a smaller, faster model to generate draft tokens
    that are verified by the target model.
    """
    
    def __init__(
        self,
        draft_model: nn.Module,
        tokenizer: Any,
        speculation_length: int = 4,
    ):
        """Initialize draft model speculator.
        
        Args:
            draft_model: Smaller model for drafting
            tokenizer: Tokenizer
            speculation_length: Number of tokens to draft
        """
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.k = speculation_length
        
        self.draft_model.eval()
    
    def predict(self, context: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """Generate k draft tokens.
        
        Args:
            context: Context tokens [batch, seq_len]
            k: Number of tokens to generate (default: self.k)
            
        Returns:
            Draft tokens [batch, k]
        """
        k = k or self.k
        
        with torch.no_grad():
            outputs = self.draft_model.generate(
                context,
                max_new_tokens=k,
                do_sample=False,
                use_cache=True,
            )
        
        return outputs[:, context.shape[1]:]


class SpeculativeDecoder:
    """Main speculative decoding engine (Ch18).
    
    Implements the draft-and-verify algorithm:
    1. Draft model generates k tokens
    2. Target model computes logits for all positions
    3. Verify each token: accept if matches target prediction
    4. Continue from first mismatch
    
    This achieves up to kx speedup when acceptance rate is high.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        speculator: Optional[Any] = None,
        speculation_length: int = 4,
    ):
        """Initialize speculative decoder.
        
        Args:
            target_model: Main (large) model
            speculator: NGramSpeculator or DraftModelSpeculator
            speculation_length: Default speculation length
        """
        self.target_model = target_model
        self.speculator = speculator or NGramSpeculator()
        self.k = speculation_length
        
        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> SpeculativeOutput:
        """Generate tokens using speculative decoding.
        
        Args:
            input_ids: Input token IDs [batch, seq]
            max_new_tokens: Maximum tokens to generate
            attention_mask: Attention mask
            
        Returns:
            SpeculativeOutput with generated tokens and statistics
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        if batch_size > 1:
            raise ValueError("Speculative decoding only supports batch_size=1")
        
        generated = input_ids.clone()
        total_accepted = 0
        total_generated = 0
        
        # Build n-gram cache from prompt if using NGramSpeculator
        if isinstance(self.speculator, NGramSpeculator):
            self.speculator.build_cache(input_ids[0].tolist())
        
        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            # Generate draft tokens
            draft_tokens = self.speculator.predict(generated, self.k)
            
            # Verify with target model
            num_accepted, verified = self._verify_tokens(
                generated, draft_tokens
            )
            
            # Append verified tokens
            generated = torch.cat([generated, verified], dim=-1)
            
            total_accepted += num_accepted
            total_generated += self.k
            
            # Update n-gram cache
            if isinstance(self.speculator, NGramSpeculator):
                self.speculator.build_cache(verified[0].tolist())
        
        # Trim to exact length
        final_length = input_ids.shape[1] + max_new_tokens
        generated = generated[:, :final_length]
        
        # Update global stats
        self.total_tokens += total_generated
        self.accepted_tokens += total_accepted
        
        return SpeculativeOutput(
            tokens=generated,
            num_accepted=total_accepted,
            num_generated=total_generated,
            acceptance_rate=total_accepted / max(total_generated, 1),
        )
    
    def _verify_tokens(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Verify draft tokens against target model.
        
        Args:
            context: Current context [1, seq]
            draft_tokens: Draft tokens to verify [1, k]
            
        Returns:
            Tuple of (num_accepted, verified_tokens)
        """
        # Concatenate for parallel verification
        full_input = torch.cat([context, draft_tokens], dim=-1)
        
        # Get target logits
        with torch.no_grad():
            outputs = self.target_model(full_input)
            logits = outputs.logits
        
        # Get target predictions for draft positions
        start = context.shape[1] - 1
        end = start + draft_tokens.shape[1]
        
        target_preds = logits[:, start:end].argmax(dim=-1)
        
        # Compare with draft
        num_accepted = 0
        for i in range(draft_tokens.shape[1]):
            if i >= target_preds.shape[1]:
                break
            if target_preds[0, i] == draft_tokens[0, i]:
                num_accepted += 1
            else:
                break
        
        # Return accepted tokens + correction
        if num_accepted < draft_tokens.shape[1] and num_accepted < target_preds.shape[1]:
            # Include correction token
            verified = torch.cat([
                draft_tokens[:, :num_accepted],
                target_preds[:, num_accepted:num_accepted + 1],
            ], dim=-1)
        else:
            verified = draft_tokens[:, :num_accepted]
        
        return num_accepted, verified
    
    def get_stats(self) -> Dict[str, float]:
        """Get speculation statistics.
        
        Returns:
            Dictionary with acceptance rate and efficiency
        """
        return {
            "total_tokens_speculated": self.total_tokens,
            "total_tokens_accepted": self.accepted_tokens,
            "acceptance_rate": self.accepted_tokens / max(self.total_tokens, 1),
            "speedup_factor": (self.accepted_tokens + 1) / max(self.total_tokens / self.k, 1),
        }
    
    def reset_stats(self) -> None:
        """Reset speculation statistics."""
        self.total_tokens = 0
        self.accepted_tokens = 0


