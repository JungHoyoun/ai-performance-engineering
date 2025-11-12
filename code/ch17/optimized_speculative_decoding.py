"""optimized_speculative_decoding.py - Optimized speculative decoding in inference/profiling.

Demonstrates speculative decoding for parallel token generation.
Speculative decoding: Uses draft model to predict multiple tokens in parallel.
Accepts/rejects tokens based on target model verification for speedup.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from common.python.compile_utils import compile_model

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")

class OptimizedSpeculativeDecodingBenchmark(Benchmark):
    """Optimized: Speculative decoding with draft/target coordination."""
    
    def __init__(self):
        self.device = resolve_device()
        self.embedding = None
        self.target_decoder = None
        self.draft_decoder = None
        self.output_head = None
        self.input_ids = None
        self.target_hidden = None
        self.draft_hidden = None
        self.prev_tokens = None
        self.current_lengths = None
        self.target_lengths = None
        self.max_length = 64
        self.speculative_length = 4
        self.hidden_dim = 512
        self.vocab_size = 16000
        self.batch_size = 8
        self.seq_len = 64
        self.target_layers = 1
        self.draft_layers = 1
    
    def setup(self) -> None:
        """Setup: Initialize target/draft models and seed hidden states."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, device=self.device, dtype=dtype)
        self.target_decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.target_layers,
            batch_first=True,
        ).to(self.device, dtype=dtype).eval()
        self.draft_decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.draft_layers,
            batch_first=True,
        ).to(self.device, dtype=dtype).eval()
        self.output_head = nn.Linear(self.hidden_dim, self.vocab_size, device=self.device, dtype=dtype)

        self.target_decoder = compile_model(
            self.target_decoder,
            mode="reduce-overhead",
        )
        self.draft_decoder = compile_model(
            self.draft_decoder,
            mode="reduce-overhead",
        )
        self.output_head = compile_model(
            self.output_head,
            mode="reduce-overhead",
        )

        self.input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
            dtype=torch.long,
        )
        self.prev_tokens = self.input_ids[:, -1:].clone()
        self.current_lengths = torch.full(
            (self.batch_size,),
            self.seq_len,
            device=self.device,
            dtype=torch.int32,
        )
        self.target_lengths = self.current_lengths + self.max_length
        prompt_embeds = self.embedding(self.input_ids)
        self.target_hidden = torch.zeros(
            self.target_layers,
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=dtype,
        )
        self.draft_hidden = torch.zeros(
            self.draft_layers,
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=dtype,
        )
        with torch.no_grad():
            _, self.target_hidden = self.target_decoder(prompt_embeds, self.target_hidden)
            _, self.draft_hidden = self.draft_decoder(prompt_embeds, self.draft_hidden)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Speculative decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                prev_tokens = self.prev_tokens.clone()
                target_hidden = self.target_hidden.clone()
                draft_hidden = self.draft_hidden.clone()

                while bool((self.current_lengths < self.target_lengths).any()):
                    proposals: list[torch.Tensor] = []
                    proposal_states: list[torch.Tensor] = []
                    running_hidden = draft_hidden
                    token_seed = prev_tokens

                    for _ in range(self.speculative_length):
                        draft_embed = self.embedding(token_seed)
                        draft_out, running_hidden = self.draft_decoder(draft_embed, running_hidden)
                        logits = self.output_head(draft_out[:, -1, :])
                        next_token = logits.argmax(dim=-1, keepdim=True)
                        proposals.append(next_token)
                        proposal_states.append(running_hidden.clone())
                        token_seed = next_token

                    proposal_tensor = torch.cat(proposals, dim=1)
                    proposal_embeds = self.embedding(proposal_tensor)
                    target_seq, seq_hidden = self.target_decoder(proposal_embeds, target_hidden)
                    target_logits = self.output_head(target_seq)
                    target_choice = target_logits.argmax(dim=-1, keepdim=True)

                    active = (self.current_lengths < self.target_lengths)
                    new_prev = prev_tokens.clone()

                    for idx in range(self.speculative_length):
                        if not active.any():
                            break
                        proposal = proposal_tensor[:, idx:idx + 1]
                        choice = target_choice[:, idx:idx + 1]
                        matches = (proposal == choice).squeeze(1)

                        accept_mask = active & matches
                        fallback_mask = active & (~matches)

                        accept_idx = torch.nonzero(accept_mask, as_tuple=False).flatten()
                        if accept_idx.numel() > 0:
                            self.current_lengths[accept_idx] += 1
                            new_prev.index_copy_(0, accept_idx, proposal[accept_idx])
                            draft_hidden[:, accept_idx, :] = proposal_states[idx][:, accept_idx, :]
                            target_hidden[:, accept_idx, :] = target_seq[accept_idx, idx, :].unsqueeze(0)

                        fallback_idx = torch.nonzero(fallback_mask, as_tuple=False).flatten()
                        if fallback_idx.numel() > 0:
                            fallback_tokens = choice[fallback_idx, idx, :]
                            new_prev.index_copy_(0, fallback_idx, fallback_tokens)
                            self.current_lengths[fallback_idx] += 1
                            target_slice = target_seq[fallback_idx, idx, :].unsqueeze(0)
                            target_hidden[:, fallback_idx, :] = target_slice
                            draft_hidden[:, fallback_idx, :] = target_slice
                            active[fallback_idx] = False

                        active = active & matches

                    active_idx = torch.nonzero(active, as_tuple=False).flatten()
                    if active_idx.numel() > 0:
                        draft_hidden[:, active_idx, :] = proposal_states[-1][:, active_idx, :]
                        target_hidden[:, active_idx, :] = seq_hidden[:, active_idx, :]
                        new_prev[active_idx] = proposal_tensor[active_idx, -1].unsqueeze(1)

                    prev_tokens = new_prev

                self.current_lengths = torch.minimum(self.current_lengths, self.target_lengths)
                self.prev_tokens = prev_tokens
        torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.target_decoder = None
        self.draft_decoder = None
        self.embedding = None
        self.output_head = None
        self.input_ids = None
        self.target_hidden = None
        self.draft_hidden = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.target_decoder is None or self.draft_decoder is None or self.output_head is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedSpeculativeDecodingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: speculative_decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
