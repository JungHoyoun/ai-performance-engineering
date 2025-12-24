"""Optimized FlexDecoding benchmark using Flash SDPA on sliding-window slices."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingBenchmark(FlexDecodingHarness):
    """Optimized path: Flash SDPA with sliding-window cache slicing."""

    def __init__(self) -> None:
        super().__init__(
            use_flex_attention=False,
            require_flex=False,
            decode_tokens=512,
            compile_enabled=False,
        )

    def setup(self) -> None:
        super().setup()
        window = self.config.window
        if window <= 0:
            raise RuntimeError("Sliding-window size must be positive")

    def _decode_step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Windowed decode not initialized")
        window = self.config.window
        start = position - window
        if start < 0:
            raise RuntimeError("Windowed decode expects position >= window size")
        end = position + 1
        q, k, v = self.model._project_token(token)
        self.model._update_cache(k, v, position)
        self.model._set_offset(position)
        k_slice = self.model.k_cache[:, start:end]
        v_slice = self.model.v_cache[:, start:end]
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k_slice.transpose(1, 2),
                v_slice.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        return self.model.o_proj(out.transpose(1, 2).reshape(token.shape[0], 1, self.config.dim))

def get_benchmark():
    return OptimizedFlexDecodingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
