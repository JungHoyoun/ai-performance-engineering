"""Optimized FlexDecoding benchmark using Flash SDPA on sliding-window slices."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        window = self.config.window
        if window <= 0:
            raise RuntimeError("Sliding-window size must be positive")

    def _decode_windowed(self, token: torch.Tensor, position: int) -> torch.Tensor:
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
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_slice.transpose(1, 2),
            v_slice.transpose(1, 2),
            attn_mask=None,
            is_causal=False,
        )
        return self.model.o_proj(out.transpose(1, 2).reshape(token.shape[0], 1, self.config.dim))

    def benchmark_fn(self):
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")

        prefill_times = []
        decode_times = []
        base_position = self.prefill_tokens.size(1)
        window = self.config.window
        if base_position < window:
            raise RuntimeError("Prefill length must be >= sliding window size")

        with torch.no_grad():
            with self._nvtx_range("flex_prefill"):
                start = time.perf_counter()
                prefill_out = self.model.prefill(self.prefill_tokens)
                torch.cuda.synchronize(self.device)
                prefill_times.append((time.perf_counter() - start) * 1000.0)

            with self._nvtx_range("flex_decode"):
                for pos in range(self.decode_tokens):
                    start = time.perf_counter()
                    decode_out = self._decode_windowed(self.decode_token, base_position + pos)
                    torch.cuda.synchronize(self.device)
                    decode_times.append((time.perf_counter() - start) * 1000.0)

        self._last_output = decode_out if "decode_out" in dir() else prefill_out
        self._history["prefill_ms"].extend(prefill_times)
        self._history["decode_ms"].extend(decode_times)
        if self._last_output is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return {"prefill_ms": prefill_times, "decode_ms": decode_times}

def get_benchmark():
    return OptimizedFlexDecodingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
