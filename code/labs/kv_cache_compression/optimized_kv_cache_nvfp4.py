"""Optimized KV-cache benchmark switching to NVFP4 block scaling when available."""

from __future__ import annotations

import sys
from typing import Optional

import torch

from labs.kv_cache_compression.baseline_kv_cache import BaselineKVCacheBenchmark, TE_AVAILABLE, TE_IMPORT_ERROR

if TE_AVAILABLE:
    from transformer_engine.pytorch import autocast as te_autocast, is_nvfp4_available
    from transformer_engine.common import recipe as te_recipe
else:  # pragma: no cover
    te_autocast = is_nvfp4_available = te_recipe = None  # type: ignore


class OptimizedKVCacheNVFP4Benchmark(BaselineKVCacheBenchmark):
    """Calibrate in FP8 and then run NVFP4 for KV-cache heavy attention."""

    def __init__(self) -> None:
        super().__init__()
        self.nvfp4_recipe = (
            te_recipe.NVFP4BlockScaling(calibration_steps=20, amax_history_len=16, fp4_tensor_block=16)
            if TE_AVAILABLE
            else None
        )
        self.nvfp4_active = False
        self._nvfp4_skip_reason: Optional[str] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        preferred_recipe = self.fp8_recipe
        # Use NVFP4 when available; otherwise fall back to the baseline recipe.
        if TE_AVAILABLE and self.nvfp4_recipe is not None and is_nvfp4_available():
            preferred_recipe = self.nvfp4_recipe
        elif self.nvfp4_recipe is not None:
            self._nvfp4_skip_reason = (
                f"Transformer Engine not available: {TE_IMPORT_ERROR}"
                if not TE_AVAILABLE
                else "NVFP4 kernels unavailable on this hardware/driver."
            )
            print(f"[NVFP4] Falling back to FP8 recipe: {self._nvfp4_skip_reason}", file=sys.stderr, flush=True)

        try:
            self._setup_with_recipe(preferred_recipe)
            self.nvfp4_active = preferred_recipe is self.nvfp4_recipe
        except Exception as exc:
            if preferred_recipe is self.nvfp4_recipe and self._fallback_recipe is not None:
                self._nvfp4_skip_reason = f"NVFP4 setup failed: {exc}"
                print(f"[NVFP4] Falling back to FP8 recipe: {self._nvfp4_skip_reason}", file=sys.stderr, flush=True)
                self._setup_with_recipe(self._fallback_recipe)
                self.nvfp4_active = False
            else:
                raise

    def validate_result(self) -> Optional[str]:
        return super().validate_result()

    def benchmark_fn(self) -> None:
        if self.model is None or self.cache is None:
            raise RuntimeError("Benchmark not initialized")
        reset_cache(self.cache)
        offset = 0
        recipe = self.runtime_recipe if self.nvfp4_active else self.fp8_recipe
        if recipe is None:
            raise RuntimeError("No NVFP4/FP8 recipe available for benchmark")
        with te_autocast(enabled=True, recipe=recipe):
            for prefill in self.prefill_inputs:
                _ = self.model(prefill, self.cache, offset)
                offset += prefill.shape[1]
            for decode in self.decode_inputs:
                _ = self.model(decode, self.cache, offset)
                offset += decode.shape[1]
        torch.cuda.synchronize()
        if self.cache and self.cache.kv is not None:
            view = self.cache.kv[0, :, :, : min(1, self.cache.kv.shape[3]), : min(8, self.cache.kv.shape[4])]
            self.output = view.detach().float().clone()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._set_verification_payload(
            inputs={
                "batch_size": torch.tensor([self.batch_size], dtype=torch.int64, device="cpu"),
                "seq_meta": torch.tensor(
                    [self.prefill_seq, self.decode_seq, self.decode_steps], dtype=torch.int64, device="cpu"
                ),
            },
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
            precision_flags={
                "fp16": False,
                "bf16": self.tensor_dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVFP4-specific metrics."""
        metrics = super().get_custom_metrics() or {}
        metrics.update({
            "kv_cache.nvfp4_active": 1.0 if self.nvfp4_active else 0.0,
            "kv_cache.compression_ratio": 4.0 if self.nvfp4_active else 2.0,  # NVFP4=4x, FP8=2x
        })
        return metrics

    def get_optimization_goal(self) -> str:
        """Memory optimization - lower memory usage is better."""
        return "memory"

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        total_tokens = self.prefill_seq + self.decode_seq * self.decode_steps
        return {
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "prefill_seq": self.prefill_seq,
            "decode_seq": self.decode_seq,
            "decode_steps": self.decode_steps,
            "nvfp4": True,
            "shapes": {
                "prefill": (self.batch_size, self.prefill_seq // 2, self.hidden_dim),
                "decode": (self.batch_size, self.decode_seq, self.hidden_dim),
                "cache": (
                    2,
                    self.batch_size,
                    self.num_heads,
                    total_tokens,
                    self.hidden_dim // self.num_heads,
                ),
            },
            "dtypes": {"activations": str(self.tensor_dtype)},
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaselineKVCacheBenchmark:
    return OptimizedKVCacheNVFP4Benchmark()
