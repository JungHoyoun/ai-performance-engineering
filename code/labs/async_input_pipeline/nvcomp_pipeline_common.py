"""Shared nvCOMP input pipeline benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass(frozen=True)
class NvcompPipelineConfig:
    batch_size: int = 256
    seq_len: int = 2048
    vocab_size: int = 200_000
    dataset_size: int = 512
    compression_level: int = 3
    hidden_size: int = 1024

    @property
    def tokens_per_iteration(self) -> int:
        return int(self.batch_size * self.seq_len)


class NvcompInputPipelineBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark that compares CPU vs GPU nvCOMP decompression paths."""

    def __init__(self, cfg: NvcompPipelineConfig, *, label: str, use_gpu_decode: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.label = label
        self.use_gpu_decode = use_gpu_decode
        self._compressed_batches: List[bytes] = []
        self._batch_index = 0
        self._uncompressed_bytes = 0
        self._zstd = None
        self._zstd_compressor = None
        self._zstd_decompressor = None
        self._cupy = None
        self._nvcomp = None
        self._nvcomp_decompressor = None
        self.embedding: Optional[nn.Embedding] = None
        self.proj: Optional[nn.Linear] = None
        self.output: Optional[torch.Tensor] = None
        self._last_tokens: Optional[torch.Tensor] = None
        self.register_workload_metadata(tokens_per_iteration=float(self.cfg.tokens_per_iteration))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("nvCOMP input pipeline requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        try:
            import zstandard as zstd  # type: ignore
        except Exception as exc:
            raise RuntimeError("nvCOMP pipeline requires the zstandard package") from exc
        self._zstd = zstd
        self._zstd_compressor = zstd.ZstdCompressor(level=self.cfg.compression_level)
        self._zstd_decompressor = zstd.ZstdDecompressor()

        if self.use_gpu_decode:
            try:
                import cupy  # type: ignore
                from cupy.cuda import nvcomp  # type: ignore
            except Exception as exc:
                raise RuntimeError("nvCOMP GPU decode requires CuPy with nvcomp support") from exc
            if not hasattr(nvcomp, "ZstdDecompressor"):
                raise RuntimeError("cupy.cuda.nvcomp must expose ZstdDecompressor")
            self._cupy = cupy
            self._nvcomp = nvcomp
            self._nvcomp_decompressor = nvcomp.ZstdDecompressor()

        rng = np.random.default_rng(42)
        self._compressed_batches = []
        self._batch_index = 0
        for _ in range(self.cfg.dataset_size):
            tokens = rng.integers(
                0,
                self.cfg.vocab_size,
                size=(self.cfg.batch_size, self.cfg.seq_len),
                dtype=np.int32,
            )
            if self._uncompressed_bytes == 0:
                self._uncompressed_bytes = int(tokens.nbytes)
            self._compressed_batches.append(self._zstd_compressor.compress(tokens.tobytes()))

        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size).to(self.device, dtype=torch.float16)
        self.proj = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, bias=False).to(self.device, dtype=torch.float16)

        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.embedding is None or self.proj is None:
            raise RuntimeError("Benchmark not initialized")
        if not self._compressed_batches:
            raise RuntimeError("Compressed dataset not initialized")

        blob = self._next_blob()
        with self._nvtx_range(self.label):
            tokens = (
                self._decompress_gpu(blob)
                if self.use_gpu_decode
                else self._decompress_cpu(blob)
            )
            self._last_tokens = tokens
            with torch.no_grad():
                emb = self.embedding(tokens)
                pooled = emb.mean(dim=1)
                self.output = self.proj(pooled)
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._last_tokens is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:128]
        parameter_count = 0
        if self.embedding is not None:
            parameter_count += self.embedding.weight.numel()
        if self.proj is not None:
            parameter_count += self.proj.weight.numel()
        self._set_verification_payload(
            inputs={"tokens": self._last_tokens},
            output=verify_output.detach().clone(),
            batch_size=int(self.cfg.batch_size),
            parameter_count=parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "seq_len": self.cfg.seq_len,
                "vocab_size": self.cfg.vocab_size,
                "compression": "zstd",
                "gpu_decode": self.use_gpu_decode,
            },
        )

    def teardown(self) -> None:
        self._compressed_batches = []
        self._batch_index = 0
        self._zstd = None
        self._zstd_compressor = None
        self._zstd_decompressor = None
        self._cupy = None
        self._nvcomp = None
        self._nvcomp_decompressor = None
        self.embedding = None
        self.proj = None
        self.output = None
        self._last_tokens = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            adaptive_iterations=False,
            measurement_timeout_seconds=180,
            use_subprocess=False,
        )

    def _next_blob(self) -> bytes:
        blob = self._compressed_batches[self._batch_index]
        self._batch_index = (self._batch_index + 1) % len(self._compressed_batches)
        return blob

    def _decompress_cpu(self, blob: bytes) -> torch.Tensor:
        if self._zstd_decompressor is None:
            raise RuntimeError("zstandard decompressor not initialized")
        raw = self._zstd_decompressor.decompress(blob, max_output_size=self._uncompressed_bytes)
        tokens = np.frombuffer(raw, dtype=np.int32)
        tokens = tokens.reshape(self.cfg.batch_size, self.cfg.seq_len)
        return torch.from_numpy(tokens).to(self.device, dtype=torch.long, non_blocking=True)

    def _decompress_gpu(self, blob: bytes) -> torch.Tensor:
        if self._cupy is None or self._nvcomp_decompressor is None:
            raise RuntimeError("nvCOMP decompressor not initialized")
        comp_np = np.frombuffer(blob, dtype=np.uint8)
        comp_gpu = self._cupy.asarray(comp_np)
        decompressed = self._nvcomp_decompressor.decompress(comp_gpu, self._uncompressed_bytes)
        decompressed = decompressed.view(self._cupy.int32)
        self._cupy.cuda.get_current_stream().synchronize()
        tokens = torch.utils.dlpack.from_dlpack(decompressed.toDlpack())
        tokens = tokens.reshape(self.cfg.batch_size, self.cfg.seq_len)
        return tokens.to(self.device, dtype=torch.long, non_blocking=True)
