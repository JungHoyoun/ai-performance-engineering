"""Structured sparsity (2:4) SwiGLU FFN helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from core.utils.structured_sparsity import prune_2_4


@dataclass(frozen=True)
class StructuredSparsityFFNConfig:
    batch_size: int = 4
    seq_len: int = 4096
    hidden_size: int = 2880
    ffn_size: int = 2880
    dtype: torch.dtype = torch.float16

    @property
    def tokens_per_iteration(self) -> int:
        return int(self.batch_size * self.seq_len)


@dataclass
class CusparseltAlgo:
    alg_id: int
    split_k: int
    split_k_mode: int


class StructuredSparsityFFN:
    """SwiGLU FFN with optional cuSPARSELt sparse GEMMs."""

    def __init__(self, cfg: StructuredSparsityFFNConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.input: Optional[torch.Tensor] = None
        self.input_t: Optional[torch.Tensor] = None
        self.w1: Optional[torch.Tensor] = None
        self.w3: Optional[torch.Tensor] = None
        self.w2: Optional[torch.Tensor] = None
        self.w1_compressed: Optional[torch.Tensor] = None
        self.w3_compressed: Optional[torch.Tensor] = None
        self.w2_compressed: Optional[torch.Tensor] = None
        self.w1_algo: Optional[CusparseltAlgo] = None
        self.w3_algo: Optional[CusparseltAlgo] = None
        self.w2_algo: Optional[CusparseltAlgo] = None

    def setup_dense(self) -> None:
        self._init_tensors()

    def setup_sparse(self) -> None:
        self._init_tensors()
        if not torch.backends.cusparselt.is_available():
            raise RuntimeError("cuSPARSELt is required for structured sparsity benchmarks")
        if not hasattr(torch._C, "_cusparselt"):
            raise RuntimeError("torch._C._cusparselt is required for cuSPARSELt mm_search")
        if self.input_t is None or self.w1 is None or self.w3 is None or self.w2 is None:
            raise RuntimeError("Structured sparsity tensors not initialized")

        self.w1_compressed = torch._cslt_compress(self.w1)
        self.w3_compressed = torch._cslt_compress(self.w3)
        self.w2_compressed = torch._cslt_compress(self.w2)

        self.w1_algo = self._mm_search(self.w1_compressed, self.input_t)
        self.w3_algo = self._mm_search(self.w3_compressed, self.input_t)

        probe = torch.randn(
            self.cfg.ffn_size,
            self.cfg.tokens_per_iteration,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.w2_algo = self._mm_search(self.w2_compressed, probe)

    def dense_forward(self) -> torch.Tensor:
        if self.input is None or self.w1 is None or self.w3 is None or self.w2 is None:
            raise RuntimeError("Structured sparsity tensors not initialized")
        y1 = F.linear(self.input, self.w1)
        y3 = F.linear(self.input, self.w3)
        act = F.silu(y1) * y3
        return F.linear(act, self.w2)

    def sparse_forward(self) -> torch.Tensor:
        if (
            self.input_t is None
            or self.w1_compressed is None
            or self.w3_compressed is None
            or self.w2_compressed is None
            or self.w1_algo is None
            or self.w3_algo is None
            or self.w2_algo is None
        ):
            raise RuntimeError("Structured sparsity tensors not initialized")
        y1 = self._sparse_mm(self.w1_compressed, self.input_t, self.w1_algo)
        y3 = self._sparse_mm(self.w3_compressed, self.input_t, self.w3_algo)
        act = F.silu(y1) * y3
        act_t = act.t().contiguous()
        return self._sparse_mm(self.w2_compressed, act_t, self.w2_algo)

    def teardown(self) -> None:
        self.input = None
        self.input_t = None
        self.w1 = None
        self.w3 = None
        self.w2 = None
        self.w1_compressed = None
        self.w3_compressed = None
        self.w2_compressed = None
        self.w1_algo = None
        self.w3_algo = None
        self.w2_algo = None

    def _init_tensors(self) -> None:
        m = self.cfg.tokens_per_iteration
        self.input = torch.randn(m, self.cfg.hidden_size, device=self.device, dtype=self.cfg.dtype)
        self.input_t = self.input.t().contiguous()

        dense_w1 = torch.randn(self.cfg.ffn_size, self.cfg.hidden_size, device=self.device, dtype=self.cfg.dtype)
        dense_w3 = torch.randn(self.cfg.ffn_size, self.cfg.hidden_size, device=self.device, dtype=self.cfg.dtype)
        dense_w2 = torch.randn(self.cfg.hidden_size, self.cfg.ffn_size, device=self.device, dtype=self.cfg.dtype)

        self.w1 = prune_2_4(dense_w1)
        self.w3 = prune_2_4(dense_w3)
        self.w2 = prune_2_4(dense_w2)

    def _mm_search(self, weight_compressed: torch.Tensor, input_t: torch.Tensor) -> CusparseltAlgo:
        alg_id, split_k, split_k_mode, _ = torch._C._cusparselt.mm_search(
            weight_compressed,
            input_t,
            None,
            None,
            None,
            True,
        )
        return CusparseltAlgo(int(alg_id), int(split_k), int(split_k_mode))

    def _sparse_mm(self, weight_compressed: torch.Tensor, input_t: torch.Tensor, algo: CusparseltAlgo) -> torch.Tensor:
        return torch._cslt_sparse_mm(
            weight_compressed,
            input_t,
            transpose_result=True,
            alg_id=algo.alg_id,
            split_k=algo.split_k,
            split_k_mode=algo.split_k_mode,
        )
