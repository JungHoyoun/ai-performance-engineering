"""Baseline disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

Baseline behavior is serialized: decode waits until all prefills complete.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)


@dataclass(frozen=True)
class PrefillDecodeConfig:
    hidden_size: int = 3072
    num_layers: int = 6
    batch_size: int = 2
    requests_per_rank: int = 24
    context_window: int = 1024
    decode_tokens: int = 384
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


@dataclass
class _LocalPair:
    prefill_device: torch.device
    decode_device: torch.device
    prefill_model: "TinyPrefillDecode"
    decode_model: "TinyPrefillDecode"
    prompts: torch.Tensor


class TinyPrefillDecode(nn.Module):
    """Simple prefill/decode model to emulate KV cache traffic."""

    def __init__(self, hidden_size: int, num_layers: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to(device=device, dtype=dtype)

    def prefill(self, prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = prompts
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.proj(x)
        kv_cache = x
        seed = logits[:, -1, :]
        return kv_cache, seed

    def decode(self, seed: torch.Tensor, kv_cache: torch.Tensor, decode_tokens: int) -> torch.Tensor:
        x = seed
        context = kv_cache.shape[1]
        for step in range(decode_tokens):
            kv = kv_cache[:, step % context, :]
            x = x + kv
            for layer in self.layers:
                x = torch.relu(layer(x))
            x = self.proj(x)
        return x


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for disaggregated prefill/decode")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("prefill_decode_disagg_multigpu requires >=2 GPUs.")
    return world_size


def _resolve_prefill_ranks(world_size: int, prefill_ranks: Optional[int]) -> int:
    if prefill_ranks is None:
        if world_size % 2 != 0:
            raise RuntimeError(
                "--prefill-ranks must be set when world_size is odd."
            )
        return world_size // 2
    requested = int(prefill_ranks)
    if requested < 1:
        raise RuntimeError(f"--prefill-ranks must be >= 1 (got {requested}).")
    if requested >= world_size:
        raise RuntimeError(
            f"--prefill-ranks={requested} must be < world_size={world_size}."
        )
    return requested


def _prefill_to_decode_rank(prefill_rank: int, prefill_ranks: int, decode_ranks: int) -> int:
    return prefill_ranks + (prefill_rank % decode_ranks)


def _decode_assigned_prefills(decode_rank: int, prefill_ranks: int, decode_ranks: int) -> List[int]:
    decode_idx = decode_rank - prefill_ranks
    return [rank for rank in range(prefill_ranks) if (rank % decode_ranks) == decode_idx]


def _emit_split_advice(prefill_ranks: int, decode_ranks: int) -> None:
    ratio = prefill_ranks / max(decode_ranks, 1)
    split_label = f"P{prefill_ranks}:D{decode_ranks}"
    if prefill_ranks == decode_ranks:
        print(f"Split {split_label} is balanced (recommended default).")
    if decode_ranks == 1 and prefill_ranks > 1:
        print(
            f"WARNING: split {split_label} may bottleneck TPOT/long outputs with a single decode rank."
        )
    if prefill_ranks == 1 and decode_ranks > 1:
        print(
            f"WARNING: split {split_label} may bottleneck TTFT for long prompts with a single prefill rank."
        )
    if ratio >= 3:
        print(
            f"WARNING: prefill-heavy split {split_label} may under-provision decode for TPOT."
        )
    elif ratio <= (1 / 3):
        print(
            f"WARNING: decode-heavy split {split_label} may under-provision prefill for TTFT."
        )
    print(
        "Recommendation: use a balanced split when unsure; prefer 2P1D for TTFT-focused runs "
        "and 1P2D+ for TPOT/long outputs."
    )


def _init_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for disaggregated prefill/decode")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run with torchrun (missing RANK/WORLD_SIZE env vars).")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run with torchrun (missing LOCAL_RANK env var).")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _run_prefill(
    cfg: PrefillDecodeConfig,
    model: TinyPrefillDecode,
    prompts: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    kv_chunks: List[torch.Tensor] = []
    seed_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for req_idx in range(cfg.requests_per_rank):
            request_prompt = prompts[req_idx]
            kv_cache, seed = model.prefill(request_prompt)
            kv_chunks.append(kv_cache)
            seed_chunks.append(seed)
    return kv_chunks, seed_chunks


def _run_decode(
    cfg: PrefillDecodeConfig,
    model: TinyPrefillDecode,
    kv_chunks: List[torch.Tensor],
    seed_chunks: List[torch.Tensor],
) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for kv_cache, seed in zip(kv_chunks, seed_chunks):
            outputs.append(model.decode(seed, kv_cache, cfg.decode_tokens))
    return outputs


def _run_torchrun_worker(
    cfg: PrefillDecodeConfig,
    *,
    overlap: bool,
    label: str,
    iters: int,
    warmup: int,
    prefill_ranks: Optional[int],
) -> None:
    rank, world_size, device = _init_distributed()
    if world_size < 2:
        raise RuntimeError("prefill_decode_disagg_multigpu requires >=2 GPUs.")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"torchrun world_size={world_size} exceeds visible GPUs ({torch.cuda.device_count()})."
        )

    prefill_ranks = _resolve_prefill_ranks(world_size, prefill_ranks)
    decode_ranks = world_size - prefill_ranks
    if decode_ranks < 1:
        raise RuntimeError("decode_ranks must be >= 1 for disaggregated prefill/decode")
    if rank == 0:
        _emit_split_advice(prefill_ranks, decode_ranks)

    pair_groups: dict[int, dist.ProcessGroup] = {}
    for prefill_rank in range(prefill_ranks):
        decode_rank = _prefill_to_decode_rank(prefill_rank, prefill_ranks, decode_ranks)
        pair_groups[prefill_rank] = dist.new_group(ranks=[prefill_rank, decode_rank])

    is_prefill = rank < prefill_ranks
    peer_rank = (
        _prefill_to_decode_rank(rank, prefill_ranks, decode_ranks)
        if is_prefill
        else -1
    )
    device_index = 0 if device.index is None else int(device.index)

    def _barrier() -> None:
        dist.barrier(device_ids=[device_index])

    def _batch_send(kv_cache: torch.Tensor, seed: torch.Tensor, dst: int, group: dist.ProcessGroup) -> None:
        ops = [
            dist.P2POp(dist.isend, kv_cache, dst, group=group),
            dist.P2POp(dist.isend, seed, dst, group=group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    def _batch_recv(kv_buf: torch.Tensor, seed_buf: torch.Tensor, src: int, group: dist.ProcessGroup) -> None:
        ops = [
            dist.P2POp(dist.irecv, kv_buf, src, group=group),
            dist.P2POp(dist.irecv, seed_buf, src, group=group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = TinyPrefillDecode(cfg.hidden_size, cfg.num_layers, device, cfg.dtype).eval()

    prompts: Optional[torch.Tensor] = None
    if is_prefill:
        prompts = torch.randn(
            cfg.requests_per_rank,
            cfg.batch_size,
            cfg.context_window,
            cfg.hidden_size,
            device=device,
            dtype=cfg.dtype,
        )

    def run_iteration() -> List[torch.Tensor]:
        if is_prefill:
            kv_chunks, seed_chunks = _run_prefill(cfg, model, prompts)
            if overlap:
                for kv_cache, seed in zip(kv_chunks, seed_chunks):
                    _batch_send(kv_cache, seed, peer_rank, pair_groups[rank])
            else:
                for kv_cache, seed in zip(kv_chunks, seed_chunks):
                    _batch_send(kv_cache, seed, peer_rank, pair_groups[rank])
                torch.cuda.synchronize(device)
            if not overlap:
                _barrier()
            return []

        kv_chunks = []
        seed_chunks = []
        assigned_prefills = _decode_assigned_prefills(rank, prefill_ranks, decode_ranks)
        outputs = []
        for src_rank in assigned_prefills:
            for _ in range(cfg.requests_per_rank):
                kv_buf = torch.empty(
                    (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                    device=device,
                    dtype=cfg.dtype,
                )
                seed_buf = torch.empty(
                    (cfg.batch_size, cfg.hidden_size), device=device, dtype=cfg.dtype
                )
                _batch_recv(kv_buf, seed_buf, src_rank, pair_groups[src_rank])
                if overlap:
                    outputs.append(model.decode(seed_buf, kv_buf, cfg.decode_tokens))
                else:
                    kv_chunks.append(kv_buf)
                    seed_chunks.append(seed_buf)

        if overlap:
            return outputs
        torch.cuda.synchronize(device)
        _barrier()
        return _run_decode(cfg, model, kv_chunks, seed_chunks)

    _barrier()
    torch.cuda.synchronize(device)

    for _ in range(max(warmup, 0)):
        run_iteration()
    torch.cuda.synchronize(device)
    _barrier()

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        run_iteration()
    torch.cuda.synchronize(device)
    _barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        total_requests = cfg.requests_per_rank * prefill_ranks * cfg.batch_size
        tokens_per_iter = total_requests * cfg.tokens_per_request
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        time_per_iter_ms = (elapsed / max(iters, 1)) * 1000.0
        print(f"rank0 {label} tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 {label} time_per_iter_ms: {time_per_iter_ms:.3f}")

    dist.destroy_process_group()


class _PrefillDecodeMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared multi-GPU disaggregated prefill/decode harness."""

    multi_gpu_required = True

    def __init__(
        self,
        *,
        overlap: bool,
        label: str,
        cfg: Optional[PrefillDecodeConfig] = None,
        prefill_ranks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PrefillDecodeConfig()
        self.world_size = _resolve_world_size()
        self.prefill_ranks = _resolve_prefill_ranks(self.world_size, prefill_ranks)
        self.decode_ranks = self.world_size - self.prefill_ranks
        if self.decode_ranks < 1:
            raise RuntimeError("decode_ranks must be >= 1 for disaggregated prefill/decode")
        self.overlap = bool(overlap)
        self.label = label
        self._pairs: List[_LocalPair] = []
        self._output: Optional[torch.Tensor] = None
        self._verify_prompt: Optional[torch.Tensor] = None
        self._param_count: int = 0

        total_requests = self.cfg.requests_per_rank * self.prefill_ranks * self.cfg.batch_size
        tokens_per_iter = total_requests * self.cfg.tokens_per_request
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(tokens_per_iter),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for disaggregated prefill/decode")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"SKIPPED: requires >= {self.world_size} GPUs (found {torch.cuda.device_count()})"
            )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self._pairs = []
        total_params = 0
        reference = TinyPrefillDecode(
            self.cfg.hidden_size,
            self.cfg.num_layers,
            torch.device("cpu"),
            torch.float32,
        ).eval()
        reference_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}

        decode_devices = [
            torch.device(f"cuda:{idx}")
            for idx in range(self.prefill_ranks, self.prefill_ranks + self.decode_ranks)
        ]
        decode_models: dict[torch.device, TinyPrefillDecode] = {}
        for device in decode_devices:
            model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                device,
                self.cfg.dtype,
            ).eval()
            model.load_state_dict(reference_state)
            decode_models[device] = model
            total_params += sum(p.numel() for p in model.parameters())

        for prefill_rank in range(self.prefill_ranks):
            prefill_device = torch.device(f"cuda:{prefill_rank}")
            decode_device = decode_devices[prefill_rank % len(decode_devices)]
            prefill_model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                prefill_device,
                self.cfg.dtype,
            ).eval()
            prefill_model.load_state_dict(reference_state)
            prompts = torch.randn(
                self.cfg.requests_per_rank,
                self.cfg.batch_size,
                self.cfg.context_window,
                self.cfg.hidden_size,
                device=prefill_device,
                dtype=self.cfg.dtype,
            )
            total_params += sum(p.numel() for p in prefill_model.parameters())
            self._pairs.append(
                _LocalPair(
                    prefill_device=prefill_device,
                    decode_device=decode_device,
                    prefill_model=prefill_model,
                    decode_model=decode_models[decode_device],
                    prompts=prompts,
                )
            )

        self._param_count = total_params
        if not self._pairs:
            raise RuntimeError("Failed to initialize prompts for verification")
        self._verify_prompt = self._pairs[0].prompts[0]
        for pair in self._pairs:
            torch.cuda.synchronize(pair.prefill_device)
            torch.cuda.synchronize(pair.decode_device)

    def benchmark_fn(self) -> None:
        if not self._pairs:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for pair in self._pairs:
                if self.overlap:
                    for req_idx in range(self.cfg.requests_per_rank):
                        kv_cache, seed = pair.prefill_model.prefill(pair.prompts[req_idx])
                        kv_cache = kv_cache.to(pair.decode_device, non_blocking=True)
                        seed = seed.to(pair.decode_device, non_blocking=True)
                        outputs.append(
                            pair.decode_model.decode(seed, kv_cache, self.cfg.decode_tokens)
                        )
                else:
                    kv_chunks, seed_chunks = _run_prefill(self.cfg, pair.prefill_model, pair.prompts)
                    kv_chunks = [kv.to(pair.decode_device) for kv in kv_chunks]
                    seed_chunks = [seed.to(pair.decode_device) for seed in seed_chunks]
                    decoded = _run_decode(self.cfg, pair.decode_model, kv_chunks, seed_chunks)
                    outputs.extend(decoded)

        self._output = torch.stack([out.detach().cpu() for out in outputs], dim=0)

    def capture_verification_payload(self) -> None:
        if self._output is None or self._verify_prompt is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        meta_dtype = torch.float32
        self._set_verification_payload(
            inputs={
                "prompt": self._verify_prompt,
                "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
                "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
                "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
            },
            output=self._output,
            batch_size=int(self._output.shape[0]),
            parameter_count=int(self._param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": self.world_size,
                "pipeline_stages": 2,
                "pipeline_stage_boundaries": [
                    (0, self.prefill_ranks - 1),
                    (self.prefill_ranks, self.prefill_ranks + self.decode_ranks - 1),
                ],
                "per_rank_batch_size": self.cfg.requests_per_rank,
                "collective_type": "send_recv",
            },
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def teardown(self) -> None:
        self._pairs = []
        self._output = None
        self._verify_prompt = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=self.world_size,
            iterations=4,
            warmup=3,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        script_args = ["--prefill-ranks", str(self.prefill_ranks)]
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=script_args,
            env={
                "NCCL_DEBUG": "WARN",
                "OMP_NUM_THREADS": "1",
                "MASTER_PORT": "29517",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


class BaselinePrefillDecodeDisaggMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Serialized prefill then decode across multi-GPU ranks."""

    def __init__(self) -> None:
        super().__init__(overlap=False, label="baseline_prefill_decode_disagg_multigpu")


def get_benchmark() -> BaseBenchmark:
    return BaselinePrefillDecodeDisaggMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--prefill-ranks",
        type=int,
        default=None,
        help="Number of prefill ranks (defaults to world_size//2 when even).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        PrefillDecodeConfig(),
        overlap=False,
        label="baseline_prefill_decode_disagg_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
