from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from bootcamp.models import (
    ensure_list,
    load_gpt_oss_model,
    tokenize,
    warm_prompt_batch,
)


@dataclass
class InferenceMetrics:
    prompts: int
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    latency_s: float
    tokens_per_s: float
    requests_per_s: float
    max_memory_bytes: int
    dtype: str
    model_id: str
    max_new_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _device_of(model: torch.nn.Module, explicit: Optional[str]) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise RuntimeError("Model contains no parameters; cannot infer device.")
    return first_param.device


def run_inference(
    prompts: Optional[Iterable[str]] = None,
    *,
    max_new_tokens: int = 64,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
    decode: bool = False,
) -> Dict[str, Any]:
    """Run a timed GPT-OSS 20B inference and return metrics (and optionally text).

    Parameters
    ----------
    prompts:
        Prompts to feed the model. Defaults to the warm prompt batch.
    max_new_tokens:
        Number of tokens to generate per sequence.
    dtype:
        Desired model dtype string.
    device:
        Optional explicit device string.
    decode:
        If True, include decoded text output in the response.
    """

    prompt_list = ensure_list(prompts or warm_prompt_batch())
    encoded, tokenizer = tokenize(prompt_list)
    model = load_gpt_oss_model(dtype=dtype, device=device)
    model_device = _device_of(model, device)

    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    if model_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model_device)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if model_device.type == "cuda":
        torch.cuda.synchronize(model_device)
        max_memory = torch.cuda.max_memory_allocated(model_device)
    else:
        max_memory = 0
    elapsed = time.perf_counter() - start

    prompt_tokens = int(attention_mask.sum().item())
    generated_per_sequence = outputs.shape[1] - input_ids.shape[1]
    batch = outputs.shape[0]
    total_generated = generated_per_sequence * batch

    metrics = InferenceMetrics(
        prompts=batch,
        prompt_tokens=prompt_tokens,
        generated_tokens=total_generated,
        total_tokens=prompt_tokens + total_generated,
        latency_s=elapsed,
        tokens_per_s=(total_generated / elapsed) if elapsed > 0 else 0.0,
        requests_per_s=(batch / elapsed) if elapsed > 0 else 0.0,
        max_memory_bytes=max_memory,
        dtype=dtype,
        model_id=model.name_or_path,
        max_new_tokens=max_new_tokens,
    )

    result: Dict[str, Any] = {"metrics": metrics.to_dict()}
    if decode:
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result["outputs"] = decoded
    return result


def run_inference_for_profiler(
    prompts: Optional[Sequence[str]] = None,
    *,
    max_new_tokens: int = 64,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
) -> None:
    """Run an inference without collecting metrics (used under Nsight profiling)."""
    prompt_list = ensure_list(prompts or warm_prompt_batch())
    encoded, _ = tokenize(prompt_list)
    model = load_gpt_oss_model(dtype=dtype, device=device)
    model_device = _device_of(model, device)

    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if model_device.type == "cuda":
        torch.cuda.synchronize(model_device)
