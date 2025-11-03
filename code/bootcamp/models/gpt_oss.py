from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("BOOTCAMP_GPT_OSS_MODEL", "openai/gpt-oss-20b")
_MODEL_CACHE: dict[Tuple[str, str], AutoModelForCausalLM] = {}


def _dtype_from_name(name: str) -> torch.dtype:
    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of {sorted(lookup)}.")
    return lookup[key]


@lru_cache(maxsize=1)
def load_gpt_oss_tokenizer():
    """Load and cache the GPT-OSS tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_gpt_oss_model(dtype: str = "bfloat16", device: str | None = None) -> AutoModelForCausalLM:
    """Load and cache the GPT-OSS 20B causal LM.

    Parameters
    ----------
    dtype:
        Desired dtype string (e.g. 'bfloat16', 'fp16').
    device:
        Optional explicit device string. Defaults to CUDA:0 if available.
    """

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prefer_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype_key = dtype.lower()
    cache_key = (dtype_key, prefer_device)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    torch_dtype = _dtype_from_name(dtype_key)
    kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if prefer_device.startswith("cuda"):
        kwargs["device_map"] = {"": torch.device(prefer_device)}
    else:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def warm_prompt_batch() -> List[str]:
    """Return a canonical set of prompts for warm-up and benchmarking."""
    return [
        "Explain how to optimize tensor core GEMM kernels on NVIDIA Blackwell in three bullet points.",
        "Summarize the key differences between tcgen05 and previous GPU tensor core instructions.",
        "Provide a short PyTorch code snippet that benchmarks a causal LM with bf16 weights.",
    ]


def ensure_list(prompts: Iterable[str] | str) -> List[str]:
    if isinstance(prompts, str):
        return [prompts]
    return list(prompts)


def tokenize(prompts: Sequence[str]):
    tokenizer = load_gpt_oss_tokenizer()
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
    )
    return encoded, tokenizer


__all__ = [
    "MODEL_ID",
    "ensure_list",
    "load_gpt_oss_model",
    "load_gpt_oss_tokenizer",
    "tokenize",
    "warm_prompt_batch",
]
