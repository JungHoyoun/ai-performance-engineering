"""Shared helpers for the TRT-LLM gpt-oss-20b lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "gpt-oss-20b" / "original"
PROMPT_TEXT = "Explain GPU kernel fusion in one sentence."


def parse_trtllm_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--engine-path", type=str, default=None)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-slice", type=int, default=256)
    args, _ = parser.parse_known_args()
    return args


def build_prompt_tokens(tokenizer, *, prompt_len: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer.encode(PROMPT_TEXT, add_special_tokens=True)
    if len(encoded) > prompt_len:
        encoded = encoded[:prompt_len]
    else:
        encoded = encoded + [tokenizer.pad_token_id] * (prompt_len - len(encoded))
    input_ids = torch.tensor([encoded] * batch_size, dtype=torch.long)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attention_mask


def slice_logits(logits: torch.Tensor, vocab_slice: int) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("Expected logits of shape [batch, vocab]")
    return logits[:, :vocab_slice]
