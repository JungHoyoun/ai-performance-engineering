# Lab - TensorRT-LLM gpt-oss-20b

## Summary
Compares Hugging Face Transformers generation against TensorRT-LLM inference for gpt-oss-20b, using identical prompts and greedy decoding.

## Learning Goals
- Measure runtime speedups from TensorRT-LLM kernels and engine optimizations.
- Validate output logits for a fixed prompt and decoding length.
- Exercise TRT-LLM generation APIs in a harness-comparable workflow.

## Files
| File | Description |
| --- | --- |
| `baseline_trtllm_gpt_oss_20b.py` | Transformers baseline (eager attention). |
| `optimized_trtllm_gpt_oss_20b.py` | TensorRT-LLM optimized generation. |
| `trtllm_common.py` | Shared prompt/token helpers. |

## Running
```bash
# Baseline vs optimized (pass engine path for TRT-LLM)
python -m cli.aisp bench run --targets labs/trtllm_gpt_oss_20b \
  --target-extra-arg labs/trtllm_gpt_oss_20b:optimized_trtllm_gpt_oss_20b="--engine-path /path/to/engine.plan"
```

## Notes
- Requires local gpt-oss-20b weights at `gpt-oss-20b/original` (override with `--model-path`).
- TRT-LLM must be built with `output_generation_logits=True` support; the benchmark validates `generation_logits`.
- Keep TRT-LLM precision aligned with the baseline (e.g., FP16) to pass output verification.

## Related Chapters
- **Ch16**: Production inference systems and engine comparisons.
- **Ch18**: TensorRT-LLM and NVFP4 workflows.
