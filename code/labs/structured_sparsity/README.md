# Lab - Structured Sparsity FFN (2:4)

## Summary
Combines Chapter 9 structured sparsity (2:4) with Chapter 12-style CUDA graph replay for a realistic LLM feed-forward block (GEMM + SwiGLU + GEMM).

## Learning Goals
- Compare dense vs cuSPARSELt sparse GEMMs on a full SwiGLU FFN.
- Understand when 2:4 sparsity delivers real speedups at LLM-scale batch/sequence sizes.
- See how CUDA graph replay reduces CPU launch overhead for steady-state inference.

## Files
| File | Description |
| --- | --- |
| `baseline_structured_sparsity_ffn.py` | Dense FFN baseline using 2:4-pruned weights. |
| `optimized_structured_sparsity_ffn.py` | cuSPARSELt sparse FFN with CUDA graph replay. |

## Running
```bash
python -m cli.aisp bench list-targets --chapter labs/structured_sparsity
python -m cli.aisp bench run --targets labs/structured_sparsity --profile minimal
```

## Notes
- Defaults match gpt-oss-20b dimensions (hidden=2880, ffn=2880, seq_len=4096, batch=4).
- Use larger batch or sequence lengths if your GPU is underutilized.
- Requires cuSPARSELt-enabled PyTorch builds.

## Related Chapters
- **Ch9**: Structured sparsity and arithmetic intensity.
- **Ch12**: CUDA graph replay for launch overhead reduction.
