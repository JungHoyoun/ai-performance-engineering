# MCP Tools Reference

The `aisp` MCP server exposes AI Systems Performance tools via JSON-RPC over stdio.

## Quick Start

```bash
# Start the MCP server (stdio)
python -m mcp.mcp_server --serve

# List tools (authoritative)
python -m mcp.mcp_server --list
```

## Common Workflow: Deep-Dive Baseline vs Optimized Compare

One-shot (recommended): `aisp_benchmark_deep_dive_compare`

```json
{
  "targets": ["ch10:atomic_reduction"],
  "output_dir": "artifacts/mcp-deep-dive",
  "async": true
}
```

This runs `bench run` with `profile=\"deep_dive\"`, writes outputs under a timestamped run dir, and returns:
- `run_dir`, `results_json`, `analysis_json`
- per-benchmark `profiles_dir` + `followup_tool_calls` for `aisp_profile_compare` / `aisp_compare_nsys` / `aisp_compare_ncu`

## Tool Names

Tool names are the exact names returned by `tools/list` / `--list` (for example: `aisp_gpu_info`, not `gpu_info`).

## Response Format

All tools return a single MCP `text` content entry containing a JSON envelope with:
- `tool`, `status`, `timestamp`, `duration_ms`
- `arguments` + `arguments_details`
- `result` + `result_preview` + `result_metadata`
- `context_summary` + `guidance.next_steps`

## `aisp_tools_*` (Non-benchmark Utilities)

These tools are intentionally **not** comparative benchmarks; they run utilities via `aisp tools <name>`.

- `aisp_tools_kv_cache`
- `aisp_tools_cost_per_token`
- `aisp_tools_compare_precision`
- `aisp_tools_detect_cutlass`
- `aisp_tools_dump_hw`
- `aisp_tools_probe_hw`

Each accepts:
- `args`: list of strings forwarded to the underlying utility script
- `timeout_seconds`: max runtime before returning
- `include_context` / `context_level`

Example call shape:

```json
{
  "args": ["--layers", "32", "--hidden", "4096", "--tokens", "4096", "--dtype", "fp16"]
}
```
