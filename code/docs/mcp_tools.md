# MCP Tools Reference

The `aisp` MCP server exposes AI Systems Performance tools via JSON-RPC over stdio.

## Quick Start

```bash
# Start the MCP server (stdio)
python -m mcp.mcp_server --serve

# List tools (authoritative)
python -m mcp.mcp_server --list
```

## Security / Authentication

The MCP server is **JSON-RPC over stdio** and is intended for **local use** (editor integration, local automation, CI). It does not implement authentication because stdio does not expose a network listener.

If you wrap MCP for network exposure, deploy it behind an authenticated transport (for example: SSH tunnel, a reverse proxy with auth, or mTLS) and treat tool execution as privileged.

## Common Workflow: Deep-Dive Baseline vs Optimized Compare

One-shot (recommended): `aisp_benchmark_deep_dive_compare`

```json
{
  "targets": ["ch10:atomic_reduction"],
  "output_dir": "artifacts/runs",
  "async": true
}
```

This runs `bench run` with `profile=\"deep_dive\"`, writes outputs under a timestamped run dir, and returns:
- `run_dir`, `results_json`, `analysis_json`
- per-benchmark `profiles_dir` + `followup_tool_calls` for `aisp_profile_compare` / `aisp_compare_nsys` / `aisp_compare_ncu`

## Common Workflow: Standard Benchmark Run (Minimal Profile)

Default: `aisp_run_benchmarks` uses `profile="minimal"` unless you explicitly request `deep_dive`.
It also runs post-benchmark triage and generates an HTML report unless you set
`auto_analyze=false` or `auto_report=false`.

```json
{
  "targets": ["ch10:atomic_reduction"]
}
```

## Tool Names

Tool names are the exact names returned by `tools/list` / `--list` (for example: `aisp_gpu_info`, not `gpu_info`).

## Response Format

All tools return a single MCP `text` content entry containing a JSON envelope with:
- `tool`, `status`, `timestamp`, `duration_ms`
- `arguments` + `arguments_details`
- `result` + `result_preview` + `result_metadata`
- `context_summary` + `guidance.next_steps`

## Profiling Metrics Payloads

Profiling tools return structured JSON metrics in their result payloads:
- `aisp_profile_nsys` includes `result.nsys_metrics` (parsed Nsight Systems metrics)
- `aisp_profile_ncu` includes `result.ncu_metrics` (parsed Nsight Compute metrics)
- `aisp_profile_torch` includes `result.torch_metrics` (summary metrics; `result.report` is an alias)
- `aisp_profile_hta` includes `result.nsys_metrics` alongside `analysis_summary`

Compare tools also return metric diffs when profiles are present:
- `aisp_compare_nsys` returns `result.nsys_comparison` and `result.ncu_comparison` when captured
- `aisp_compare_ncu` returns `result.ncu_comparison` and `result.nsys_comparison` when captured
- `aisp_profile_compare` always attaches both comparisons when captured

Use these metric payloads to explain speedups/regressions and identify bottleneck shifts.

## Async Jobs

Some tools can return an async job ticket (`job_id`) that you can poll via `aisp_job_status`. Job records are kept in-memory with bounded retention:

- `AISP_MCP_JOB_TTL_SECONDS` (default: `3600`)
- `AISP_MCP_JOB_MAX_ENTRIES` (default: `1000`)
- `AISP_MCP_JOB_CLEANUP_INTERVAL_SECONDS` (default: `30`)

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
