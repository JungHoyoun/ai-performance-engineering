# MCP Server & Client

This directory contains both the MCP server implementation and a robust client library.

## Files

- `mcp_server.py` - MCP server with 80+ tools for AI performance engineering
- `mcp_client.py` - Robust client implementation with proper message ID tracking
- `tool_generator.py` - Tool generation utilities

## Quick Start

### Server

```bash
# Start server
python -m mcp.mcp_server --serve

# List available tools
python -m mcp.mcp_server --list
```

## Common Workflow: Baseline vs Optimized Deep-Dive Compare

This is the standard "run → deep_dive profile → compare baseline vs optimized (nsys+ncu)" chain.

### One-shot (recommended)

Call the single MCP tool `aisp_benchmark_deep_dive_compare`:

```json
{
  "targets": ["ch10:atomic_reduction"],
  "output_dir": "artifacts/mcp-deep-dive",
  "async": true
}
```

Then poll `aisp_job_status` until complete and read:
- `run_dir`, `results_json`, `analysis_json`
- `benchmarks[]` entries, each with `profiles_dir` and `followup_tool_calls` (ready-to-run chaining inputs)

### Manual chaining (explicit steps)

1. Discover targets: `aisp_benchmark_targets`
2. Run: `aisp_run_benchmarks` with `profile=\"deep_dive\"` and an `artifacts_dir`
3. If async: poll with `aisp_job_status`
4. Analyze benchmark results: `aisp_benchmark_triage` using the returned `results_json`
5. Compare profiles: `aisp_profile_compare`, `aisp_compare_nsys`, `aisp_compare_ncu`

### Client

```python
from mcp.mcp_client import create_client

# Simple usage
with create_client(debug=True) as client:
    # List tools
    tools = client.list_tools()
    print(f"Found {len(tools)} tools")
    
    # Call a tool
    result = client.call_tool("aisp_status", {})
    print(result)
```

## Robustness Features

### Server (`mcp_server.py`)
- ✅ Request deduplication
- ✅ Automatic stale request cleanup
- ✅ Graceful error handling
- ✅ Debug logging support

### Client (`mcp_client.py`)
- ✅ Thread-safe message ID generation
- ✅ Request/response correlation
- ✅ Automatic timeout handling
- ✅ Duplicate response detection
- ✅ Graceful error recovery
- ✅ Context manager support

## Examples

See `examples/mcp_client_example.py` for usage examples.

## Documentation

- `docs/mcp_client_guide.md` - Comprehensive robustness guide
- `docs/mcp_tools.md` - Tool reference documentation

## Testing

```bash
# Test server
pytest tests/test_mcp_tools.py

# Test client
pytest tests/test_mcp_client.py
```

## Configuration

### Environment Variables

- `AISP_MCP_DEBUG=1` - Enable debug logging (server)
- `AISP_MCP_REQUEST_TIMEOUT=300` - Request timeout in seconds (server)

### Client Options

```python
client = RobustMCPClient(
    command=["python", "-m", "mcp.mcp_server", "--serve"],
    cwd="/path/to/workspace",
    timeout=300.0,  # Request timeout
    enable_debug=True  # Debug logging
)
```

## Architecture

```
┌─────────────────┐
│  MCP Client     │  ← Robust client with message ID tracking
│  (mcp_client)   │
└────────┬────────┘
         │ JSON-RPC over stdio
         ↓
┌─────────────────┐
│  MCP Server     │  ← Server with deduplication & cleanup
│  (mcp_server)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ PerformanceEngine│  ← Core functionality
│  (80+ tools)    │
└─────────────────┘
```

## Troubleshooting

### "Unknown message ID" errors

The robust client handles this automatically. If using a custom client:
1. Track all pending requests by message ID
2. Handle unknown IDs gracefully (log but don't crash)
3. Clean up stale requests periodically
4. Reset on reconnection

See `docs/mcp_client_guide.md` for details.
