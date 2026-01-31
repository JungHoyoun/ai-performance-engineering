"""Explicit registry of dashboard HTTP API routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from core.api import handlers


@dataclass(frozen=True)
class ApiRoute:
    method: str
    path: str
    name: str
    handler: Callable[[Dict[str, object]], Dict[str, object]]
    description: Optional[str] = None
    engine_op: Optional[str] = None
    mcp_tool: Optional[str] = None
    meta: bool = False


_ROUTES: List[ApiRoute] = [
    ApiRoute(
        "GET",
        "/api/gpu/info",
        "gpu.info",
        handlers.gpu_info,
        engine_op="gpu.info",
        mcp_tool="aisp_gpu_info",
    ),
    ApiRoute(
        "GET",
        "/api/gpu/topology",
        "gpu.topology",
        handlers.gpu_topology,
        engine_op="gpu.topology",
        mcp_tool="aisp_gpu_topology",
    ),
    ApiRoute(
        "GET",
        "/api/gpu/power",
        "gpu.power",
        handlers.gpu_power,
        engine_op="gpu.power",
        mcp_tool="aisp_gpu_power",
    ),
    ApiRoute(
        "GET",
        "/api/gpu/bandwidth",
        "gpu.bandwidth",
        handlers.gpu_bandwidth,
        engine_op="gpu.bandwidth",
        mcp_tool="aisp_gpu_bandwidth",
    ),
    ApiRoute(
        "GET",
        "/api/system/software",
        "system.software",
        handlers.system_software,
        engine_op="system.software",
        mcp_tool="aisp_system_software",
    ),
    ApiRoute(
        "GET",
        "/api/system/dependencies",
        "system.dependencies",
        handlers.system_dependencies,
        engine_op="system.dependencies",
        mcp_tool="aisp_system_dependencies",
    ),
    ApiRoute(
        "GET",
        "/api/system/context",
        "system.context",
        handlers.system_context,
        engine_op="system.context",
        mcp_tool="aisp_system_context",
    ),
    ApiRoute(
        "GET",
        "/api/system/capabilities",
        "system.capabilities",
        handlers.system_capabilities,
        engine_op="system.capabilities",
        mcp_tool="aisp_system_capabilities",
    ),
    ApiRoute(
        "GET",
        "/api/benchmark/data",
        "benchmark.data",
        handlers.benchmark_data,
        engine_op="benchmark.data",
    ),
    ApiRoute(
        "GET",
        "/api/benchmark/overview",
        "benchmark.overview",
        handlers.benchmark_overview,
        engine_op="benchmark.overview",
    ),
    ApiRoute(
        "GET",
        "/api/benchmark/history",
        "benchmark.history",
        handlers.benchmark_history,
        engine_op="benchmark.history",
    ),
    ApiRoute(
        "GET",
        "/api/benchmark/trends",
        "benchmark.trends",
        handlers.benchmark_trends,
        engine_op="benchmark.trends",
    ),
    ApiRoute(
        "GET",
        "/api/benchmark/compare",
        "benchmark.compare",
        handlers.benchmark_compare,
        engine_op="benchmark.compare",
    ),
    ApiRoute(
        "GET",
        "/api/benchmark/targets",
        "benchmark.targets",
        handlers.benchmark_targets,
        engine_op="benchmark.targets",
        mcp_tool="aisp_benchmark_targets",
    ),
    ApiRoute(
        "GET",
        "/api/profile/flame",
        "profile.flame_graph",
        handlers.profile_flame,
        engine_op="profile.flame_graph",
        mcp_tool="aisp_profile_flame",
    ),
    ApiRoute(
        "GET",
        "/api/profile/kernels",
        "profile.kernels",
        handlers.profile_kernels,
        engine_op="profile.kernels",
        mcp_tool="aisp_profile_kernels",
    ),
    ApiRoute(
        "GET",
        "/api/profile/memory",
        "profile.memory_timeline",
        handlers.profile_memory,
        engine_op="profile.memory_timeline",
        mcp_tool="aisp_profile_memory",
    ),
    ApiRoute(
        "GET",
        "/api/profile/hta",
        "profile.hta",
        handlers.profile_hta,
        engine_op="profile.hta",
        mcp_tool="aisp_profile_hta",
    ),
    ApiRoute(
        "GET",
        "/api/profile/compile",
        "profile.compile_analysis",
        handlers.profile_compile,
        engine_op="profile.compile_analysis",
    ),
    ApiRoute(
        "GET",
        "/api/profile/roofline",
        "profile.roofline",
        handlers.profile_roofline,
        engine_op="profile.roofline",
        mcp_tool="aisp_profile_roofline",
    ),
    ApiRoute(
        "GET",
        "/api/profile/list",
        "profile.list_profiles",
        handlers.profile_list,
        engine_op="profile.list_profiles",
    ),
    ApiRoute(
        "GET",
        "/api/profile/compare",
        "profile.compare",
        handlers.profile_compare,
        engine_op="profile.compare",
        mcp_tool="aisp_profile_compare",
    ),
    ApiRoute(
        "GET",
        "/api/profile/recommendations",
        "profile.recommendations",
        handlers.profile_recommendations,
        engine_op="profile.recommendations",
    ),
    ApiRoute(
        "GET",
        "/api/ai/status",
        "ai.status",
        handlers.ai_status,
        engine_op="ai.status",
        mcp_tool="aisp_ai_status",
    ),
    ApiRoute(
        "POST",
        "/api/ai/ask",
        "ai.ask",
        handlers.ai_ask,
        engine_op="ai.ask",
        mcp_tool="aisp_ask",
    ),
    ApiRoute(
        "POST",
        "/api/ai/explain",
        "ai.explain",
        handlers.ai_explain,
        engine_op="ai.explain",
        mcp_tool="aisp_explain",
    ),
    ApiRoute(
        "GET",
        "/api/ai/tools",
        "ai.tools",
        handlers.ai_tools,
        meta=True,
    ),
    ApiRoute(
        "POST",
        "/api/ai/execute",
        "ai.execute",
        handlers.ai_execute,
        meta=True,
    ),
    ApiRoute(
        "GET",
        "/api/jobs/status",
        "jobs.status",
        handlers.job_status,
        mcp_tool="aisp_job_status",
        meta=True,
    ),
]


def get_routes() -> List[ApiRoute]:
    return list(_ROUTES)


def get_dashboard_mcp_tools() -> List[str]:
    """Return MCP tool names explicitly surfaced by dashboard routes."""
    tools = {route.mcp_tool for route in _ROUTES if route.mcp_tool}
    return sorted(tools)
