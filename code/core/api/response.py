"""Unified HTTP response envelope aligned with MCP tooling."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.engine import get_engine


_CONTEXT_CACHE: Dict[str, Any] = {"summary": None, "full": None}
_CONTEXT_TS: Dict[str, float] = {"summary": 0.0, "full": 0.0}
_CONTEXT_TTL_SECONDS = 60.0


def _looks_like_error(result: Any, had_exception: bool = False) -> bool:
    if had_exception:
        return True
    if isinstance(result, dict):
        if result.get("error"):
            return True
        if result.get("success") is False:
            return True
    return False


def _build_context(level: str) -> Dict[str, Any]:
    engine = get_engine()
    if level == "summary":
        return {
            "gpu": engine.gpu.info(),
            "software": engine.system.software(),
            "dependencies": engine.system.dependencies(),
        }
    return engine.system.context()


def get_cached_context(level: str) -> Any:
    now = time.time()
    level = "full" if level == "full" else "summary"
    if _CONTEXT_CACHE.get(level) is None or (now - _CONTEXT_TS.get(level, 0.0)) > _CONTEXT_TTL_SECONDS:
        _CONTEXT_CACHE[level] = _build_context(level)
        _CONTEXT_TS[level] = now
    return _CONTEXT_CACHE[level]


def build_response(
    tool: str,
    arguments: Optional[Dict[str, Any]],
    result: Any,
    duration_ms: int,
    *,
    had_exception: bool = False,
    include_context: bool = False,
    context_level: str = "summary",
) -> Dict[str, Any]:
    """Build a response envelope mirroring MCP-style metadata."""
    status_is_error = _looks_like_error(result, had_exception)
    payload: Dict[str, Any] = {
        "tool": tool,
        "status": "error" if status_is_error else "ok",
        "success": not status_is_error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": duration_ms,
        "arguments": arguments or {},
        "result": result,
        "context_summary": get_cached_context("summary"),
    }
    if include_context:
        payload["context"] = get_cached_context(context_level)
        payload["context_level"] = "full" if context_level == "full" else "summary"
    return payload
