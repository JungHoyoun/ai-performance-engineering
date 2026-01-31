from __future__ import annotations

from core.api.registry import get_routes


def test_registry_routes_unique_and_valid():
    routes = get_routes()
    assert routes

    seen = set()
    names = set()
    for route in routes:
        key = (route.method, route.path)
        assert key not in seen
        seen.add(key)
        assert route.name not in names
        names.add(route.name)
        assert route.path.startswith("/api/")
        assert route.method in {"GET", "POST"}
        assert callable(route.handler)
        assert route.engine_op or route.mcp_tool or route.meta
        if route.engine_op:
            assert "." in route.engine_op
