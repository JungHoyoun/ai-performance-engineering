from __future__ import annotations

from core.api.response import build_response


def test_response_envelope_status():
    ok_payload = build_response("tool.ok", {"value": 1}, {"ok": True}, 5)
    assert ok_payload["status"] == "ok"
    assert ok_payload["success"] is True
    assert ok_payload["tool"] == "tool.ok"
    assert ok_payload["arguments"] == {"value": 1}
    assert "context_summary" in ok_payload

    err_payload = build_response("tool.err", {}, {"error": "nope"}, 5)
    assert err_payload["status"] == "error"
    assert err_payload["success"] is False
