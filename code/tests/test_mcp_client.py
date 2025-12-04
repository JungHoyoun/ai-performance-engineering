#!/usr/bin/env python3
"""
Tests for robust MCP client implementation.
"""

import pytest
import time
import threading
from mcp.mcp_client import RobustMCPClient, MCPResponse, PendingRequest


def test_message_id_generation():
    """Test that message IDs are unique and thread-safe."""
    client = RobustMCPClient(command=["echo"], timeout=1.0)
    
    ids = set()
    num_threads = 10
    ids_per_thread = 100
    
    def generate_ids():
        for _ in range(ids_per_thread):
            ids.add(client._get_next_id())
    
    threads = [threading.Thread(target=generate_ids) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have generated unique IDs
    assert len(ids) == num_threads * ids_per_thread


def test_request_tracking():
    """Test request tracking and completion."""
    client = RobustMCPClient(command=["echo"], timeout=1.0)
    
    msg_id = 1
    method = "test_method"
    params = {"test": "value"}
    
    # Track request
    pending = client._track_request(msg_id, method, params, timeout=10.0)
    assert msg_id in client._pending_requests
    assert client._pending_requests[msg_id] == pending
    
    # Complete request
    response = MCPResponse(msg_id=msg_id, result={"success": True})
    completed = client._complete_request(msg_id, response)
    assert completed is True
    assert msg_id not in client._pending_requests
    
    # Verify future was set
    assert pending.future.done()
    assert pending.future.result() == response


def test_unknown_message_id_handling():
    """Test handling of unknown message IDs gracefully."""
    client = RobustMCPClient(command=["echo"], timeout=1.0)
    
    # Try to complete a request that was never tracked
    response = MCPResponse(msg_id=999, result={"test": True})
    completed = client._complete_request(999, response)
    
    # Should return False but not crash
    assert completed is False


def test_stale_request_cleanup():
    """Test cleanup of stale requests."""
    client = RobustMCPClient(command=["echo"], timeout=0.1)  # Short timeout
    
    # Track a request
    msg_id = 1
    pending = client._track_request(msg_id, "test", {}, timeout=0.1)
    
    # Wait for timeout
    time.sleep(0.2)
    
    # Cleanup should remove it
    client._cleanup_stale_requests()
    
    assert msg_id not in client._pending_requests
    assert pending.future.done()
    
    # Future should have error response
    result = pending.future.result()
    assert result.error is not None
    assert "timed out" in result.error["message"]


def test_duplicate_message_id_prevention():
    """Test that duplicate message IDs are handled."""
    client = RobustMCPClient(command=["echo"], timeout=1.0)
    
    msg_id = 1
    
    # Track first request
    pending1 = client._track_request(msg_id, "method1", {}, timeout=10.0)
    
    # Track second request with same ID (should replace)
    pending2 = client._track_request(msg_id, "method2", {}, timeout=10.0)
    
    # Should only have one pending request
    assert len(client._pending_requests) == 1
    assert client._pending_requests[msg_id] == pending2


def test_response_parsing():
    """Test parsing of different response formats."""
    # Test result response
    result_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": []}
    }
    mcp_response = MCPResponse(
        msg_id=1,
        result=result_response.get("result"),
        error=result_response.get("error")
    )
    assert mcp_response.result is not None
    assert mcp_response.error is None
    
    # Test error response
    error_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": -32601, "message": "Method not found"}
    }
    mcp_response = MCPResponse(
        msg_id=2,
        result=error_response.get("result"),
        error=error_response.get("error")
    )
    assert mcp_response.result is None
    assert mcp_response.error is not None
    assert mcp_response.error["code"] == -32601


if __name__ == "__main__":
    pytest.main([__file__, "-v"])






