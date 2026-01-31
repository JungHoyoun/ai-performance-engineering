from __future__ import annotations

import json

from fastapi.testclient import TestClient

from core.engine import reset_engine
from dashboard.api import server


def _write_run(path, timestamp, results):
    payload = {"timestamp": timestamp, "results": results}
    path.write_text(json.dumps(payload))


def test_compare_runs_endpoint(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    _write_run(
        baseline_path,
        "2025-01-01 00:00:00",
        [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "gemm",
                        "best_speedup": 2.0,
                        "baseline_time_ms": 100.0,
                        "status": "succeeded",
                    },
                    {
                        "example": "mem",
                        "best_speedup": 1.0,
                        "baseline_time_ms": 50.0,
                        "status": "skipped",
                    },
                ],
            }
        ],
    )

    _write_run(
        candidate_path,
        "2025-01-02 00:00:00",
        [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "gemm",
                        "best_speedup": 3.0,
                        "baseline_time_ms": 100.0,
                        "status": "succeeded",
                    }
                ],
            },
            {
                "chapter": "ch02",
                "benchmarks": [
                    {
                        "example": "transfer",
                        "best_speedup": 1.5,
                        "baseline_time_ms": 200.0,
                        "status": "failed",
                    }
                ],
            },
        ],
    )

    reset_engine()
    server._configure_engine(baseline_path)
    client = TestClient(server.fastapi_app)

    response = client.get(
        f"/api/benchmark/compare?baseline={baseline_path}&candidate={candidate_path}&top=5"
    )
    assert response.status_code == 200
    result = response.json()["result"]

    assert result["overlap"]["common"] == 1
    assert result["overlap"]["added"] == 1
    assert result["overlap"]["removed"] == 1
    assert result["improvements"][0]["name"] == "gemm"
    assert result["improvements"][0]["delta"] == 1.0
    assert result["added_benchmarks"][0]["name"] == "transfer"
    assert result["removed_benchmarks"][0]["name"] == "mem"

    reset_engine()
