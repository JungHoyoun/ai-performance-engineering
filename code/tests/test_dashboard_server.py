import json

from typer.testing import CliRunner

from core.engine import get_engine, reset_engine
from dashboard.api import server


def test_configure_engine_uses_data_file(tmp_path):
    data = {
        "timestamp": "2025-01-01 00:00:00",
        "results": [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "example_a",
                        "best_speedup": 2.0,
                        "baseline_time_ms": 100.0,
                        "baseline_gpu_metrics": {"power_draw_w": 250},
                        "optimizations": [],
                        "status": "succeeded",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "benchmark_test_results.json"
    path.write_text(json.dumps(data))

    reset_engine()
    server._configure_engine(path)
    result = get_engine().benchmark.data()

    assert result["summary"]["total_benchmarks"] == 1
    assert result["benchmarks"][0]["name"] == "example_a"

    reset_engine()


def test_dashboard_cli_has_serve_command():
    runner = CliRunner()
    result = runner.invoke(server.cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the dashboard API server." in result.output
