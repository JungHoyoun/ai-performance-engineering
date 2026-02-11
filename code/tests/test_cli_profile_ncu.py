"""CLI integration tests for `aisp profile ncu` with the new launch-limiting options."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ncu_available() -> bool:
    """Check if ncu is available on PATH."""
    return shutil.which("ncu") is not None


def _nvcc_available() -> bool:
    """Check if nvcc is available on PATH."""
    return shutil.which("nvcc") is not None


# Minimal CUDA kernel source for testing
MINIMAL_CUDA_SOURCE = r'''
#include <cstdio>

__global__ void test_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(idx * idx);
    }
}

int main() {
    constexpr int N = 1024;
    float* d_out = nullptr;
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Launch kernel multiple times to test launch limiting
    for (int i = 0; i < 10; ++i) {
        test_kernel<<<4, 256>>>(d_out, N);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_out);
    printf("OK\n");
    return 0;
}
'''


@pytest.fixture(scope="module")
def compiled_cuda_binary(tmp_path_factory):
    """Compile a minimal CUDA binary for testing."""
    if not _nvcc_available():
        pytest.skip("nvcc not available")
    
    tmp_dir = tmp_path_factory.mktemp("cuda_test")
    source_path = tmp_dir / "test_kernel.cu"
    binary_path = tmp_dir / "test_kernel"
    
    source_path.write_text(MINIMAL_CUDA_SOURCE)
    
    # Compile with -lineinfo for better NCU source mapping
    compile_cmd = [
        "nvcc",
        "-lineinfo",
        "-o", str(binary_path),
        str(source_path),
    ]
    result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"nvcc compilation failed: {result.stderr}")
    
    return binary_path


@pytest.mark.cuda
@pytest.mark.skipif(not _ncu_available(), reason="ncu not available")
def test_cli_profile_ncu_help():
    """Test that aisp profile ncu --help shows new options."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "profile", "ncu", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Help failed: {result.stderr}"
    
    # Check that new options are documented
    help_text = result.stdout.lower()
    assert "--launch-skip" in help_text or "launch-skip" in help_text
    assert "--launch-count" in help_text or "launch-count" in help_text
    assert "--metric-set" in help_text or "metric-set" in help_text
    assert "--replay-mode" in help_text or "replay-mode" in help_text


@pytest.mark.cuda
@pytest.mark.skipif(not _ncu_available(), reason="ncu not available")
def test_cli_profile_ncu_minimal_metric_set(compiled_cuda_binary, tmp_path):
    """Test aisp profile ncu with --metric-set minimal."""
    output_dir = tmp_path / "ncu_output"
    output_dir.mkdir()
    
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.aisp", "profile", "ncu",
            "--command", str(compiled_cuda_binary),
            "--output-dir", str(output_dir),
            "--output-name", "minimal_test",
            "--metric-set", "minimal",
            "--timeout", "60",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    
    # Check command succeeded
    assert result.returncode == 0, f"NCU profiling failed: {result.stderr}\n{result.stdout}"
    
    # Check output mentions success
    assert "✓" in result.stdout or "success" in result.stdout.lower() or "ncu-rep" in result.stdout


@pytest.mark.cuda
@pytest.mark.skipif(not _ncu_available(), reason="ncu not available")
def test_cli_profile_ncu_launch_limiting(compiled_cuda_binary, tmp_path):
    """Test aisp profile ncu with --launch-skip and --launch-count."""
    output_dir = tmp_path / "ncu_output"
    output_dir.mkdir()
    
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.aisp", "profile", "ncu",
            "--command", str(compiled_cuda_binary),
            "--output-dir", str(output_dir),
            "--output-name", "launch_limit_test",
            "--metric-set", "minimal",
            "--launch-skip", "2",
            "--launch-count", "1",
            "--timeout", "60",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    
    assert result.returncode == 0, f"NCU with launch limiting failed: {result.stderr}\n{result.stdout}"
    
    # Check that the output references the limiting options
    combined_output = result.stdout + result.stderr
    # The command should have run successfully
    assert "✓" in result.stdout or "success" in result.stdout.lower() or "ncu-rep" in result.stdout


@pytest.mark.cuda
@pytest.mark.skipif(not _ncu_available(), reason="ncu not available")
def test_cli_profile_ncu_kernel_replay_mode(compiled_cuda_binary, tmp_path):
    """Test aisp profile ncu with --replay-mode kernel."""
    output_dir = tmp_path / "ncu_output"
    output_dir.mkdir()
    
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.aisp", "profile", "ncu",
            "--command", str(compiled_cuda_binary),
            "--output-dir", str(output_dir),
            "--output-name", "kernel_replay_test",
            "--metric-set", "minimal",
            "--replay-mode", "kernel",
            "--timeout", "60",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    
    assert result.returncode == 0, f"NCU with kernel replay failed: {result.stderr}\n{result.stdout}"


@pytest.mark.cuda
@pytest.mark.skipif(not _ncu_available(), reason="ncu not available")
def test_cli_profile_ncu_kernel_filter(compiled_cuda_binary, tmp_path):
    """Test aisp profile ncu with --kernel-filter."""
    output_dir = tmp_path / "ncu_output"
    output_dir.mkdir()
    
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.aisp", "profile", "ncu",
            "--command", str(compiled_cuda_binary),
            "--output-dir", str(output_dir),
            "--output-name", "kernel_filter_test",
            "--kernel-filter", "test_kernel",
            "--metric-set", "minimal",
            "--launch-skip", "1",
            "--launch-count", "1",
            "--timeout", "60",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    
    assert result.returncode == 0, f"NCU with kernel filter failed: {result.stderr}\n{result.stdout}"


@pytest.mark.cuda
@pytest.mark.skipif(not _ncu_available(), reason="ncu not available")
def test_cli_profile_ncu_script_mode(tmp_path):
    """Test aisp profile ncu with a Python script (script mode)."""
    output_dir = tmp_path / "ncu_output"
    output_dir.mkdir()
    
    # Create a minimal Python script that does some GPU work
    script_path = tmp_path / "gpu_script.py"
    script_path.write_text('''
import torch
if not torch.cuda.is_available():
    print("CUDA not available")
    exit(1)
x = torch.randn(1000, 1000, device="cuda")
for _ in range(16):
    y = torch.mm(x, x)
torch.cuda.synchronize()
print("OK")
''')
    
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.aisp", "profile", "ncu",
            str(script_path),
            "--output-dir", str(output_dir),
            "--output-name", "script_test",
            "--metric-set", "minimal",
            "--launch-skip", "5",
            "--launch-count", "1",
            "--timeout", "120",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    
    # Script mode should work
    assert result.returncode == 0, f"NCU script mode failed: {result.stderr}\n{result.stdout}"
