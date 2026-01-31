from core.profiling.nvtx_helper import standardize_nvtx_label


def test_standardize_nvtx_label_keeps_known_prefix() -> None:
    assert standardize_nvtx_label("setup:inputs") == "setup:inputs"


def test_standardize_nvtx_label_strips_variants() -> None:
    assert standardize_nvtx_label("baseline_matmul_pytorch") == "compute_math:matmul_pytorch"
    assert standardize_nvtx_label("optimized_matmul_pytorch") == "compute_math:matmul_pytorch"


def test_standardize_nvtx_label_transfer_async() -> None:
    assert standardize_nvtx_label("optimized_memory_copy_async") == "transfer_async:memory_copy_async"


def test_standardize_nvtx_label_graph() -> None:
    assert standardize_nvtx_label("graph_replay") == "compute_graph:graph_replay"


def test_standardize_nvtx_label_empty() -> None:
    assert standardize_nvtx_label("") == "compute_kernel:unnamed"
