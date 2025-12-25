"""Minimal nvCOMP Zstd decompressor shim exposed as cupy.cuda.nvcomp."""

from __future__ import annotations

import ctypes
import platform
import sys
from typing import Optional

import cupy as cp

_NVCOMP_LIB: Optional[ctypes.CDLL] = None


class NvcompError(RuntimeError):
    """Raised when nvCOMP calls fail."""


def _nvcomp_lib_path() -> str:
    arch = platform.machine()
    if arch == "x86_64":
        return "/usr/lib/x86_64-linux-gnu/nvcomp/13/libnvcomp.so"
    if arch == "aarch64":
        return "/usr/lib/aarch64-linux-gnu/nvcomp/13/libnvcomp.so"
    raise NvcompError(f"Unsupported architecture for nvCOMP: {arch}")


def _load_nvcomp() -> ctypes.CDLL:
    global _NVCOMP_LIB
    if _NVCOMP_LIB is not None:
        return _NVCOMP_LIB
    lib_path = _nvcomp_lib_path()
    try:
        _NVCOMP_LIB = ctypes.CDLL(lib_path)
    except OSError as exc:
        raise NvcompError(
            f"Failed to load nvCOMP from {lib_path}. "
            "Install libnvcomp5-dev-cuda-13 and ensure the library path is readable."
        ) from exc
    _bind_nvcomp_symbols(_NVCOMP_LIB)
    return _NVCOMP_LIB


class nvcompBatchedZstdDecompressOpts_t(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("reserved", ctypes.c_char * 60),
    ]


NVCOMP_DECOMPRESS_BACKEND_DEFAULT = 0
NVCOMP_SUCCESS = 0


def _bind_nvcomp_symbols(lib: ctypes.CDLL) -> None:
    lib.nvcompBatchedZstdDecompressGetTempSizeAsync.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        nvcompBatchedZstdDecompressOpts_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.nvcompBatchedZstdDecompressGetTempSizeAsync.restype = ctypes.c_int
    lib.nvcompBatchedZstdDecompressAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        nvcompBatchedZstdDecompressOpts_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.nvcompBatchedZstdDecompressAsync.restype = ctypes.c_int


def _check_status(status: int, name: str) -> None:
    if status != NVCOMP_SUCCESS:
        raise NvcompError(f"{name} failed with nvcompStatus={status}")


class ZstdDecompressor:
    """Device-side Zstd decompression using nvCOMP."""

    def __init__(self, backend: int = NVCOMP_DECOMPRESS_BACKEND_DEFAULT) -> None:
        self._lib = _load_nvcomp()
        self._opts = nvcompBatchedZstdDecompressOpts_t(backend=backend, reserved=b"\x00" * 60)
        self._comp_ptrs = cp.empty((1,), dtype=cp.uintp)
        self._comp_sizes = cp.empty((1,), dtype=cp.uint64)
        self._out_ptrs = cp.empty((1,), dtype=cp.uintp)
        self._out_sizes = cp.empty((1,), dtype=cp.uint64)
        self._actual_sizes = cp.empty((1,), dtype=cp.uint64)
        self._status = cp.empty((1,), dtype=cp.int32)

    def decompress(self, comp_gpu: cp.ndarray, uncompressed_bytes: int) -> cp.ndarray:
        if comp_gpu.dtype != cp.uint8:
            comp_gpu = comp_gpu.view(cp.uint8)
        if not comp_gpu.flags.c_contiguous:
            comp_gpu = cp.ascontiguousarray(comp_gpu)
        if uncompressed_bytes <= 0:
            raise NvcompError("uncompressed_bytes must be positive")

        num_chunks = 1
        self._comp_ptrs[0] = comp_gpu.data.ptr
        self._comp_sizes[0] = comp_gpu.nbytes
        self._out_sizes[0] = uncompressed_bytes
        out = cp.empty(int(uncompressed_bytes), dtype=cp.uint8)
        self._out_ptrs[0] = out.data.ptr

        temp_bytes = ctypes.c_size_t()
        status = self._lib.nvcompBatchedZstdDecompressGetTempSizeAsync(
            ctypes.c_size_t(num_chunks),
            ctypes.c_size_t(uncompressed_bytes),
            self._opts,
            ctypes.byref(temp_bytes),
            ctypes.c_size_t(uncompressed_bytes),
        )
        _check_status(status, "nvcompBatchedZstdDecompressGetTempSizeAsync")

        temp_buf = None
        temp_ptr = 0
        if temp_bytes.value:
            temp_buf = cp.empty(int(temp_bytes.value), dtype=cp.uint8)
            temp_ptr = int(temp_buf.data.ptr)

        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        status = self._lib.nvcompBatchedZstdDecompressAsync(
            ctypes.c_void_p(int(self._comp_ptrs.data.ptr)),
            ctypes.c_void_p(int(self._comp_sizes.data.ptr)),
            ctypes.c_void_p(int(self._out_sizes.data.ptr)),
            ctypes.c_void_p(int(self._actual_sizes.data.ptr)),
            ctypes.c_size_t(num_chunks),
            ctypes.c_void_p(temp_ptr),
            ctypes.c_size_t(temp_bytes.value),
            ctypes.c_void_p(int(self._out_ptrs.data.ptr)),
            self._opts,
            ctypes.c_void_p(int(self._status.data.ptr)),
            ctypes.c_void_p(stream_ptr),
        )
        _check_status(status, "nvcompBatchedZstdDecompressAsync")
        return out


def register_nvcomp() -> None:
    """Expose this module as cupy.cuda.nvcomp."""
    module = sys.modules[__name__]
    sys.modules.setdefault("cupy.cuda.nvcomp", module)
    if not hasattr(cp.cuda, "nvcomp"):
        cp.cuda.nvcomp = module  # type: ignore[attr-defined]
