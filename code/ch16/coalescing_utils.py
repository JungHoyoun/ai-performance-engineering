"""Shared helpers for Chapter 16 coalescing benchmarks."""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_matrix_shape(
    rows: Optional[int],
    cols: Optional[int],
    total_elements: int,
) -> Tuple[int, int]:
    """
    Return a valid (rows, cols) pair for a flattened tensor.

    Some harnesses may overwrite ``rows``/``cols`` when running in smoke mode.
    This helper reconstructs the missing dimension so we can pass fully
    specified shapes to the CUDA extension.
    """
    resolved_rows = int(rows) if rows else 0
    resolved_cols = int(cols) if cols else 0
    total = int(total_elements)

    if resolved_rows and resolved_cols:
        pass
    elif resolved_rows:
        if total % resolved_rows != 0:
            raise RuntimeError(
                f"resolve_matrix_shape: total_elements={total} not divisible by rows={resolved_rows}"
            )
        resolved_cols = total // resolved_rows
    elif resolved_cols:
        if total % resolved_cols != 0:
            raise RuntimeError(
                f"resolve_matrix_shape: total_elements={total} not divisible by cols={resolved_cols}"
            )
        resolved_rows = total // resolved_cols
    else:
        if total == 0:
            raise RuntimeError("resolve_matrix_shape: unable to infer shape for empty tensor")
        resolved_rows = int(total**0.5)
        resolved_rows = max(resolved_rows, 1)
        while total % resolved_rows != 0 and resolved_rows > 1:
            resolved_rows -= 1
        resolved_cols = total // resolved_rows

    if resolved_rows * resolved_cols != total:
        raise RuntimeError(
            f"resolve_matrix_shape: rows*cols ({resolved_rows}*{resolved_cols}) != total_elements ({total})"
        )
    return resolved_rows, resolved_cols
