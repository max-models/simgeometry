"""Normalize scalar and sequence geometry arguments."""

from __future__ import annotations

from collections.abc import Iterable


def ensure_dim_tuple(value: object, *, ndim: int | None = None) -> tuple:
    if isinstance(value, tuple):
        normalized = value
    elif isinstance(value, list):
        normalized = tuple(value)
    elif isinstance(value, (str, bytes, dict)):
        raise TypeError(
            f"value must be a scalar number/bool or sequence, got {type(value)}"
        )
    elif isinstance(value, Iterable):
        normalized = tuple(value)
    else:
        normalized = (value,)
    if ndim is not None and len(normalized) != ndim:
        raise ValueError(f"value must have length {ndim}, got {len(normalized)}")
    return normalized
