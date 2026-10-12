"""Types used by geometry without importing simulation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from numpy import ndarray

Array: TypeAlias = ndarray
Scalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = (
    Scalar
    | Array
    | list["ConfigValue"]
    | tuple["ConfigValue", ...]
    | set["ConfigValue"]
    | dict[str, "ConfigValue"]
)
Printable: TypeAlias = str | int | float | bool | None
BoundsLike: TypeAlias = Sequence[int | float] | ndarray
Vertices2DLike: TypeAlias = Sequence[BoundsLike] | ndarray
