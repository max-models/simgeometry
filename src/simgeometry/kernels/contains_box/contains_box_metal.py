"""The Metal version (Apple-silicon GPU, float32) of the ``contains_box`` kernel.

``contains_box_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` is ``(ndim, N)``); see :mod:`simgeometry.kernels`. The test is
done in float32, so a marker within about 1e-7 (relative) of the boundary may be
classified differently from the CPU version.
"""

from __future__ import annotations

import numpy as np

from simgeometry._types import Array
from simgeometry.kernels import KERNEL_PACKAGE

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n = positions_shape[1];
    if (marker >= n) return;
    uint ndim = positions_shape[0];
    bool inside = true;
    for (uint a = 0; a < ndim; ++a) {
        float x = positions[a * n + marker];
        if (x < lower[a] || x > upper[a]) inside = false;
    }
    if (flags[0] == 1) inside = !inside;
    inside_flag[marker] = inside ? 1 : 0;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "simgeometry_contains_box",
    _SOURCE,
    inputs=["positions", "lower", "upper", "flags"],
    outputs=["inside_flag"],
)


def contains_box(
    positions: Array,
    mask: Array,
    lower: Array,
    upper: Array,
    inverted: int,
) -> None:
    """Set ``mask[marker]`` to whether the marker lies in the box ``[lower, upper]``."""
    if not positions.shape[1]:
        return
    flags = np.zeros(positions.shape[1], dtype=np.uint8)
    _KERNEL(positions, lower, upper, np.array([inverted], dtype=np.int32), out=flags)
    mask[:] = flags.astype(bool)
