"""The Metal version (Apple-silicon GPU, float32) of the ``contains_ball`` kernel.

``contains_ball_pyccel.py`` next to this file is the CPU version with the same
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
    float distance_squared = 0.0f;
    for (uint a = 0; a < ndim; ++a) {
        float delta = positions[a * n + marker] - midpoint[a];
        distance_squared += delta * delta;
    }
    bool inside = distance_squared <= params[0] * params[0];
    if (params[1] == 1.0f) inside = !inside;
    inside_flag[marker] = inside ? 1 : 0;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "simgeometry_contains_ball",
    _SOURCE,
    inputs=["positions", "midpoint", "params"],
    outputs=["inside_flag"],
)


def contains_ball(
    positions: Array,
    mask: Array,
    midpoint: Array,
    radius: float,
    inverted: int,
) -> None:
    """Set ``mask[marker]`` to whether the marker lies within ``radius`` of ``midpoint``."""
    if not positions.shape[1]:
        return
    flags = np.zeros(positions.shape[1], dtype=np.uint8)
    _KERNEL(
        positions, midpoint, np.array([radius, inverted], dtype=np.float32), out=flags
    )
    mask[:] = flags.astype(bool)
