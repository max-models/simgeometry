"""The Metal version (Apple-silicon GPU, float32) of the ``contains_polygon_2d`` kernel.

``contains_polygon_2d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker (``positions`` is ``(2, N)``); see :mod:`simgeometry.kernels._metal`. The test is
done in float32, so a marker within about 1e-7 (relative) of the boundary may be
classified differently from the CPU version.
"""

from __future__ import annotations

import numpy as np

from simgeometry._types import Array
from simgeometry.kernels._metal import make_kernel

_SOURCE = """
    uint marker = thread_position_in_grid.x;
    uint n_markers = positions_shape[1];
    if (marker >= n_markers) return;
    float x = positions[marker];
    float y = positions[n_markers + marker];
    int n = (int)vertices_shape[0];
    bool inside = false;
    float xj = vertices[2 * (n - 1)];
    float yj = vertices[2 * (n - 1) + 1];
    for (int i = 0; i < n; ++i) {
        float xi = vertices[2 * i];
        float yi = vertices[2 * i + 1];
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-16f) + xi) {
            inside = !inside;
        }
        xj = xi;
        yj = yi;
    }
    if (flags[0] == 1) inside = !inside;
    inside_flag[marker] = inside ? 1 : 0;
"""

_KERNEL = make_kernel(
    "simgeometry_contains_polygon_2d",
    _SOURCE,
    inputs=["positions", "vertices", "flags"],
    outputs=["inside_flag"],
)


def contains_polygon_2d(
    positions: Array,
    mask: Array,
    vertices: Array,
    inverted: int,
) -> None:
    """Set ``mask[marker]`` to whether the marker lies inside the polygon."""
    if not positions.shape[1]:
        return
    flags = np.zeros(positions.shape[1], dtype=np.uint8)
    _KERNEL(positions, vertices, np.array([inverted], dtype=np.int32), out=flags)
    mask[:] = flags.astype(bool)
