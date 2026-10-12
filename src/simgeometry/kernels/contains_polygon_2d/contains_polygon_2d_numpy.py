"""The NumPy version of the ``contains_polygon_2d`` kernel, run when Pyccel is unavailable.

Same arguments and result as ``contains_polygon_2d_pyccel.py``, vectorized over
the markers (a loop over the vertices).
"""

from __future__ import annotations

import numpy as np


def contains_polygon_2d(
    positions: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
    inverted: int,
) -> None:
    """Set ``mask`` to whether each marker lies inside the polygon."""
    x = positions[0]
    y = positions[1]
    inside = np.zeros(positions.shape[1], dtype=bool)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        inside ^= ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-16) + xi
        )
        xj, yj = xi, yi
    mask[...] = ~inside if inverted == 1 else inside
