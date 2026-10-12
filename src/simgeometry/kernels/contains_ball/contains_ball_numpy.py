"""The NumPy version of the ``contains_ball`` kernel, run when Pyccel is unavailable.

Same arguments and result as ``contains_ball_pyccel.py``, vectorized over the markers.
"""

from __future__ import annotations

import numpy as np


def contains_ball(
    positions: np.ndarray,
    mask: np.ndarray,
    midpoint: np.ndarray,
    radius: float,
    inverted: int,
) -> None:
    """Set ``mask`` to whether each marker lies within ``radius`` of ``midpoint``."""
    inside = (
        np.sum((positions - np.asarray(midpoint)[:, None]) ** 2, axis=0)
        <= radius * radius
    )
    mask[...] = ~inside if inverted == 1 else inside
