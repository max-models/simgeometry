"""The NumPy version of the ``contains_box`` kernel, run when Pyccel is unavailable.

Same arguments and result as ``contains_box_pyccel.py``, vectorized over the markers.
"""

from __future__ import annotations

import numpy as np


def contains_box(
    positions: np.ndarray,
    mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    inverted: int,
) -> None:
    """Set ``mask`` to whether each marker lies in the box ``[lower, upper]``."""
    inside = np.all(
        (positions >= np.asarray(lower)[:, None])
        & (positions <= np.asarray(upper)[:, None]),
        axis=0,
    )
    mask[...] = ~inside if inverted == 1 else inside
