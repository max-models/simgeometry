"""Axisymmetric (x, r) grids: node volumes, ring volumes and radial sampling.

A marker in cylindrical coordinates stands for a ring of charge around the axis. Its weight
is the number of physical particles in the ring. Markers are placed uniformly in ``(x, r)``
and carry weights proportional to ``r`` (:func:`radial_weight_factors`), so every cell holds
about the same number of markers, also next to the axis where beams are extracted.
"""

from __future__ import annotations

import numpy as np


def node_volumes(num_gridpoints, spacing) -> np.ndarray:
    """Return the volume [m^3] belonging to each node of an ``(x, r)`` grid with the
    axis at ``r = 0``, shape ``(nx, nr)``.

    The volumes are those of linear (cloud-in-cell) weighting, so a uniform density of
    ring markers deposits a uniform density on every node: ``2 pi r_j h_r`` per unit
    length inside, ``pi h_r^2 / 3`` on the axis and ``pi h_r (r_N - h_r / 3)`` on the
    outer node, times ``h_x`` (``h_x / 2`` on the first and last x nodes).
    """
    nx, nr = (int(n) for n in num_gridpoints)
    hx, hr = (float(h) for h in spacing)
    r = hr * np.arange(nr)
    area = 2.0 * np.pi * r * hr
    area[0] = np.pi * hr**2 / 3.0
    area[-1] = np.pi * hr * (r[-1] - hr / 3.0)
    length = np.full(nx, hx)
    length[[0, -1]] = 0.5 * hx
    return length[:, None] * area[None, :]


def ring_volume(lower, upper) -> float:
    """Return the volume of the annular region ``lower <= (x, r) <= upper``."""
    (x0, r0), (x1, r1) = lower, upper
    return float(np.pi * (r1**2 - r0**2) * (x1 - x0))


def radial_weight_factors(r, r0: float, r1: float):
    """Return the weight factors ``2 r / (r0 + r1)`` (mean 1) of ring markers placed
    uniformly in ``r`` on ``[r0, r1]`` that stand for a uniform density."""
    return 2.0 * r / (r0 + r1) if r1 + r0 > 0.0 else 0.0 * r + 1.0
