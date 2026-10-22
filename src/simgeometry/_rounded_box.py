"""Line intervals, normals and closest points of boxes with rounded edges.

A box of half sides ``half`` rounded with radius ``radius`` is the box of half sides
``half - radius`` widened by a ball of that radius. It is convex and the union of
simple pieces: the box widened along one axis at a time, and the discs (2D) or the
edge cylinders and corner balls (3D) around the inner box's corners and edges. The
interval of a line inside it is the hull of the pieces' intervals.

The functions take component-major ``(ndim, n)`` arrays and an array module
``array_xp`` (NumPy for host arrays, ``xp`` for the active cunumpy backend).
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np


def _slab(a, d, lo, hi, array_xp):
    """Interval ``(near, far)`` of ``a + t d`` in ``lo <= x <= hi``, empty as (inf, -inf)."""
    moving = d != 0
    safe = array_xp.where(moving, d, 1.0)
    t_lo, t_hi = (lo - a) / safe, (hi - a) / safe
    near = array_xp.where(moving, array_xp.minimum(t_lo, t_hi), -np.inf)
    far = array_xp.where(moving, array_xp.maximum(t_lo, t_hi), np.inf)
    outside = ~moving & ((a < lo) | (a > hi))
    return array_xp.where(outside, np.inf, near), array_xp.where(outside, -np.inf, far)


def rounded_box_interval(a, d, centre, half, radius, array_xp=np):
    """Parameters ``(t0, t1)`` where the lines ``a + t d`` enter and leave the box.

    ``a`` and ``d`` are ``(ndim, n)``; ``centre`` and ``half`` hold one value per
    axis. A line that misses gets ``t0 > t1``; a step of zero length inside gets
    ``(-inf, inf)``.
    """
    ndim, n = a.shape
    rel = a - array_xp.asarray(centre, dtype=float)[:, None]
    half = np.asarray(half, dtype=float)
    inner = half - radius
    slabs = [_slab(rel[k], d[k], -inner[k], inner[k], array_xp) for k in range(ndim)]
    t0 = array_xp.full(n, np.inf)
    t1 = array_xp.full(n, -np.inf)

    def add(near, far):
        nonlocal t0, t1
        kept = near <= far
        t0 = array_xp.minimum(t0, array_xp.where(kept, near, np.inf))
        t1 = array_xp.maximum(t1, array_xp.where(kept, far, -np.inf))

    # the inner box widened by the radius along one axis
    for k in range(ndim):
        near, far = _slab(rel[k], d[k], -half[k], half[k], array_xp)
        for j in range(ndim):
            if j != k:
                near = array_xp.maximum(near, slabs[j][0])
                far = array_xp.minimum(far, slabs[j][1])
        add(near, far)
    # discs, edge cylinders and corner balls: round in the axes of ``axes``
    for size in range(2, ndim + 1):
        for axes in combinations(range(ndim), size):
            for signs in product((-1.0, 1.0), repeat=size):
                aa = sum(d[k] * d[k] for k in axes)
                bb = sum((rel[k] - s * inner[k]) * d[k] for k, s in zip(axes, signs))
                cc = (
                    sum((rel[k] - s * inner[k]) ** 2 for k, s in zip(axes, signs))
                    - radius**2
                )
                moving = aa > 0
                safe = array_xp.where(moving, aa, 1.0)
                disc = bb * bb - aa * cc
                root = array_xp.sqrt(array_xp.where(disc > 0, disc, 0.0))
                cut = moving & (disc >= 0)
                still = ~moving & (cc <= 0)
                near = array_xp.where(
                    cut, (-bb - root) / safe, array_xp.where(still, -np.inf, np.inf)
                )
                far = array_xp.where(
                    cut, (-bb + root) / safe, array_xp.where(still, np.inf, -np.inf)
                )
                for j in range(ndim):
                    if j not in axes:
                        near = array_xp.maximum(near, slabs[j][0])
                        far = array_xp.minimum(far, slabs[j][1])
                add(near, far)
    return t0, t1


def rounded_box_normal(points, centre, half, radius, array_xp=np):
    """Unit outward normals ``(ndim, n)`` of the box at (or near) the points.

    The direction from the nearest point of the inner box; inside the inner box, the
    normal of the nearest face.
    """
    ndim = points.shape[0]
    half = np.asarray(half, dtype=float)
    inner = array_xp.asarray(half - radius)[:, None]
    q = points - array_xp.asarray(centre, dtype=float)[:, None]
    n = q - array_xp.clip(q, -inner, inner)
    length = array_xp.sqrt(array_xp.sum(n * n, axis=0))
    face = array_xp.argmax(array_xp.abs(q) - array_xp.asarray(half)[:, None], axis=0)
    sign = array_xp.where(q >= 0, 1.0, -1.0)
    axes = array_xp.arange(ndim)[:, None]
    flat = array_xp.where(axes == face[None, :], sign, 0.0)
    unit = n / array_xp.where(length > 0, length, 1.0)[None, :]
    return array_xp.where((length > 0)[None, :], unit, flat)


def rounded_box_closest(points, centre, half, radius):
    """Closest points ``(ndim, n)`` on the boundary of the box, on the host."""
    half = np.asarray(half, dtype=float)
    inner = (half - radius)[:, None]
    centre = np.asarray(centre, dtype=float)[:, None]
    q = points - centre
    c = np.clip(q, -inner, inner)
    n = q - c
    length = np.sqrt(np.sum(n * n, axis=0))
    round_part = c + radius * n / np.where(length > 0, length, 1.0)
    # inside the inner box the nearest boundary point is on the nearest face
    face = np.argmax(np.abs(q) - half[:, None], axis=0)
    flat = q.copy()
    columns = np.arange(q.shape[1])
    flat[face, columns] = np.where(q[face, columns] >= 0, 1.0, -1.0) * half[face]
    return centre + np.where(length > 0, round_part, flat)
