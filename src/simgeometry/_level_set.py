"""Signed-distance formulas of the primitives and root finding on level sets.

The formulas are the standard exact signed distances of boxes, segments, capped
cones and tori (see Inigo Quilez's 2D and 3D distance functions), written for
component-major arrays. They use only ``xp`` operations, so they run on the active
cunumpy backend (NumPy, or CuPy on a GPU).
"""

from __future__ import annotations

from collections.abc import Callable

import cunumpy as xp
import numpy as np

#: Most Illinois (modified regula falsi) steps before a crossing falls back to
#: bisection of the membership test.
SECANT_STEPS = 12
#: Largest number of (point, edge) pairs a polygon distance broadcasts at once.
MAX_PAIRS = 4_000_000


def has_level_set(shape) -> bool:
    """Whether ``shape`` provides :meth:`~simgeometry.shapes.Shape.level_set`."""
    return bool(getattr(shape, "has_level_set", False))


def box_distance(q, half):
    """Signed distance of the points ``q`` (``(ndim, n)``) to the centred box ``half``.

    ``half`` holds the half side lengths, one per row of ``q``.
    """
    d = xp.abs(q) - half
    outside = xp.sqrt(xp.sum(xp.maximum(d, 0.0) ** 2, axis=0))
    inside = xp.minimum(xp.max(d, axis=0), 0.0)
    return outside + inside


def rectangle_distance(x, y, half_x, half_y):
    """Signed distance of ``(x, y)`` to the rectangle ``|x| <= half_x, |y| <= half_y``."""
    dx = xp.abs(x) - half_x
    dy = xp.abs(y) - half_y
    outside = xp.sqrt(xp.maximum(dx, 0.0) ** 2 + xp.maximum(dy, 0.0) ** 2)
    return outside + xp.minimum(xp.maximum(dx, dy), 0.0)


def capped_cone_distance(radial, axial, half_length, r_low, r_high):
    """Signed distance of ``(radial, axial)`` to a frustum of revolution.

    The frustum spans ``-half_length <= axial <= half_length`` with radius ``r_low``
    at the low end and ``r_high`` at the high end; ``radial`` is the distance from
    its axis.
    """
    # distance to the nearer cap: the disc of radius r_low or r_high
    cap_radius = xp.where(axial < 0.0, r_low, r_high)
    cap_x = radial - xp.minimum(radial, cap_radius)
    cap_y = xp.abs(axial) - half_length
    # distance to the slanted side, the segment from (r_high, h) to (r_low, -h)
    side_x, side_y = r_low - r_high, -2.0 * half_length
    length2 = side_x * side_x + side_y * side_y
    t = xp.clip(
        ((radial - r_high) * side_x + (axial - half_length) * side_y) / length2,
        0.0,
        1.0,
    )
    slant_x = radial - (r_high + t * side_x)
    slant_y = axial - (half_length + t * side_y)
    inside = (slant_x < 0.0) & (cap_y < 0.0)
    distance = xp.sqrt(
        xp.minimum(cap_x * cap_x + cap_y * cap_y, slant_x * slant_x + slant_y * slant_y)
    )
    return xp.where(inside, -distance, distance)


def segment_distance(points, starts, ends):
    """Unsigned distance of the ``(2, n)`` points to the nearest of the segments.

    ``starts`` and ``ends`` are ``(2, m)``. Points are processed in chunks so that
    at most :data:`MAX_PAIRS` point-segment pairs are held at once.
    """
    n, m = points.shape[1], starts.shape[1]
    edge = ends - starts
    length2 = xp.sum(edge * edge, axis=0)
    safe = xp.where(length2 > 0, length2, 1.0)
    result = xp.empty(n)
    chunk = max(1, MAX_PAIRS // max(1, m))
    for lo in range(0, n, chunk):
        p = points[:, lo : lo + chunk]
        wx = p[0][:, None] - starts[0][None, :]
        wy = p[1][:, None] - starts[1][None, :]
        t = xp.clip((wx * edge[0][None, :] + wy * edge[1][None, :]) / safe, 0.0, 1.0)
        dx = wx - t * edge[0][None, :]
        dy = wy - t * edge[1][None, :]
        result[lo : lo + chunk] = xp.sqrt(xp.min(dx * dx + dy * dy, axis=1))
    return result


def illinois_roots(
    level: Callable,
    start,
    step,
    t_a,
    t_b,
    f_a,
    f_b,
    tolerance: float,
    steps: int = SECANT_STEPS,
    array_xp=np,
):
    """Roots of ``level(start + t * step)`` bracketed by ``t_a`` and ``t_b``.

    ``level`` maps ``(ndim, m)`` points to ``(m,)`` values; ``f_a`` and ``f_b`` are its
    values at ``t_a`` and ``t_b`` and must have opposite signs. Each Illinois step
    replaces the bracket end of the same sign as the new value by the secant root and
    halves the other end's value if it was kept twice in a row, so both ends converge
    (superlinearly for a smooth level set). An entry has converged when the secant
    step or the bracket is shorter than ``tolerance``, or the level set is down to
    the round-off of the coordinates (``eps * |x|``), below which no step can
    improve the root; converged entries leave the working set.

    ``array_xp`` is the array module of the arguments: NumPy for host arrays, ``xp``
    for arrays on the active cunumpy backend.

    Returns ``(t, t_a, t_b, converged)``: the root estimates and the final brackets,
    with ``t_a`` still on the side of the first ``f_a``.
    """
    ta, tb = array_xp.array(t_a, dtype=float), array_xp.array(t_b, dtype=float)
    fa, fb = array_xp.array(f_a, dtype=float), array_xp.array(f_b, dtype=float)
    out_t, out_a, out_b = ta.copy(), ta.copy(), tb.copy()
    done = array_xp.zeros(fa.shape, dtype=bool)
    active = array_xp.arange(fa.shape[0])
    t = (ta * fb - tb * fa) / (fb - fa)
    # the absolute round-off of level-set values near these points
    noise = np.finfo(float).eps * (
        array_xp.max(array_xp.abs(start), axis=0)
        + array_xp.max(array_xp.abs(step), axis=0)
    )
    replaced_a = None  # which end the previous step replaced
    for _ in range(steps):
        if active.shape[0] == 0:
            break
        f = level(start + t[None, :] * step)
        replace_a = (f > 0) == (fa > 0)
        if replaced_a is not None:
            # Illinois: halve the value of an end kept for the second time in a row
            fb = array_xp.where(replace_a & replaced_a, 0.5 * fb, fb)
            fa = array_xp.where(~(replace_a | replaced_a), 0.5 * fa, fa)
        ta = array_xp.where(replace_a, t, ta)
        fa = array_xp.where(replace_a, f, fa)
        tb = array_xp.where(replace_a, tb, t)
        fb = array_xp.where(replace_a, fb, f)
        replaced_a = replace_a
        root = array_xp.abs(f) <= noise
        t_next = array_xp.where(root, t, (ta * fb - tb * fa) / (fb - fa))
        converged = (
            root
            | (array_xp.abs(t_next - t) <= tolerance)
            | (array_xp.abs(tb - ta) <= tolerance)
        )
        t = t_next
        if bool(array_xp.any(converged)):
            index = active[converged]
            out_t[index], out_a[index], out_b[index] = (
                t[converged],
                ta[converged],
                tb[converged],
            )
            done[index] = True
            keep = ~converged
            active, t, ta, tb, fa, fb, replaced_a, noise = (
                v[keep] for v in (active, t, ta, tb, fa, fb, replaced_a, noise)
            )
            start, step = start[:, keep], step[:, keep]
    out_t[active], out_a[active], out_b[active] = t, ta, tb
    return out_t, out_a, out_b, done


def level_set_gradient(shape, points, spacing: float, array_xp=np):
    """Central-difference gradient ``(ndim, n)`` of ``shape.level_set`` at the points.

    ``array_xp`` is the array module of ``points``, as in :func:`illinois_roots`.
    """

    def level(p):
        values = shape.level_set(p)
        return (
            np.asarray(xp.to_numpy(values), dtype=float) if array_xp is np else values
        )

    ndim = points.shape[0]
    gradient = array_xp.empty(points.shape)
    for axis in range(ndim):
        offset = array_xp.zeros((ndim, 1))
        offset[axis] = spacing
        gradient[axis] = (level(points + offset) - level(points - offset)) / (
            2.0 * spacing
        )
    return gradient
