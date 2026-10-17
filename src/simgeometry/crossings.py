"""First boundary crossings of many segments at once, for particle pushers.

:func:`crossings` takes the start and end of every particle's step as component-major
``(ndim, N)`` arrays and finds, for each step, the first point where it crosses the
shape's boundary, entering or leaving. Because the whole step is tested, not only its
end point, a fast particle cannot pass through a thin plate unnoticed.

Polygonal 2D shapes (exact outlines), circles, spheres and axis-aligned boxes are
analytic; affine transforms of them stay analytic. Every other shape (Boolean
combinations, collections, function objects, ...) is sampled: membership is tested at
``samples`` points along each step and the first change is bisected, so features
thinner than ``1 / samples`` of a step can be missed, and the normals are estimated.

Conventions: fractions lie in ``(0, 1]``. A crossing at the very start of a step
(fraction below :data:`START_TOLERANCE`) is not reported, so a particle that stops
exactly on a boundary is not counted again on its next step; a crossing exactly at
the end is. Grazing contacts (a step that touches a circle, or slides along an edge)
are not crossings. Normals have unit length and point out of the region where
``contains`` is true, so they flip for inverted shapes. An open curve (a shape with
``closed = False``) has no inside: its normal faces the incoming step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np

from simgeometry.shapes import _component_major

#: Crossings closer than this fraction to the start of a step are ignored.
START_TOLERANCE = 1e-12
#: Largest number of (segment, edge) pairs a polygon test broadcasts at once.
MAX_PAIRS = 4_000_000


@dataclass(frozen=True)
class Crossings:
    """First boundary crossing of each segment.

    ``hit`` is ``(N,)``; ``fraction`` is ``(N,)`` in ``(0, 1]``; ``point`` and
    ``normal`` are ``(ndim, N)``. Where ``hit`` is false they are NaN. ``exact`` is
    true for analytic results and false for sampled ones.
    """

    hit: Any
    fraction: Any
    point: Any
    normal: Any
    exact: bool

    def __len__(self) -> int:
        return int(self.hit.shape[0])


def crossings(shape, start, end, *, samples: int = 64, iterations: int = 40):
    """Return the first boundary crossing of each segment ``start[:, i] -> end[:, i]``.

    See the module docstring for the conventions and which shapes are analytic.
    """
    a = _component_major(start, shape.ndim)
    b = _component_major(end, shape.ndim)
    if a.shape != b.shape:
        raise ValueError(f"start {a.shape} and end {b.shape} must have the same shape")
    if int(samples) != samples or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    t, normal, exact = _first(shape, a, b - a, int(samples), int(iterations))
    hit = xp.isfinite(t)
    fraction = xp.where(hit, t, xp.nan)
    point = xp.where(hit[None, :], a + xp.where(hit, t, 0.0)[None, :] * (b - a), xp.nan)
    normal = xp.where(hit[None, :], normal, xp.nan)
    return Crossings(hit, fraction, point, normal, exact)


def _first(shape, a, d, samples, iterations):
    """Fraction ``(N,)`` (inf without a hit) and normal ``(ndim, N)`` of the first crossing."""
    from simgeometry.objects import (
        Aperture2D,
        Circle,
        Cuboid,
        Line,
        Polygon,
        Rectangle,
        Sphere,
        Triangle,
    )
    from simgeometry.operations import TransformedShape

    if isinstance(shape, TransformedShape):
        return _transformed(shape, a, d, samples, iterations)
    if isinstance(shape, (Circle, Sphere)):
        return (*_ball(shape, a, d), True)
    if isinstance(shape, (Rectangle, Cuboid, Line)):
        return (*_box(shape, a, d), True)
    polygonal = (Polygon, Triangle, Aperture2D)
    if shape.ndim == 2 and (
        getattr(shape, "exact_outline", False) or isinstance(shape, polygonal)
    ):
        return (*_polygon(shape, a, d), True)
    return (*_sampled(shape, a, d, samples, iterations), False)


def _flip(shape, normal):
    return -normal if shape.inverted else normal


def _first_of_interval(t0, t1):
    """First crossing of a convex shape whose line occupies ``(t0, t1)``, or inf."""
    inf = xp.full(t0.shape, xp.inf)
    entering = (t0 > START_TOLERANCE) & (t0 <= 1.0)
    leaving = ~entering & (t1 > START_TOLERANCE) & (t1 <= 1.0)
    proper = t0 < t1  # touching (t0 == t1) is not a crossing
    t = xp.where(entering, t0, xp.where(leaving, t1, inf))
    return xp.where(proper, t, inf), entering


def _ball(shape, a, d):
    centre = xp.asarray(np.asarray(xp.to_numpy(shape.midpoint), float))[:, None]
    radius = float(shape.radius)
    rel = a - centre
    aa = xp.sum(d * d, axis=0)
    bb = xp.sum(rel * d, axis=0)
    cc = xp.sum(rel * rel, axis=0) - radius**2
    safe = xp.where(aa > 0, aa, 1.0)
    disc = bb * bb - aa * cc
    root = xp.sqrt(xp.where(disc > 0, disc, 0.0))
    valid = (aa > 0) & (disc > 0)
    t0 = xp.where(valid, (-bb - root) / safe, xp.inf)
    t1 = xp.where(valid, (-bb + root) / safe, xp.inf)
    t, _ = _first_of_interval(t0, t1)
    finite = xp.isfinite(t)
    point = a + xp.where(finite, t, 0.0)[None, :] * d
    normal = (point - centre) / radius
    return t, _flip(shape, normal)


def _box(shape, a, d):
    lower = np.asarray(xp.to_numpy(shape.lower_bound), float).reshape(-1)
    upper = np.asarray(xp.to_numpy(shape.upper_bound), float).reshape(-1)
    n = a.shape[1]
    t0 = xp.full(n, -xp.inf)
    t1 = xp.full(n, xp.inf)
    enter_axis = xp.zeros(n, dtype=int)
    exit_axis = xp.zeros(n, dtype=int)
    for k in range(shape.ndim):
        moving = d[k] != 0
        safe = xp.where(moving, d[k], 1.0)
        lo = (lower[k] - a[k]) / safe
        hi = (upper[k] - a[k]) / safe
        near = xp.where(moving, xp.minimum(lo, hi), -xp.inf)
        far = xp.where(moving, xp.maximum(lo, hi), xp.inf)
        # a step parallel to this axis's faces never crosses them; outside the slab
        # it misses the box entirely
        outside = ~moving & ((a[k] < lower[k]) | (a[k] > upper[k]))
        near = xp.where(outside, xp.inf, near)
        far = xp.where(outside, -xp.inf, far)
        enter_axis = xp.where(near > t0, k, enter_axis)
        exit_axis = xp.where(far < t1, k, exit_axis)
        t0 = xp.maximum(t0, near)
        t1 = xp.minimum(t1, far)
    t, entering = _first_of_interval(t0, t1)
    axis = xp.where(entering, enter_axis, exit_axis)
    # leaving through a face moves along +d; entering moves against the face normal
    sign = xp.where(entering, -1.0, 1.0)
    normal = xp.zeros(a.shape)
    for k in range(shape.ndim):
        normal[k] = xp.where(axis == k, sign * xp.sign(d[k]), 0.0)
    return t, _flip(shape, normal)


def _polygon(shape, a, d):
    closed = getattr(shape, "closed", True)
    starts, ends = [], []
    for loop in shape.outlines():
        loop = np.asarray(loop, float)
        if closed:
            starts.append(loop)
            ends.append(np.roll(loop, -1, axis=1))
        else:
            starts.append(loop[:, :-1])
            ends.append(loop[:, 1:])
    p = xp.asarray(np.concatenate(starts, axis=1))
    e = xp.asarray(np.concatenate(ends, axis=1)) - p
    edge_length = xp.sqrt(xp.sum(e * e, axis=0))
    n = a.shape[1]
    chunk = max(1, MAX_PAIRS // max(1, p.shape[1]))
    t = xp.full(n, xp.inf)
    edge = xp.zeros(n, dtype=int)
    for lo in range(0, n, chunk):
        sl = slice(lo, min(n, lo + chunk))
        t_part, edge_part = _polygon_chunk(a[:, sl], d[:, sl], p, e, edge_length)
        t[sl] = t_part
        edge[sl] = edge_part
    # right-hand normal of the edge: out of a counter-clockwise loop's inside
    ex, ey = e[0][edge], e[1][edge]
    length = edge_length[edge]
    normal = xp.stack((ey / length, -ex / length))
    if closed:
        return t, _flip(shape, normal)
    facing = xp.sum(normal * d, axis=0)
    return t, xp.where(facing > 0, -normal, normal)


def _polygon_chunk(a, d, p, e, edge_length):
    d0, d1 = d[0][:, None], d[1][:, None]
    w0 = p[0][None, :] - a[0][:, None]
    w1 = p[1][None, :] - a[1][:, None]
    den = d0 * e[1][None, :] - d1 * e[0][None, :]
    step = xp.sqrt(d0 * d0 + d1 * d1)
    proper = xp.abs(den) > 1e-14 * step * edge_length[None, :]
    safe = xp.where(proper, den, 1.0)
    t = (w0 * e[1][None, :] - w1 * e[0][None, :]) / safe
    u = (w0 * d1 - w1 * d0) / safe
    tol = 1e-12
    valid = (
        proper & (u >= -tol) & (u <= 1 + tol) & (t > START_TOLERANCE) & (t <= 1 + tol)
    )
    t = xp.where(valid, xp.minimum(t, 1.0), xp.inf)
    edge = xp.argmin(t, axis=1) if t.shape[1] else xp.zeros(t.shape[0], dtype=int)
    first = (
        xp.take_along_axis(t, edge[:, None], axis=1)[:, 0]
        if t.shape[1]
        else xp.full(t.shape[0], xp.inf)
    )
    return first, edge


def _transformed(shape, a, d, samples, iterations):
    inverse = xp.asarray(shape.inverse)
    offset = xp.asarray(shape.offset)[:, None]
    local_a = inverse @ (a - offset)
    local_d = inverse @ d
    t, normal, exact = _first(shape.source, local_a, local_d, samples, iterations)
    normal = inverse.T @ xp.where(xp.isfinite(normal), normal, 0.0)
    length = xp.sqrt(xp.sum(normal * normal, axis=0))
    normal = normal / xp.where(length > 0, length, xp.nan)
    return t, normal, exact


def _sampled(shape, a, d, samples, iterations):
    """First membership change along each segment, sampled then bisected."""
    n = a.shape[1]
    ndim = a.shape[0]
    if n == 0:
        return xp.full(0, xp.inf), xp.zeros((ndim, 0))
    ts = xp.linspace(0.0, 1.0, samples + 1)
    state = _inside(shape, a, d, ts)  # (n, samples + 1)
    change = state[:, :-1] != state[:, 1:]
    has = xp.any(change, axis=1)
    index = xp.argmax(change, axis=1)
    lo = ts[index]
    hi = ts[index + 1]
    start_state = xp.take_along_axis(state, index[:, None], axis=1)[:, 0]
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        same = _contains(shape, a + mid[None, :] * d) == start_state
        lo = xp.where(same, mid, lo)
        hi = xp.where(same, hi, mid)
    t = 0.5 * (lo + hi)
    t = xp.where(has & (t > START_TOLERANCE), t, xp.inf)
    return t, _estimated_normal(shape, a + xp.where(has, t, 0.0)[None, :] * d)


def _contains(shape, points):
    return xp.asarray(shape.contains_points(points), dtype=bool)


def _inside(shape, a, d, ts):
    """Membership ``(n, len(ts))`` at ``a + ts * d``, in chunks of at most MAX_PAIRS."""
    n, m = a.shape[1], ts.shape[0]
    rows = []
    chunk = max(1, MAX_PAIRS // m)
    for lo in range(0, n, chunk):
        sl = slice(lo, min(n, lo + chunk))
        points = a[:, sl, None] + d[:, sl, None] * ts[None, None, :]
        k = points.shape[1]
        rows.append(_contains(shape, points.reshape(a.shape[0], k * m)).reshape(k, m))
    return xp.concatenate(rows, axis=0)


def _estimated_normal(shape, points):
    """Unit normal out of the contains-region from membership in many directions.

    The vectorised form of :func:`simgeometry.queries.estimate_surface_normal`: each
    point is probed a small distance away in a fan (2D) or sphere (3D) of directions.
    """
    ndim, n = points.shape
    if ndim == 1:
        directions = np.array([[-1.0, 1.0]])
    elif ndim == 2:
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        directions = np.stack((np.cos(theta), np.sin(theta)))
    else:
        indices = np.arange(128, dtype=float) + 0.5
        z = 1 - 2 * indices / 128
        theta = np.pi * (1 + np.sqrt(5)) * indices
        radial = np.sqrt(1 - z * z)
        directions = np.stack((radial * np.cos(theta), radial * np.sin(theta), z))
    extent = np.asarray(xp.to_numpy(shape.upper_bound), float) - np.asarray(
        xp.to_numpy(shape.lower_bound), float
    )
    scale = float(np.max(extent)) if np.all(np.isfinite(extent)) else 1.0
    epsilon = max(scale * 1e-5, np.finfo(float).eps * 128)
    directions = xp.asarray(directions)
    m = directions.shape[1]
    if n == 0:
        return xp.zeros((ndim, 0))
    probes = points[:, :, None] + epsilon * directions[:, None, :]
    inside = _contains(shape, probes.reshape(ndim, n * m)).reshape(n, m)
    weights = xp.where(inside, -1.0, 1.0)  # (n, m)
    normal = weights @ directions.T  # (n, ndim)
    length = xp.sqrt(xp.sum(normal * normal, axis=1))
    on_boundary = xp.any(inside, axis=1) & ~xp.all(inside, axis=1) & (length > 0)
    normal = normal / xp.where(length > 0, length, 1.0)[:, None]
    return xp.where(on_boundary[None, :], normal.T, xp.nan)
