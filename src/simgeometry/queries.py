"""Surface queries, analytic for primitives and sampled for general membership.

All coordinates and results here are NumPy host arrays. The bisection helper
batches containment queries so GPU implementations can still accelerate them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.objects import (
    Circle,
    Cuboid,
    Line,
    Polygon,
    Rectangle,
    Sphere,
    Triangle,
)
from simgeometry.operations import BooleanShape, TransformedShape, mask

BALLS = (Circle, Sphere)
BOXES = (Rectangle, Cuboid, Line)
POLYGONS = (Polygon, Triangle)


def points_array(points, ndim):
    p = np.asarray(xp.to_numpy(points), float)
    single = p.ndim == 1
    if single:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] != ndim or not np.all(np.isfinite(p)):
        raise ValueError(f"Expected finite ({ndim}, n) or ({ndim},) coordinates")
    return p, single


def intersection_fractions(obj, outside, inside, *, iterations=48, wrap=None):
    """Bisect bracketed outside-to-inside edges; return raw, unclamped fractions.

    Endpoints have shape (ndim, n). Each segment must bracket a single crossing;
    use intersect_segment for multiple crossings or equal-membership endpoints.
    ``wrap`` maps periodic images into the domain for membership tests.
    """
    outside, _ = points_array(outside, obj.ndim)
    inside, _ = points_array(inside, obj.ndim)
    if outside.shape != inside.shape or iterations < 1:
        raise ValueError("Endpoints must match and iterations must be positive")
    if wrap is None:

        def wrap(p):
            return p

    if mask(obj, wrap(outside)).any() or not mask(obj, wrap(inside)).all():
        raise ValueError("Bisection requires outside-to-inside brackets")
    lo, hi = np.zeros(outside.shape[1]), np.ones(outside.shape[1])
    step = inside - outside
    for _ in range(iterations):
        mid = (lo + hi) / 2
        hit = mask(obj, wrap(outside + mid * step))
        hi = np.where(hit, mid, hi)
        lo = np.where(hit, lo, mid)
    return hi


@dataclass(frozen=True)
class SegmentIntersections:
    """Ordered hits; points/normals have shape (ndim, nhits).

    Fractions are in [0, 1]. ``entering`` is +1 entering, -1 leaving, 0 for
    tangency. ``exact`` means analytic intersections rather than sampled search.
    Object names are provided for collections. Normals can be NaN at singularities.
    """

    fractions: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    entering: np.ndarray
    object_names: tuple[str | None, ...]
    exact: bool

    def __len__(self):
        return len(self.fractions)


def _polygon_vertices(shape):
    return np.asarray(xp.to_numpy(shape._vertices), float)


def _candidates(shape, a, b, samples, tolerance):
    d = b - a
    if isinstance(shape, TransformedShape):
        return _candidates(
            shape.source,
            shape.inverse @ (a - shape.offset),
            shape.inverse @ (b - shape.offset),
            samples,
            tolerance,
        )
    if isinstance(shape, BooleanShape):
        left, e1 = _candidates(shape.left, a, b, samples, tolerance)
        right, e2 = _candidates(shape.right, a, b, samples, tolerance)
        candidates = sorted(left + right)
        kept = []
        # Remove interfaces hidden by the Boolean operation. Tangencies remain
        # if the membership at the contact differs from its neighbourhood.
        for i, t in enumerate(candidates):
            gaps = [abs(t - s) for s in candidates if abs(t - s) > tolerance]
            epsilon = min([1e-6, *(g / 4 for g in gaps)])
            state = mask(
                shape, a[:, None] + d[:, None] * np.array([t - epsilon, t, t + epsilon])
            )
            if state[0] != state[2] or state[1] != state[0]:
                kept.append(t)
        return kept, e1 and e2
    if isinstance(shape, BALLS):
        rel = a - np.asarray(shape.midpoint)
        aa = np.dot(d, d)
        centre = -np.dot(rel, d) / aa
        nearest = rel + centre * d
        radial = shape.radius**2 - np.dot(nearest, nearest)
        roundoff = (
            32 * np.finfo(float).eps * max(shape.radius**2, np.dot(nearest, nearest))
        )
        if radial < -roundoff:
            return [], True
        half = np.sqrt(max(radial, 0) / aa)
        return [centre - half, centre + half], True
    if isinstance(shape, BOXES):
        lower, upper = np.asarray(shape.lower_bound), np.asarray(shape.upper_bound)
        lo, hi = -np.inf, np.inf
        for i in range(shape.ndim):
            if d[i] == 0:
                if a[i] < lower[i] or a[i] > upper[i]:
                    return [], True
            else:
                ts = sorted(((lower[i] - a[i]) / d[i], (upper[i] - a[i]) / d[i]))
                lo, hi = max(lo, ts[0]), min(hi, ts[1])
        return ([lo, hi] if lo <= hi else []), True
    if isinstance(shape, POLYGONS):
        vertices = _polygon_vertices(shape)
        hits = []

        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]

        for p, q in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = q - p
            denominator = cross(d, edge)
            if abs(denominator) > np.finfo(float).eps * np.linalg.norm(
                d
            ) * np.linalg.norm(edge):
                t, u = cross(p - a, edge) / denominator, cross(p - a, d) / denominator
                if -tolerance <= u <= 1 + tolerance:
                    hits.append(t)
            elif abs(cross(p - a, d)) <= tolerance * np.dot(d, d):
                hits.extend(
                    [np.dot(p - a, d) / np.dot(d, d), np.dot(q - a, d) / np.dot(d, d)]
                )
        return hits, True
    if int(samples) != samples or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    ts = np.linspace(0, 1, int(samples) + 1)
    states = mask(shape, a[:, None] + d[:, None] * ts)
    changes = np.flatnonzero(states[:-1] != states[1:])
    hits = []
    for i in changes:
        lo, hi = ts[i], ts[i + 1]
        for _ in range(48):
            mid = (lo + hi) / 2
            if shape.contains(a + mid * d) == states[i]:
                lo = mid
            else:
                hi = mid
        hits.append((lo + hi) / 2)
    return hits, False


def intersect_segment(shape, start, end, *, samples=256, tolerance=1e-10):
    a, b = np.asarray(start, float), np.asarray(end, float)
    if (
        a.shape != (shape.ndim,)
        or b.shape != a.shape
        or not np.all(np.isfinite([a, b]))
    ):
        raise ValueError("Segment endpoints must be finite coordinate vectors")
    if tolerance <= 0 or not np.isfinite(tolerance) or np.linalg.norm(b - a) == 0:
        raise ValueError("Segment length and tolerance must be positive")
    if hasattr(shape, "named_objects"):
        hits = [
            (name, intersect_segment(obj, a, b, samples=samples, tolerance=tolerance))
            for name, obj in shape.named_objects.items()
        ]
        records = sorted(
            (float(t), name, hit, j)
            for name, hit in hits
            for j, t in enumerate(hit.fractions)
        )
        return SegmentIntersections(
            np.array([r[0] for r in records]),
            np.column_stack([r[2].points[:, r[3]] for r in records])
            if records
            else np.empty((shape.ndim, 0)),
            np.column_stack([r[2].normals[:, r[3]] for r in records])
            if records
            else np.empty((shape.ndim, 0)),
            np.array([r[2].entering[r[3]] for r in records], dtype=int),
            tuple(r[1] for r in records),
            all(h.exact for _, h in hits),
        )
    candidates, exact = _candidates(shape, a, b, samples, tolerance)
    ts = []
    for t in sorted(t for t in candidates if -tolerance <= t <= 1 + tolerance):
        t = float(np.clip(t, 0, 1))
        if not ts or t - ts[-1] > tolerance:
            ts.append(t)
    ts = np.asarray(ts)
    points = a[:, None] + (b - a)[:, None] * ts
    eps = np.full(len(ts), 1e-7)
    if len(ts) > 1:
        eps[:-1] = np.minimum(eps[:-1], np.diff(ts) / 4)
        eps[1:] = np.minimum(eps[1:], np.diff(ts) / 4)
    before = mask(shape, a[:, None] + (b - a)[:, None] * (ts - eps))
    after = mask(shape, a[:, None] + (b - a)[:, None] * (ts + eps))
    normals = surface_normal(shape, points)
    return SegmentIntersections(
        ts,
        points,
        normals,
        after.astype(int) - before.astype(int),
        (None,) * len(ts),
        exact,
    )


def _closest_segments(p, segments):
    a, b = segments[:, 0], segments[:, 1]
    d = b - a
    denominator = np.sum(d * d, axis=1)
    t = np.sum((p - a) * d, axis=1) / np.where(denominator > 0, denominator, 1)
    return a + np.clip(t, 0, 1)[:, None] * d


def closest_on_mesh(points, vertices, cells):
    if not len(cells):
        raise ValueError("No surface resolved; enlarge bounds or increase resolution")
    faces = vertices[cells]
    result = []
    for p in points.T:
        if cells.shape[1] == 2:
            candidates = _closest_segments(p, faces)
        else:
            # Closest projection within a triangle, or its three edges.
            a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
            u, v = b - a, c - a
            n = np.cross(u, v)
            n2 = np.sum(n * n, axis=1)
            q = p - (np.sum((p - a) * n, axis=1) / np.where(n2 > 0, n2, 1))[:, None] * n
            uu, uv, vv = (
                np.sum(u * u, axis=1),
                np.sum(u * v, axis=1),
                np.sum(v * v, axis=1),
            )
            wu, wv = np.sum((q - a) * u, axis=1), np.sum((q - a) * v, axis=1)
            den = uu * vv - uv * uv
            s = (vv * wu - uv * wv) / np.where(den > 0, den, 1)
            t = (uu * wv - uv * wu) / np.where(den > 0, den, 1)
            valid = (den > 0) & (s >= 0) & (t >= 0) & (s + t <= 1)
            candidates = np.concatenate(
                [
                    q[valid],
                    *(
                        _closest_segments(p, faces[:, pair])
                        for pair in ([0, 1], [1, 2], [2, 0])
                    ),
                ]
            )
        result.append(candidates[np.argmin(np.sum((candidates - p) ** 2, axis=1))])
    return np.asarray(result).reshape(-1, points.shape[0]).T


def boundary_points_1d(shape, *, resolution=64, bounds=None):
    """Resolve boundary points of a 1D shape or section by segment crossings."""
    lower, upper = (
        bounds if bounds is not None else (shape.lower_bound, shape.upper_bound)
    )
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if bounds is None:
        margin = (upper - lower) * 0.05
        lower, upper = lower - margin, upper + margin
    hits = shape.intersect_segment(lower, upper, samples=resolution)
    if not len(hits):
        raise ValueError("No boundary resolved; enlarge bounds or increase resolution")
    return hits.points


@sp.profile("simgeometry.closest_point", tags=_PROFILE_TAGS)
def closest_point(shape, points, *, resolution=64, bounds=None):
    p, single = points_array(points, shape.ndim)
    if isinstance(shape, BALLS):
        delta = p - np.asarray(shape.midpoint)[:, None]
        radius = np.linalg.norm(delta, axis=0)
        unit = delta / np.where(radius > 0, radius, 1)
        unit[0, radius == 0] = 1
        q = np.asarray(shape.midpoint)[:, None] + shape.radius * unit
    elif isinstance(shape, BOXES):
        lower, upper = np.asarray(shape.lower_bound), np.asarray(shape.upper_bound)
        q = np.clip(p, lower[:, None], upper[:, None])
        inside = np.all((p >= lower[:, None]) & (p <= upper[:, None]), axis=0)
        for i in np.flatnonzero(inside):
            distances = np.r_[p[:, i] - lower, upper - p[:, i]]
            face = np.argmin(distances)
            axis = face % shape.ndim
            q[axis, i] = lower[axis] if face < shape.ndim else upper[axis]
    elif isinstance(shape, POLYGONS):
        vertices = _polygon_vertices(shape)
        cells = np.column_stack(
            (np.arange(len(vertices)), np.roll(np.arange(len(vertices)), -1))
        )
        q = closest_on_mesh(p, vertices, cells)
    elif isinstance(shape, TransformedShape) and np.allclose(
        shape.matrix.T @ shape.matrix,
        np.eye(shape.ndim) * np.sum(shape.matrix[:, 0] ** 2),
    ):
        q = (
            shape.matrix
            @ closest_point(
                shape.source,
                shape.inverse @ (p - shape.offset[:, None]),
                resolution=resolution,
                bounds=None,
            )
            + shape.offset[:, None]
        )
    elif shape.ndim == 1:
        boundary = boundary_points_1d(shape, resolution=resolution, bounds=bounds)
        closest = np.argmin(abs(p.T - boundary), axis=1)
        q = boundary[:, closest]
    else:
        mesh = shape.to_mesh(resolution=resolution, bounds=bounds)
        q = closest_on_mesh(p, mesh.vertices, mesh.cells)
    return q[:, 0] if single else q


@sp.profile("simgeometry.signed_distance", tags=_PROFILE_TAGS)
def signed_distance(shape, points, *, resolution=64, bounds=None):
    p, single = points_array(points, shape.ndim)
    q = closest_point(shape, p, resolution=resolution, bounds=bounds)
    distance = np.linalg.norm(p - q, axis=0)
    distance = np.where(mask(shape, p), -distance, distance)
    return float(distance[0]) if single else distance


def estimate_surface_normal(obj, point, scale, *, require_boundary=True):
    """Solid-outward unit normal estimated by directional membership samples.

    Set require_boundary=False to retain the original Picnic estimator at
    periodic image points, including its cancellation behavior away from a surface.
    """
    point = np.asarray(point, float)
    ndim = point.size
    if ndim == 1:
        directions = np.asarray([[-1.0], [1.0]])
    elif ndim == 2:
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        directions = np.column_stack((np.cos(theta), np.sin(theta)))
    else:
        indices = np.arange(128, dtype=float) + 0.5
        z = 1 - 2 * indices / 128
        theta = np.pi * (1 + np.sqrt(5)) * indices
        radial = np.sqrt(1 - z * z)
        directions = np.column_stack(
            (radial * np.cos(theta), radial * np.sin(theta), z)
        )
    epsilon = max(scale * 1e-5, np.finfo(float).eps * 128)
    hits = mask(obj, point[:, None] + epsilon * directions.T)
    normal = np.sum(np.where(hits[:, None], -directions, directions), axis=0)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0 or (require_boundary and (hits.all() or not hits.any())):
        return np.full(ndim, np.nan)
    return normal / magnitude


def surface_normal(shape, points, *, scale=None):
    p, single = points_array(points, shape.ndim)
    if isinstance(shape, TransformedShape):
        n = shape.inverse.T @ surface_normal(
            shape.source, shape.inverse @ (p - shape.offset[:, None]), scale=scale
        )
    elif isinstance(shape, BALLS):
        n = p - np.asarray(shape.midpoint)[:, None]
        if shape.inverted:
            n = -n
    elif isinstance(shape, BOXES):
        distances = np.concatenate(
            (
                abs(p - np.asarray(shape.lower_bound)[:, None]),
                abs(p - np.asarray(shape.upper_bound)[:, None]),
            ),
            axis=0,
        )
        face = np.argmin(distances, axis=0)
        n = np.zeros_like(p)
        n[face % shape.ndim, np.arange(p.shape[1])] = np.where(face < shape.ndim, -1, 1)
        if shape.inverted:
            n = -n
    elif isinstance(shape, POLYGONS):
        vertices = _polygon_vertices(shape)
        segments = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
        edges = segments[:, 1] - segments[:, 0]
        signed_area = np.sum(
            vertices[:, 0] * np.roll(vertices[:, 1], -1)
            - vertices[:, 1] * np.roll(vertices[:, 0], -1)
        )
        if signed_area == 0:
            n = np.full_like(p, np.nan)
        else:
            outward = np.column_stack((edges[:, 1], -edges[:, 0])) * np.sign(
                signed_area
            )
            n = np.empty_like(p)
            for i, point in enumerate(p.T):
                closest = _closest_segments(point, segments)
                edge = np.argmin(np.sum((closest - point) ** 2, axis=1))
                n[:, i] = outward[edge]
            if shape.inverted:
                n = -n
    else:
        scale = (
            float(np.max(np.asarray(shape.upper_bound) - shape.lower_bound))
            if scale is None
            else float(scale)
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normal sampling scale must be positive and finite")
        n = (
            np.column_stack(
                [estimate_surface_normal(shape, point, scale) for point in p.T]
            )
            if p.shape[1]
            else np.empty_like(p)
        )
    length = np.linalg.norm(n, axis=0)
    n = n / np.where(length > 0, length, np.nan)
    return n[:, 0] if single else n
