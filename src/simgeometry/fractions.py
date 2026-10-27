"""Subcell quadrature for approximate solid volume and face-area fractions."""

from dataclasses import dataclass
from itertools import product

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.operations import mask


@dataclass(frozen=True)
class CellFractions:
    """Sampled solid fractions in [0, 1]; fluid fractions are one minus these.

    ``volume`` has one entry per cell. ``faces[axis]`` has one extra entry along
    that axis, including both periodic boundary copies. Fraction times the cell
    volume/face area gives the corresponding solid measure. This is midpoint
    quadrature, not a reconstructed interface; sub-sample features can be missed.
    """

    volume: np.ndarray
    faces: tuple[np.ndarray, ...]
    samples_per_axis: int


@sp.profile("simgeometry.cell_fractions", tags=_PROFILE_TAGS)
def cell_fractions(geometry, grid, *, samples_per_axis=4):
    """Estimate cell and face occupancy using a regular subcell sample lattice."""
    if geometry.ndim != grid.ndim:
        raise ValueError("Geometry and grid dimensions must match")
    if int(samples_per_axis) != samples_per_axis or samples_per_axis < 1:
        raise ValueError("samples_per_axis must be a positive integer")
    n = int(samples_per_axis)
    axes = [np.asarray(a, float) for a in grid.get_grid_points()]
    periodic = getattr(grid, "periodic", None) or (False,) * grid.ndim
    h = np.asarray(grid.grid_spacing)
    lower = np.asarray(grid.lower_bound)
    upper = np.asarray(grid.upper_bound)
    cell_counts = tuple(
        len(a) if flag else len(a) - 1 for a, flag in zip(axes, periodic)
    )
    starts = [a if flag else a[:-1] for a, flag in zip(axes, periodic)]
    offsets = (np.arange(n) + 0.5) / n

    def sample(base_axes, fixed=None):
        result = np.zeros(tuple(len(a) for a in base_axes))
        base = np.stack([a.ravel() for a in np.meshgrid(*base_axes, indexing="ij")])
        combinations = list(
            product(*([0.0] if axis == fixed else offsets for axis in range(grid.ndim)))
        )
        for delta in combinations:
            points = base + h[:, None] * np.asarray(delta)[:, None]
            for axis, flag in enumerate(periodic):
                if flag:
                    points[axis] = lower[axis] + (points[axis] - lower[axis]) % (
                        upper[axis] - lower[axis]
                    )
            result += mask(geometry, points).reshape(result.shape)
        return result / len(combinations)

    volume = sample(starts).reshape(cell_counts)
    faces = []
    for axis in range(grid.ndim):
        face_axes = list(starts)
        face_axes[axis] = (
            np.r_[axes[axis], upper[axis]] if periodic[axis] else axes[axis]
        )
        faces.append(sample(face_axes, fixed=axis))
    return CellFractions(volume, tuple(faces), n)


#: Relative step past a crossing before the next one is searched for, and the
#: offset of the membership probes beside an edge.
_NUDGE = 1e-9
#: Three-point Gauss-Legendre rule on [0, 1].
_GAUSS_POINTS = 0.5 + 0.5 * np.sqrt(0.6) * np.array([-1.0, 0.0, 1.0])
_GAUSS_WEIGHTS = np.array([5.0, 8.0, 5.0]) / 18.0
#: The turn of a trace's normal between neighbouring lines taken for a corner.
_KINK_SINE = 0.05


@dataclass(frozen=True)
class YeeFractions:
    """Solid fractions of the edges and faces of a staggered (Yee) grid, and the traces.

    The arrays follow the edge and face numbering of a Yee grid on the grid's nodes:

    - ``edges[a]``: the fraction of each edge along axis ``a`` inside the geometry,
      shape: cells along ``a``, nodes across (``cells = nodes - 1``, or ``nodes`` on a
      periodic axis, whose last edge closes the period).
    - ``faces[a]``: the fraction of each face normal to ``a`` inside, shape: nodes
      along ``a``, cells across. In 2D there is one family, ``faces[0]``, the cells of
      the grid (shape: cells), as faces normal to the plane.
    - ``traces[a]``: the length of the surface's trace in each face, ``int dl`` over
      the curves where the surface crosses the face [length].
    - ``projected_traces[a]``: ``int |n x e| dl``, ``n`` the surface's unit normal and
      ``e`` the face's: the trace weighted by the sine of the angle between the
      surface and the face's normal. In 2D it equals ``traces``. Summed over the faces
      of all three families, each times the grid spacing along its normal, it is
      twice the surface's area.

    Fluid (outside) fractions are one minus these. Edge fractions come from all the
    crossings along each edge (``crossings``, repeated); face fractions and traces
    integrate the crossings of lines across the face, piecewise Gauss-Legendre
    between the face's edge crossings, in both directions. Both are exact for planes (straight
    traces) up to the crossings' accuracy; for curved surfaces the face fractions and
    traces converge at second order or better. Shapes without an analytic crossing
    are sampled (``samples`` points per segment), so features thinner than a
    ``samples``-th of a cell can be missed; a feature that crosses no edge of a face and
    misses the face's quadrature lines is missed.

    A surface lying on grid lines (an axis-aligned wall on a node plane) is treated as
    a staircase: points in a wall that lies in the grid plane of an edge or face count
    as inside, whatever ``contains`` returns on the wall (so the edges and faces in it
    are inside, also where the wall covers them only partly), and a wall along a face's
    edge counts for the face on its outside, with sine one.
    """

    edges: tuple[np.ndarray, ...]
    faces: tuple[np.ndarray, ...]
    traces: tuple[np.ndarray, ...]
    projected_traces: tuple[np.ndarray, ...]
    spacing: tuple[float, ...]


def _all_hits(geometry, start, end, *, samples, max_hits):
    """Every crossing of the segments ``start -> end``, ``(ndim, n)``.

    Returns the fractions along each segment, ``(n, k)`` sorted with NaN padding,
    and the unit normals there, ``(ndim, n, k)``. Repeats :func:`crossings` from just
    past the last crossing found.
    """
    from simgeometry.crossings import crossings

    ndim, n = start.shape
    step = end - start
    found_t, found_n = [], []
    begin = np.zeros(n)
    active = np.arange(n)
    for _ in range(max_hits):
        if active.size == 0:
            break
        origin = start[:, active] + begin[active] * step[:, active]
        with np.errstate(invalid="ignore"):  # sampled shapes: inf * 0 for misses
            result = crossings(geometry, origin, end[:, active], samples=samples)
        hit = np.asarray(xp.to_numpy(result.hit), dtype=bool)
        fraction = np.asarray(xp.to_numpy(result.fraction), dtype=float)
        normal = np.asarray(xp.to_numpy(result.normal), dtype=float)
        t = np.full(n, np.nan)
        nrm = np.full((ndim, n), np.nan)
        which = active[hit]
        t[which] = begin[which] + fraction[hit] * (1.0 - begin[which])
        nrm[:, which] = normal[:, hit]
        found_t.append(t)
        found_n.append(nrm)
        follow = t[which] + _NUDGE
        keep = follow < 1.0
        active = which[keep]
        begin[active] = follow[keep]
    if not found_t:
        return np.full((n, 0), np.nan), np.full((ndim, n, 0), np.nan)
    return np.stack(found_t, axis=1), np.stack(found_n, axis=2)


def _staircase_mask(geometry, points, offsets):
    """Membership where a surface on a grid plane counts as inside.

    ``offsets`` are ``(axis, distance)`` pairs: a point is inside if it or one of its
    neighbours at ``+-distance`` along ``axis`` is, so a point in a wall that lies in
    the grid plane through it is inside, whatever ``contains`` returns on the wall.
    """
    inside = mask(geometry, points)
    for axis, distance in offsets:
        for sign in (-1.0, 1.0):
            probe = points.copy()
            probe[axis] += sign * distance
            inside |= mask(geometry, probe)
    return inside


def _inside_length(geometry, start, end, t, offsets=()):
    """Fraction of each segment inside, from its sorted crossings ``t`` ``(n, k)``."""
    n = start.shape[1]
    breaks = np.concatenate(
        [np.zeros((n, 1)), np.where(np.isnan(t), 1.0, t), np.ones((n, 1))], axis=1
    )
    breaks = np.sort(breaks, axis=1)
    middle = 0.5 * (breaks[:, 1:] + breaks[:, :-1])
    step = end - start
    points = start[:, :, None] + step[:, :, None] * middle[None, :, :]
    inside = _staircase_mask(
        geometry, points.reshape(start.shape[0], -1), offsets
    ).reshape(middle.shape)
    return np.sum(np.diff(breaks, axis=1) * inside, axis=1)


def _near(geometry, centres, radius):
    """Points closer to the surface than ``radius`` (all of them without a level set)."""
    from simgeometry._level_set import has_level_set

    if not has_level_set(geometry) or centres.shape[1] == 0:
        return np.ones(centres.shape[1], dtype=bool)
    from simgeometry.queries import level_values

    # a level set is the signed distance or a bound on it (never larger)
    return np.abs(level_values(geometry, centres)) <= radius * (1 + 1e-9)


def _segments_inside(geometry, start, end, *, samples, max_hits, offsets=()):
    """Inside fractions and crossings of segments; far ones by their midpoint."""
    n = start.shape[1]
    length = np.sqrt(np.sum((end - start) ** 2, axis=0))
    middle = 0.5 * (start + end)
    near = _near(geometry, middle, 0.5 * length)
    fraction = mask(geometry, middle).astype(float)
    hits = np.full((n, 0), np.nan)
    chosen = np.flatnonzero(near)
    if chosen.size:
        t, _ = _all_hits(
            geometry,
            start[:, chosen],
            end[:, chosen],
            samples=samples,
            max_hits=max_hits,
        )
        fraction[chosen] = _inside_length(
            geometry, start[:, chosen], end[:, chosen], t, offsets
        )
        hits = np.full((n, t.shape[1]), np.nan)
        hits[chosen] = t
    return fraction, hits


def _lines_across(geometry, starts, stops, along, across, h, samples, max_hits):
    """Integrate lines along ``along`` across faces, between breakpoints ``stops``.

    ``starts`` are the faces' lower corners ``(ndim, m)``, ``stops`` relative
    positions along ``across`` (NaN for none). Returns the inside fraction, the trace
    and the projected trace of every face, and the lines' positions ``(m, p)``,
    crossings ``(m, p, k)`` and in-plane normals ``(2, m, p, k)`` (along, across).
    """
    ndim, m = starts.shape
    stops = np.where(np.isnan(stops), 1.0, stops)
    stops = np.sort(
        np.concatenate([np.zeros((m, 1)), stops, np.ones((m, 1))], axis=1), axis=1
    )
    low, high = stops[:, :-1], stops[:, 1:]
    # three Gauss-Legendre points per piece: inside it, never at a breakpoint, where a
    # crossing appears or disappears
    positions = low[:, :, None] + (high - low)[:, :, None] * _GAUSS_POINTS
    weights = ((high - low)[:, :, None] * _GAUSS_WEIGHTS).reshape(m, -1)
    count = positions.shape[1] * 3
    start = np.repeat(starts, count, axis=1)
    start[across] += positions.reshape(-1) * h[across]
    end = start.copy()
    end[along] += h[along]
    # a surface along the face's edges (on a grid line): the edges along ``across`` at
    # the lines' ends, where the face's side is outside and the side across the edge
    # inside, sampled at the middle of each piece (away from the crossings); it counts
    # for the face outside, with sine one
    nudge = _NUDGE * h[along]
    # a wall in the face's plane counts as inside
    offsets = (
        [(3 - along - across, _NUDGE * h[3 - along - across])] if ndim == 3 else []
    )
    ends_on_edge, ends_differ = [], []
    for point, inward in ((start, 1.0), (end, -1.0)):
        middle = point.reshape(ndim, m, -1, 3)[:, :, :, 1].reshape(ndim, -1)
        near_side, far_side = middle.copy(), middle.copy()
        near_side[along] += inward * nudge
        far_side[along] -= inward * nudge
        near_in = _staircase_mask(geometry, near_side, offsets)
        far_in = _staircase_mask(geometry, far_side, offsets)
        ends_on_edge.append(np.repeat(~near_in & far_in, 3))
        ends_differ.append(np.repeat(near_in != far_in, 3))
    on_edges = (1.0 * ends_on_edge[0] + ends_on_edge[1]).reshape(m, -1)
    along_edges = h[across] * np.sum(weights * on_edges, axis=1)
    t, normal = _all_hits(geometry, start, end, samples=samples, max_hits=max_hits)
    # a wall in the face's plane counts as inside; a crossing with that inside on both
    # sides (the edge of such a wall) is not one
    offsets = (
        [(3 - along - across, _NUDGE * h[3 - along - across])] if ndim == 3 else []
    )
    if offsets and t.shape[1]:
        found = np.isfinite(t)
        rows, cols = np.nonzero(found)
        step = end[:, rows] - start[:, rows]
        sides = [
            _staircase_mask(
                geometry, start[:, rows] + (t[rows, cols] + shift) * step, offsets
            )
            for shift in (-1e-6, 1e-6)
        ]
        hidden = np.zeros_like(found)
        hidden[rows, cols] = sides[0] & sides[1]
        t = np.where(hidden, np.nan, t)
        normal = np.where(hidden[None], np.nan, normal)
    # a crossing at the line's end, on an edge the surface runs along, is the edge's
    # (counted above, for the face outside)
    at_edge = (t > 1.0 - 1e-9) & ends_differ[1][:, None]
    normal = np.where(at_edge[None], 0.0, np.nan_to_num(normal, nan=0.0))
    inside = _inside_length(geometry, start, end, t, offsets).reshape(m, -1)
    n_along = np.abs(normal[along])
    n_plane = np.hypot(normal[along], normal[across])
    plain = np.sum(
        np.where(n_plane > 0, n_along / np.where(n_plane > 0, n_plane, 1.0), 0.0),
        axis=-1,
    ).reshape(m, -1)
    area = np.sum(weights * inside, axis=1)
    trace = along_edges + h[across] * np.sum(weights * plain, axis=1)
    projected = along_edges + h[across] * np.sum(
        weights * np.sum(n_along, axis=-1).reshape(m, -1), axis=1
    )
    k = t.shape[1]
    t = np.where(at_edge, np.nan, t)
    in_plane = np.stack([normal[along], normal[across]]).reshape(2, m, -1, k)
    return (
        area,
        trace,
        projected,
        positions.reshape(m, -1),
        t.reshape(m, -1, k),
        in_plane,
    )


def _kinks(positions, t, normals, h_along, h_across):
    """Corners of the traces between neighbouring lines, relative positions ``(m, j)``.

    Crossings are matched by their order on lines with as many crossings; where the
    normals of a matched pair differ, the tangents through the two crossings meet at
    the corner if the trace is straight on both sides of it. A meeting point strictly
    between the two lines and inside the face is returned (NaN otherwise); on a
    coarsely resolved curved trace it is a harmless extra breakpoint.
    """
    m, p, k = t.shape
    if p < 2 or k == 0:
        return np.full((m, 0), np.nan)
    counts = np.sum(np.isfinite(t), axis=2)
    same = counts[:, :-1] == counts[:, 1:]
    x0, x1 = t[:, :-1] * h_along, t[:, 1:] * h_along  # along, (m, p - 1, k)
    y0 = (positions[:, :-1] * h_across)[:, :, None]
    y1 = (positions[:, 1:] * h_across)[:, :, None]
    a0, b0 = normals[0][:, :-1], normals[1][:, :-1]
    a1, b1 = normals[0][:, 1:], normals[1][:, 1:]
    det = a0 * b1 - b0 * a1
    norm = np.hypot(a0, b0) * np.hypot(a1, b1)
    # a corner, not the slow turn of a resolved curved trace (about 3 degrees)
    turning = np.abs(det) > _KINK_SINE * np.where(norm > 0, norm, np.inf)
    ok = turning & same[:, :, None] & np.isfinite(x0) & np.isfinite(x1)
    safe = np.where(ok, det, 1.0)
    r0 = a0 * x0 + b0 * y0
    r1 = a1 * x1 + b1 * y1
    x = (r0 * b1 - b0 * r1) / safe
    y = (a0 * r1 - r0 * a1) / safe
    inside = ok & (y > y0) & (y < y1) & (x > 0) & (x < h_along)
    corner = np.where(inside, y / h_across, np.nan)
    return corner.reshape(m, -1)


def _face_integrals(
    geometry, shape, starts, b, c, hits, h, nodes, periodic, *, samples, max_hits
):
    """Inside fractions, traces and projected traces of the faces spanned by ``b, c``.

    ``starts`` are the faces' lower corners ``(ndim, n)``; ``hits[a]`` the crossings of
    the edges along ``a``, shaped like the edge array plus one axis.
    """
    ndim, n = starts.shape
    index = np.indices(shape).reshape(ndim, -1)

    def edge_hits(axis, across):
        """Crossings of the face's edges along ``axis`` at its two ``across`` ends."""
        edge = hits[axis]
        edge_shape = edge.shape[:-1]
        flat = edge.reshape(int(np.prod(edge_shape)), edge.shape[-1])
        ends = []
        for shift in (0, 1):
            multi = index.copy()
            multi[across] = multi[across] + shift
            if periodic[across]:
                multi[across] %= nodes[across]
            ends.append(flat[np.ravel_multi_index(tuple(multi), edge_shape)])
        return np.concatenate(ends, axis=1)

    breaks_c = edge_hits(c, b)  # lines along b, positioned along c
    breaks_b = edge_hits(b, c)
    centres = starts.copy()
    centres[b] += 0.5 * h[b]
    centres[c] += 0.5 * h[c]
    area = mask(geometry, centres).astype(float)
    trace = np.zeros(n)
    projected = np.zeros(n)
    cut = np.any(np.isfinite(breaks_c), axis=1) | np.any(np.isfinite(breaks_b), axis=1)
    from simgeometry._level_set import has_level_set

    if has_level_set(geometry):
        cut |= _near(geometry, centres, 0.5 * float(np.hypot(h[b], h[c])))
    else:
        corner = mask(geometry, starts)
        cut |= area.astype(bool) != corner
    chosen = np.flatnonzero(cut)
    if chosen.size == 0:
        return area, trace, projected
    areas = []
    for along, across, breaks in ((b, c, breaks_c), (c, b, breaks_b)):
        stops = np.where(np.isnan(breaks[chosen]), 1.0, breaks[chosen])
        result = _lines_across(
            geometry, starts[:, chosen], stops, along, across, h, samples, max_hits
        )
        # a corner of the surface inside the face (a kink of the integrands): where
        # the tangents of the trace at neighbouring lines meet, a new breakpoint
        kinks = _kinks(*result[3:], h[along], h[across])
        again = np.flatnonzero(np.any(np.isfinite(kinks), axis=1))
        if again.size:
            refined = _lines_across(
                geometry,
                starts[:, chosen[again]],
                np.concatenate([stops[again], kinks[again]], axis=1),
                along,
                across,
                h,
                samples,
                max_hits,
            )
            for value, update in zip(result[:3], refined[:3]):
                value[again] = update
        areas.append(result[0])
        trace[chosen] += result[1]
        projected[chosen] += result[2]
    area[chosen] = 0.5 * (areas[0] + areas[1])
    # the quadrature's round-off next to fully inside or outside faces
    area = np.where(area < 1e-12, 0.0, np.where(area > 1 - 1e-12, 1.0, area))
    return area, trace, projected


@sp.profile("simgeometry.yee_fractions", tags=_PROFILE_TAGS)
def yee_fractions(geometry, grid, *, samples=64, max_hits=16):
    """Edge-length and face-area fractions of a Yee grid cut by ``geometry``.

    For cut-cell (conformal) finite-difference and finite-integration schemes: the
    edges and faces of the grid of nodes ``grid`` (a :class:`~simgeometry.Grid`, 2D or
    3D), with the parts of their lengths and areas inside the geometry and the trace of
    the surface in each face. See :class:`YeeFractions` for the layout and accuracy.
    ``samples`` is the sampling of shapes without analytic crossings and ``max_hits``
    the most crossings counted per segment.
    """
    ndim = grid.ndim
    if geometry.ndim != ndim:
        raise ValueError("Geometry and grid dimensions must match")
    if ndim not in (2, 3):
        raise ValueError("Yee fractions need a 2D or 3D grid")
    if int(max_hits) != max_hits or max_hits < 1:
        raise ValueError("max_hits must be a positive integer")
    periodic = tuple(getattr(grid, "periodic", None) or (False,) * ndim)
    nodes = tuple(int(n) for n in grid.num_gridpoints)
    cells = tuple(n if p else n - 1 for n, p in zip(nodes, periodic))
    if min(cells) < 1:
        raise ValueError("Yee fractions need at least one cell per axis")
    h = np.asarray(xp.to_numpy(grid.grid_spacing), dtype=float)
    lower = np.asarray(xp.to_numpy(grid.lower_bound), dtype=float).reshape(-1)

    def corners(shape):
        index = np.indices(shape).reshape(ndim, -1).astype(float)
        return lower[:, None] + index * h[:, None]

    edges, hits = [], []
    for a in range(ndim):
        shape = tuple(cells[b] if b == a else nodes[b] for b in range(ndim))
        start = corners(shape)
        end = start.copy()
        end[a] += h[a]
        # a wall along the edge (in a grid plane through it) counts as inside
        offsets = [(b, _NUDGE * h[b]) for b in range(ndim) if b != a]
        fraction, t = _segments_inside(
            geometry, start, end, samples=samples, max_hits=max_hits, offsets=offsets
        )
        edges.append(fraction.reshape(shape))
        hits.append(t.reshape(*shape, t.shape[1]))

    families = [(1, 2), (2, 0), (0, 1)] if ndim == 3 else [(0, 1)]
    faces, traces, projected = [], [], []
    for number, (b, c) in enumerate(families):
        normal_axis = number if ndim == 3 else None
        shape = tuple(nodes[d] if d == normal_axis else cells[d] for d in range(ndim))
        result = _face_integrals(
            geometry,
            shape,
            corners(shape),
            b,
            c,
            hits,
            h,
            nodes,
            periodic,
            samples=samples,
            max_hits=max_hits,
        )
        faces.append(result[0].reshape(shape))
        traces.append(result[1].reshape(shape))
        projected.append(result[2].reshape(shape))
    return YeeFractions(
        tuple(edges),
        tuple(faces),
        tuple(traces),
        tuple(projected),
        tuple(float(v) for v in h),
    )
