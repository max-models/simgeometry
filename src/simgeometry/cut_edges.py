"""Geometry of grid-edge intersections, independent of differential operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simgeometry.operations import mask
from simgeometry.queries import intersection_fractions


@dataclass(frozen=True)
class BracketedEdgeGeometry:
    """Single crossings from fluid nodes to solid neighbours.

    ``fractions[(axis, side)]`` is a grid-shaped array (NaN for uncut edges),
    with side 0 lower and 1 upper. Fractions are *not* clamped. This fast path
    assumes one crossing on every outside-to-inside bracket. Use ``cut_edges``
    to inspect multiple hits or objects smaller than a grid cell.
    """

    solid: np.ndarray
    owners: np.ndarray
    fractions: dict
    cut: np.ndarray
    nodes: np.ndarray


def bracketed_edge_geometry(
    grid, objects, *, spacing=None, object_masks=None, periodic=None
):
    """Vectorized edge bisections used by embedded-boundary finite differences.

    Objects overlap with last-listed precedence. No boundary-condition values,
    stencil weights, minimum-distance clamps, or solver policy are applied.
    """
    objects = list(objects or [])
    shape = tuple(int(n) for n in grid.num_gridpoints)
    spacing = tuple(
        float(h) for h in (grid.grid_spacing if spacing is None else spacing)
    )
    periodic = tuple(
        periodic
        if periodic is not None
        else (getattr(grid, "periodic", None) or (False,) * grid.ndim)
    )
    if (
        len(spacing) != grid.ndim
        or any(h <= 0 for h in spacing)
        or len(periodic) != grid.ndim
    ):
        raise ValueError("Spacing and periodicity must match the grid dimension")
    if any(obj.ndim != grid.ndim for obj in objects):
        raise ValueError("Objects and grid must have the same dimension")
    if object_masks is None:
        object_masks = [obj.node_mask(grid) for obj in objects]
    if len(object_masks) != len(objects):
        raise ValueError("Each object needs one node mask")
    solid = np.zeros(shape, dtype=bool)
    owners = np.full(shape, -1, dtype=int)
    for number, m in enumerate(object_masks):
        m = np.asarray(m, dtype=bool)
        if m.shape != shape:
            raise ValueError("Object masks must match the grid shape")
        solid |= m
        owners[m] = number
    nodes = grid.node_points().reshape(grid.ndim, *shape)
    lower = np.asarray(grid.lower_bound)
    periods = np.asarray(shape) * spacing

    def wrap(points):
        points = points.copy()
        for axis, flag in enumerate(periodic):
            if flag:
                offset = (points[axis] - lower[axis]) % periods[axis]
                # a node's image across the face, built as node + step, can land a
                # rounding error short of the period; it is the first node
                offset[periods[axis] - offset <= 1e-9 * periods[axis]] = 0.0
                points[axis] = lower[axis] + offset
        return points

    fractions, cut = {}, np.zeros(shape, dtype=bool)
    for dim in range(grid.ndim):
        for side, offset in enumerate((-1, 1)):
            fraction = np.full(shape, np.nan)
            if periodic[dim]:
                far_solid, far_owners = (
                    np.roll(solid, -offset, axis=dim),
                    np.roll(owners, -offset, axis=dim),
                )
                step = np.zeros((grid.ndim, 1))
                step[dim] = offset * spacing[dim]
                for number, obj in enumerate(objects):
                    mine = ~solid & far_solid & (far_owners == number)
                    if mine.any():
                        outside = nodes[:, mine]
                        fraction[mine] = intersection_fractions(
                            obj, outside, outside + step, wrap=wrap
                        )
            else:
                near, far = [slice(None)] * grid.ndim, [slice(None)] * grid.ndim
                near[dim] = slice(1, None) if offset == -1 else slice(None, -1)
                far[dim] = slice(None, -1) if offset == -1 else slice(1, None)
                near, far = tuple(near), tuple(far)
                for number, obj in enumerate(objects):
                    mine = ~solid[near] & solid[far] & (owners[far] == number)
                    if mine.any():
                        outside = nodes[(slice(None), *near)][:, mine]
                        inside = nodes[(slice(None), *far)][:, mine]
                        fraction[near][mine] = intersection_fractions(
                            obj, outside, inside
                        )
            fractions[dim, side] = fraction
            cut |= np.isfinite(fraction)
    return BracketedEdgeGeometry(solid, owners, fractions, cut, nodes)


@dataclass(frozen=True)
class CutEdges:
    """Every detected hit, ordered by axis, lower-node index and fraction.

    Multiple hits can share an edge index. ``points``/``normals`` are (ndim, n),
    ``indices`` is (n, ndim), and ``object_names`` identifies each boundary.
    This diagnostic path visits every edge; bracketed_edge_geometry is the
    batched solver path. ``exact=False`` identifies a sampled search.
    """

    grid: object
    indices: np.ndarray
    axes: np.ndarray
    fractions: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    object_names: tuple
    exact: bool

    @property
    def distances(self):
        """Distance from the edge's lower node, in physical units."""
        return self.fractions * np.asarray(self.grid.grid_spacing)[self.axes]

    def show(
        self, *, ax=None, normals=False, display=True, save=None, x=None, y=None, z=None
    ):
        """Plot hits and optional normals; a fixed axis selects one grid plane."""
        import matplotlib.pyplot as plt

        axes = list(range(self.grid.ndim))
        selection = [(i, v) for i, v in enumerate((x, y, z)) if v is not None]
        if len(selection) > 1 or any(i >= self.grid.ndim for i, _ in selection):
            raise ValueError("Select at most one existing grid plane")
        keep = np.ones(len(self.fractions), dtype=bool)
        if selection:
            axis, value = selection[0]
            axes.remove(axis)
            keep &= np.isclose(
                self.points[axis],
                value,
                atol=float(self.grid.grid_spacing[axis]) * 1e-7,
                rtol=0,
            )
        if len(axes) == 3:
            ax = ax or plt.figure().add_subplot(projection="3d")
            ax.scatter(*self.points[:, keep], s=12)
            if normals:
                ax.quiver(
                    *self.points[:, keep],
                    *self.normals[:, keep],
                    length=min(self.grid.grid_spacing) * 0.3,
                )
        elif len(axes) == 2:
            ax = ax or plt.subplots()[1]
            p, n = self.points[axes][:, keep], self.normals[axes][:, keep]
            ax.scatter(*p, s=15)
            if normals:
                ax.quiver(
                    *p,
                    *n,
                    angles="xy",
                    scale_units="xy",
                    scale=3 / min(self.grid.grid_spacing),
                )
            ax.set_aspect("equal")
        else:
            ax = ax or plt.subplots()[1]
            ax.scatter(self.points[axes[0], keep], np.zeros(keep.sum()))
        ax.set_xlabel("xyz"[axes[0]])
        if len(axes) > 1:
            ax.set_ylabel("xyz"[axes[1]])
        if save:
            ax.figure.savefig(save, bbox_inches="tight")
        if display:
            plt.show()
        return ax


def cut_edges(geometry, grid, *, samples=64, tolerance=1e-10):
    """Find all surface crossings, including objects between two fluid nodes.

    Nonperiodic primitive edges are analytic. Periodic seam edges use wrapped
    membership and sampling; the periodic discontinuity can itself be a hit.
    General-function features narrower than a sampling interval can be missed.
    """
    from simgeometry.shapes import Shape

    if geometry.ndim != grid.ndim:
        raise ValueError("Geometry and grid dimensions must match")
    objects = getattr(geometry, "named_objects", {"geometry": geometry})
    axes_coords = [np.asarray(a) for a in grid.get_grid_points()]
    periodic = getattr(grid, "periodic", None) or (False,) * grid.ndim
    records, exact = [], True
    for axis in range(grid.ndim):
        counts = [len(a) for a in axes_coords]
        if not periodic[axis]:
            counts[axis] -= 1
        for index in np.ndindex(*counts):
            a = np.array([coords[i] for coords, i in zip(axes_coords, index)])
            b = a.copy()
            b[axis] += grid.grid_spacing[axis]
            for name, obj in objects.items():
                target = obj
                seam = periodic[axis] and index[axis] == len(axes_coords[axis]) - 1
                if seam:

                    class Wrapped(Shape):
                        def __init__(self, source):
                            self.source = source
                            super().__init__(
                                source.ndim, source.lower_bound, source.upper_bound
                            )

                        def contains_points(self, p):
                            p = np.asarray(p).copy()
                            p[axis] = grid.lower_bound[axis] + (
                                p[axis] - grid.lower_bound[axis]
                            ) % (grid.upper_bound[axis] - grid.lower_bound[axis])
                            return self.source.contains_points(p)

                        def contains(self, p):
                            return bool(mask(self, np.asarray(p)[:, None])[0])

                    target = Wrapped(obj)
                hit = target.intersect_segment(
                    a, b, samples=samples, tolerance=tolerance
                )
                exact &= hit.exact
                for i, fraction in enumerate(hit.fractions):
                    records.append(
                        (
                            index,
                            axis,
                            fraction,
                            hit.points[:, i],
                            hit.normals[:, i],
                            name,
                        )
                    )
    records.sort(key=lambda r: (r[1], r[0], r[2], r[5]))
    n = len(records)
    return CutEdges(
        grid,
        np.array([r[0] for r in records], dtype=int).reshape(n, grid.ndim),
        np.array([r[1] for r in records], dtype=int),
        np.array([r[2] for r in records]),
        np.array([r[3] for r in records]).reshape(n, grid.ndim).T,
        np.array([r[4] for r in records]).reshape(n, grid.ndim).T,
        tuple(r[5] for r in records),
        exact,
    )


def diagnose(geometry, grid, *, min_fraction=1e-3, samples=64):
    """Return evidence from node masks and edge hits, not a proof of resolution.

    Reports overlap node counts, objects without solid nodes, edges with multiple
    hits, and hits within min_fraction of an endpoint. A completely unsampled
    generic feature may be absent from both nodes and hits.
    """
    if not 0 <= min_fraction <= 0.5:
        raise ValueError("min_fraction must be between zero and one half")
    objects = getattr(geometry, "named_objects", {"geometry": geometry})
    masks = {name: obj.node_mask(grid) for name, obj in objects.items()}
    names = list(objects)
    overlaps = {
        f"{a} / {b}": int(np.count_nonzero(masks[a] & masks[b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if np.any(masks[a] & masks[b])
    }
    hits = cut_edges(geometry, grid, samples=samples)
    counts = {}
    for index, axis, name in zip(hits.indices, hits.axes, hits.object_names):
        key = (tuple(index), int(axis), name)
        counts[key] = counts.get(key, 0) + 1
    return {
        "overlap_nodes": overlaps,
        "objects_without_solid_nodes": [
            name for name in names if not masks[name].any()
        ],
        "multiple_hit_edges": sum(n > 1 for n in counts.values()),
        "short_distance_hits": int(
            np.count_nonzero(
                np.minimum(hits.fractions, 1 - hits.fractions) < min_fraction
            )
        ),
        "intersection_search_exact": hits.exact,
        "scope": "node and edge samples; does not certify feature resolution",
    }
