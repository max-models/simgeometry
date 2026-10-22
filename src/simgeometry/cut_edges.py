"""Geometry of grid-edge intersections, independent of differential operators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.operations import mask
from simgeometry.queries import intersection_fractions


@dataclass(frozen=True)
class BracketedEdgeGeometry:
    """Single crossings from fluid nodes to solid neighbours.

    ``fractions[(axis, side)]`` is a grid-shaped array (NaN for uncut edges),
    with side 0 lower and 1 upper. Fractions are *not* clamped. This fast path
    assumes one crossing on every outside-to-inside bracket. Use ``cut_edges``
    to inspect multiple hits or objects smaller than a grid cell.

    ``points`` and ``normals`` are the crossings in world coordinates, (ndim, n),
    in the order of ``fractions``: key by key, each as ``f[np.isfinite(f)]``.
    """

    solid: np.ndarray
    owners: np.ndarray
    fractions: dict
    cut: np.ndarray
    nodes: np.ndarray
    spacing: tuple
    periodic: tuple
    objects: tuple

    @cached_property
    def _crossings(self):
        ndim = self.nodes.shape[0]
        lower = self.nodes.reshape(ndim, -1)[:, 0]
        periods = np.asarray(self.solid.shape) * np.asarray(self.spacing)
        points, owners = [np.empty((ndim, 0))], [np.empty(0, dtype=int)]
        for (axis, side), fraction in self.fractions.items():
            cut = np.isfinite(fraction)
            offset = 1 if side else -1
            point = self.nodes[:, cut]
            point[axis] += offset * self.spacing[axis] * fraction[cut]
            points.append(point)
            owners.append(np.roll(self.owners, -offset, axis=axis)[cut])
        points = _wrap(np.concatenate(points, axis=1), lower, periods, self.periodic)
        return points, np.concatenate(owners)

    @property
    def points(self):
        """Crossing points in world coordinates, (ndim, n); periodic images wrapped."""
        return self._crossings[0]

    @property
    def crossing_owners(self):
        """Index into ``objects`` of the object each crossing enters, (n,)."""
        return self._crossings[1]

    @cached_property
    def normals(self):
        """Outward unit normals of the entered object at ``points``, (ndim, n)."""
        points, owners = self._crossings
        normals = np.full_like(points, np.nan)
        for number, obj in enumerate(self.objects):
            mine = owners == number
            if mine.any():
                normals[:, mine] = obj.surface_normal(points[:, mine])
        return normals

    def show_3d(self, **kwargs):
        """Draw the objects, the crossings and optionally normals and solid nodes.

        Takes the keyword arguments of :meth:`CutEdges.show_3d`; several objects
        are drawn as one collection.
        """
        from simgeometry.collection import GeometryCollection

        if len(self.objects) == 1:
            shape = self.objects[0]
        else:
            shape = GeometryCollection(list(self.objects))
        return _show_crossings_3d(
            shape,
            self.points,
            self.normals if kwargs.get("normals") else None,
            self.nodes[:, self.solid],
            self.spacing,
            **kwargs,
        )


def _wrap(points, lower, periods, periodic):
    """Map points across periodic faces into the domain."""
    points = points.copy()
    for axis, flag in enumerate(periodic):
        if flag:
            offset = (points[axis] - lower[axis]) % periods[axis]
            # a node's image across the face, built as node + step, can land a
            # rounding error short of the period; it is the first node
            offset[periods[axis] - offset <= 1e-9 * periods[axis]] = 0.0
            points[axis] = lower[axis] + offset
    return points


def _show_crossings_3d(
    shape,
    points,
    normal_vectors,
    solid_nodes,
    spacing,
    *,
    normals=False,
    nodes=False,
    surface=True,
    backend="matplotlib",
    ax=None,
    resolution=48,
    bounds=None,
    opacity=0.25,
    labels=True,
    point_size=None,
    normal_length=None,
    display=True,
    save=None,
):
    from simgeometry.visualization import _finish, show_3d

    if points.shape[0] != 3:
        raise ValueError("show_3d requires a three-dimensional grid")
    if backend not in ("matplotlib", "pyvista"):
        raise ValueError("backend must be matplotlib or pyvista")
    length = 0.5 * min(spacing) if normal_length is None else float(normal_length)
    if normals:
        finite = np.all(np.isfinite(normal_vectors), axis=0)
        points_n, normal_vectors = points[:, finite], normal_vectors[:, finite]
    if surface:
        ax = show_3d(
            shape,
            ax=ax,
            backend=backend,
            resolution=resolution,
            bounds=bounds,
            opacity=opacity,
            labels=labels,
            show_bounds=False,
            display=False,
        )
    if backend == "pyvista":
        if ax is None:
            from importlib import import_module

            try:
                pv = import_module("pyvista")
            except ImportError as exc:
                raise ImportError(
                    "Interactive 3D needs simgeometry[interactive]"
                ) from exc
            ax = pv.Plotter(off_screen=not display)
        if nodes and solid_nodes.shape[1]:
            ax.add_points(
                solid_nodes.T,
                color="tab:blue",
                point_size=10 if point_size is None else 1.25 * point_size,
                render_points_as_spheres=True,
                name="solid nodes",
            )
        if points.shape[1]:
            ax.add_points(
                points.T,
                color="black",
                point_size=8 if point_size is None else point_size,
                render_points_as_spheres=True,
                name="crossings",
            )
        if normals and points_n.shape[1]:
            ax.add_arrows(
                points_n.T,
                normal_vectors.T,
                mag=length,
                color="tab:red",
                name="normals",
            )
        if display or save:
            ax.show(screenshot=str(save) if save else None, auto_close=False)
        return ax
    import matplotlib.pyplot as plt

    ax = ax if ax is not None else plt.figure().add_subplot(projection="3d")
    if not hasattr(ax, "set_zlabel"):
        raise ValueError("Pass 3D axes to show_3d")
    size = 12 if point_size is None else point_size
    if nodes and solid_nodes.shape[1]:
        ax.scatter(*solid_nodes, s=size, color="tab:blue", alpha=0.5)
    ax.scatter(*points, s=size, color="black")
    if normals and points_n.shape[1]:
        ax.quiver(
            *points_n,
            *normal_vectors,
            length=length,
            color="tab:red",
        )
    return _finish(ax, display, save)


@sp.profile("simgeometry.bracketed_edge_geometry", tags=_PROFILE_TAGS)
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
        return _wrap(points, lower, periods, periodic)

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
    return BracketedEdgeGeometry(
        solid, owners, fractions, cut, nodes, spacing, periodic, tuple(objects)
    )


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
    geometry: object

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

    def show_3d(
        self,
        *,
        normals=False,
        nodes=False,
        surface=True,
        backend="matplotlib",
        ax=None,
        resolution=48,
        bounds=None,
        opacity=0.25,
        labels=True,
        point_size=None,
        normal_length=None,
        display=True,
        save=None,
    ):
        """Draw the translucent surface, the crossings, and optional normals and nodes.

        Crossings are black, outward normals red arrows (``normal_length``, default
        half a grid spacing) and the solid grid nodes blue. ``surface``,
        ``resolution``, ``bounds``, ``opacity`` and ``labels`` go to ``show_3d``.
        Returns the 3D axes or, for ``backend="pyvista"``, the plotter, which the
        caller should close; its actors ``"crossings"``, ``"normals"`` and
        ``"solid nodes"`` hold the points and arrows.
        """
        nodes_inside = np.empty((self.grid.ndim, 0))
        if nodes:
            solid = self.geometry.node_mask(self.grid).ravel()
            nodes_inside = self.grid.node_points()[:, solid]
        return _show_crossings_3d(
            self.geometry,
            self.points,
            self.normals,
            nodes_inside,
            tuple(float(h) for h in self.grid.grid_spacing),
            normals=normals,
            nodes=nodes,
            surface=surface,
            backend=backend,
            ax=ax,
            resolution=resolution,
            bounds=bounds,
            opacity=opacity,
            labels=labels,
            point_size=point_size,
            normal_length=normal_length,
            display=display,
            save=save,
        )


@sp.profile("simgeometry.cut_edges", tags=_PROFILE_TAGS)
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
        geometry,
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
