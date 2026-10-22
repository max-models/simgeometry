"""Boolean composition, affine transformations and queryable plane sections."""

from __future__ import annotations

import copy
from itertools import product

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.shapes import Shape, _component_major, _refine_loop, _signed_area


def corners(shape):
    return np.array(list(product(*zip(shape.lower_bound, shape.upper_bound))), float).T


def mask(shape, points):
    return np.asarray(xp.to_numpy(shape.contains_points(points)), dtype=bool)


class BooleanShape(Shape):
    """A snapshot of a union, intersection, or difference of two shapes.

    Bounds conservatively enclose operands. ``inverted`` indicates the result
    may be unbounded, so sampling requires explicit bounds in that case.

    When both operands have a level set, so does the result: the minimum (union),
    maximum (intersection) or ``max(a, -b)`` (difference) of theirs. That is a valid
    level set and a bound on the distance, but not the exact distance.
    """

    level_set_exact = False

    def __init__(self, left, right, operation):
        if left.ndim != right.ndim:
            raise ValueError("Boolean operands must have the same dimension")
        if operation not in ("union", "intersection", "difference"):
            raise ValueError("Unknown Boolean operation")
        self.left, self.right = copy.deepcopy(left), copy.deepcopy(right)
        self.operation = operation
        self._inverted = bool(left.inverted or right.inverted)
        super().__init__(
            left.ndim,
            np.minimum(left.lower_bound, right.lower_bound),
            np.maximum(left.upper_bound, right.upper_bound),
        )

    def contains_points(self, points):
        points = _component_major(points, self.ndim)
        a, b = self.left.contains_points(points), self.right.contains_points(points)
        if self.operation == "union":
            return a | b
        if self.operation == "intersection":
            return a & b
        return a & ~b

    @property
    def has_level_set(self):
        """Whether both operands have a level set."""
        return bool(
            getattr(self.left, "has_level_set", False)
            and getattr(self.right, "has_level_set", False)
        )

    def level_set(self, points):
        """Min/max combination of the operands' level sets; see the class docstring."""
        if not self.has_level_set:
            raise NotImplementedError("Both Boolean operands need a level set")
        points = _component_major(points, self.ndim)
        a, b = self.left.level_set(points), self.right.level_set(points)
        if self.operation == "union":
            return xp.minimum(a, b)
        if self.operation == "intersection":
            return xp.maximum(a, b)
        return xp.maximum(a, -b)

    def contains(self, coords):
        return bool(
            xp.to_numpy(self.contains_points(np.asarray(coords).reshape(self.ndim, 1)))[
                0
            ]
        )

    @sp.profile("simgeometry.boolean_outlines", tags=_PROFILE_TAGS)
    def outlines(self, max_spacing=None):
        """The boundary loops of a 2D Boolean shape, as ``(2, n)`` host arrays.

        Loops keep the region on their left: outer boundaries are counter-clockwise
        and holes clockwise (a plate minus a disc gives one of each). Each operand
        enters as the polygon through its own ``outlines(max_spacing)``, so a curved
        operand such as a ``Circle`` is approximated by its inscribed polygon; the
        clipped loops are then refined to ``max_spacing`` (see ``Shape2D.outlines``).
        Needs shapely (the ``clip`` extra). Inverted operands, non-2D shapes and
        operands without outlines raise ``NotImplementedError``.
        """
        geometry = _shapely_geometry(self, max_spacing)
        loops = []
        for polygon in _polygons(geometry):
            loops.append(_shapely_ring(polygon.exterior, counter_clockwise=True))
            loops.extend(
                _shapely_ring(ring, counter_clockwise=False)
                for ring in polygon.interiors
            )
        loops = [loop for loop in loops if loop.shape[1] >= 3]
        if max_spacing is None:
            return loops
        return [_refine_loop(loop, max_spacing) for loop in loops]

    def boundary_panels(self, max_spacing=None):
        """The edges of :meth:`outlines` as panels; see ``Shape2D.boundary_panels``."""
        from simgeometry.panels import boundary_panels

        return boundary_panels(self, max_spacing)


def _shapely():
    try:
        import shapely
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Outlines of Boolean shapes need shapely: pip install 'simgeometry[clip]'"
        ) from exc
    return shapely


def _shapely_geometry(shape, max_spacing):
    """A shapely geometry of a 2D shape, built from its outline loops."""
    shapely = _shapely()
    if shape.ndim != 2:
        raise NotImplementedError("outlines() is only defined for 2D shapes")
    if shape.inverted:
        raise NotImplementedError(
            "outlines() of a Boolean shape needs bounded operands, not inverted ones"
        )
    if isinstance(shape, BooleanShape):
        left = _shapely_geometry(shape.left, max_spacing)
        right = _shapely_geometry(shape.right, max_spacing)
        result = {
            "union": shapely.union,
            "intersection": shapely.intersection,
            "difference": shapely.difference,
        }[shape.operation](left, right)
        return shapely.make_valid(result)
    from simgeometry.objects.function_object import FunctionObject2D

    if isinstance(shape, FunctionObject2D) or not hasattr(shape, "outlines"):
        raise NotImplementedError(
            f"{type(shape).__name__} has no outlines to combine in a Boolean shape"
        )
    if not getattr(shape, "closed", True):
        raise NotImplementedError(
            f"{type(shape).__name__} is an open curve and encloses no area"
        )
    outer, holes = [], []
    for loop in shape.outlines(max_spacing):
        loop = np.asarray(loop, dtype=float)
        if loop.shape[1] < 3:
            continue
        (outer if _signed_area(loop) > 0 else holes).append(shapely.Polygon(loop.T))
    geometry = shapely.union_all(outer) if outer else shapely.Polygon()
    if holes:
        geometry = shapely.difference(geometry, shapely.union_all(holes))
    return shapely.make_valid(geometry)


def _polygons(geometry):
    """The nonempty polygons of a shapely geometry, in a deterministic order."""
    shapely = _shapely()
    polygons = []
    for part in shapely.get_parts(geometry):
        if isinstance(part, shapely.Polygon):
            if part.area > 0:
                polygons.append(part)
        elif isinstance(part, (shapely.MultiPolygon, shapely.GeometryCollection)):
            polygons.extend(_polygons(part))
    return sorted(polygons, key=lambda p: (p.bounds[0], p.bounds[1], -p.area))


def _shapely_ring(ring, *, counter_clockwise):
    """A shapely ring as a ``(2, n)`` loop, without repeated vertices."""
    coords = np.asarray(ring.coords, dtype=float)[:-1]
    keep = np.any(coords != np.roll(coords, 1, axis=0), axis=1)
    loop = np.ascontiguousarray(coords[keep].T)
    if (_signed_area(loop) > 0) != counter_clockwise:
        loop = np.ascontiguousarray(loop[:, ::-1])
    return loop


class TransformedShape(Shape):
    """Independent affine image of a source, mapping p to matrix @ p + offset.

    The level set is the source's at the mapped-back point, times the smallest
    singular value of ``matrix``, which keeps it a bound on the distance. Rigid
    motions and uniform scalings (similarities) keep an exact level set exact;
    nonuniform scalings and shears do not.
    """

    def __init__(self, source, matrix, offset=None):
        self.source = copy.deepcopy(source)
        self.matrix = np.asarray(matrix, float)
        self.offset = (
            np.zeros(source.ndim) if offset is None else np.asarray(offset, float)
        )
        if self.matrix.shape != (source.ndim, source.ndim) or self.offset.shape != (
            source.ndim,
        ):
            raise ValueError("Matrix and offset must match the shape dimension")
        if not np.all(np.isfinite(self.matrix)) or not np.all(np.isfinite(self.offset)):
            raise ValueError("Transform must be finite")
        try:
            self.inverse = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Transform must be invertible") from exc
        vertices = self.matrix @ corners(source) + self.offset[:, None]
        singular = np.linalg.svd(self.matrix, compute_uv=False)
        self._stretch = float(singular.min())
        self._similarity = bool(np.allclose(singular, self._stretch, rtol=1e-12))
        self._inverted = source.inverted
        self._bc = getattr(source, "_bc", None)
        self._bc_type = getattr(source, "_bc_type", "dirichlet")
        super().__init__(source.ndim, vertices.min(axis=1), vertices.max(axis=1))

    def contains_points(self, points):
        points = _component_major(points, self.ndim)
        local = xp.asarray(self.inverse) @ (points - xp.asarray(self.offset)[:, None])
        return self.source.contains_points(local)

    def contains(self, coords):
        return self.source.contains(self.inverse @ (np.asarray(coords) - self.offset))

    @property
    def has_level_set(self):
        """Whether the source has a level set."""
        return bool(getattr(self.source, "has_level_set", False))

    @property
    def level_set_exact(self):
        """Exact for an exact source under a rigid motion or uniform scaling."""
        return bool(getattr(self.source, "level_set_exact", False) and self._similarity)

    def level_set(self, points):
        """The source's level set, scaled by the smallest stretch of the transform."""
        if not self.has_level_set:
            raise NotImplementedError(f"{type(self.source).__name__} has no level set")
        points = _component_major(points, self.ndim)
        local = xp.asarray(self.inverse) @ (points - xp.asarray(self.offset)[:, None])
        return self._stretch * self.source.level_set(local)

    def outlines(self, max_spacing=None):
        """The source's outline loops, transformed (2D only); see ``Shape2D.outlines``.

        The source is refined to ``max_spacing`` divided by the transform's largest
        stretch, so the transformed vertices are at most ``max_spacing`` apart.
        """
        if self.ndim != 2 or not hasattr(self.source, "outlines"):
            raise NotImplementedError("outlines() needs a 2D source with outlines")
        stretch = float(np.linalg.norm(self.matrix, 2))
        if max_spacing is None:
            local = None
        elif callable(max_spacing):

            def local(points):
                world = self.matrix @ points + self.offset[:, None]
                return np.asarray(max_spacing(world), dtype=float) / stretch

        else:
            local = max_spacing / stretch
        # a reflection reverses the loops; an open curve keeps its vertex order
        flip = np.linalg.det(self.matrix) < 0 and self.closed
        loops = []
        for loop in self.source.outlines(local):
            loop = self.matrix @ loop + self.offset[:, None]
            loops.append(loop[:, ::-1] if flip else loop)
        return loops

    @property
    def closed(self):
        """Whether the source encloses an area (False for an open curve)."""
        return getattr(self.source, "closed", True)

    @property
    def exact_outline(self):
        """Whether the outline is exact: an affine map keeps a polygon a polygon."""
        return getattr(self.source, "exact_outline", False)

    def boundary_panels(self, max_spacing=None):
        """The edges of :meth:`outlines` as panels; see ``Shape2D.boundary_panels``."""
        from simgeometry.panels import boundary_panels

        return boundary_panels(self, max_spacing)

    @property
    def named_objects(self):
        if not hasattr(self.source, "named_objects"):
            raise AttributeError("This transformed shape has no named children")
        return {
            name: obj.transformed(self.matrix, self.offset)
            for name, obj in self.source.named_objects.items()
        }

    @property
    def names(self):
        return getattr(self.source, "names", ("inside",))

    @property
    def styles(self):
        return getattr(self.source, "styles", {})

    def labels(self, points):
        if not hasattr(self.source, "labels"):
            return mask(self, points).astype(int)
        local = self.inverse @ (np.asarray(xp.to_numpy(points)) - self.offset[:, None])
        return self.source.labels(local)

    def evaluate(self, *args, quantity=None, **kwargs):
        if quantity is None:
            quantity = "labels" if hasattr(self.source, "labels") else "occupancy"
        return super().evaluate(*args, quantity=quantity, **kwargs)


def rotation(shape, angle, *, axis=(0, 0, 1), origin=None):
    angle = np.deg2rad(float(angle))
    c, s = np.cos(angle), np.sin(angle)
    if shape.ndim == 2:
        matrix = np.array([[c, -s], [s, c]])
    elif shape.ndim == 3:
        axis = np.asarray(axis, float)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) == 0
        ):
            raise ValueError("Rotation axis must be a finite nonzero 3-vector")
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        matrix = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * cross
    else:
        raise ValueError("Rotation requires 2D or 3D geometry")
    origin = np.zeros(shape.ndim) if origin is None else np.asarray(origin, float)
    return shape.transformed(matrix, origin - matrix @ origin)


class Section(Shape):
    """A membership-preserving section, including holes and disconnected pieces.

    ``to_world`` and ``to_local`` use component-major arrays. ``to_local`` is
    an orthogonal projection. For oblique planes, x/y denote plane coordinates.
    """

    def __init__(self, source, *, x=None, y=None, z=None, origin=None, normal=None):
        if source.ndim < 2:
            raise ValueError("Cannot slice one-dimensional geometry")
        self.source = copy.deepcopy(source)
        selection = {k: v for k, v in zip("xyz", (x, y, z)) if v is not None}
        if origin is not None or normal is not None:
            if selection or source.ndim != 3 or origin is None or normal is None:
                raise ValueError(
                    "An oblique section requires only a 3D origin and normal"
                )
            self.origin, n = np.asarray(origin, float), np.asarray(normal, float)
            if (
                self.origin.shape != (3,)
                or n.shape != (3,)
                or not np.all(np.isfinite(n))
                or np.linalg.norm(n) == 0
            ):
                raise ValueError("Plane origin and nonzero normal must be 3-vectors")
            n = n / np.linalg.norm(n)
            u = np.eye(3)[np.argmin(np.abs(n))]
            u -= n * np.dot(u, n)
            u /= np.linalg.norm(u)
            self.basis = np.column_stack((u, np.cross(n, u)))
            self.axis_names = ("x", "y")
        else:
            if len(selection) != 1:
                raise ValueError("Fix exactly one axis to form a section")
            name, value = next(iter(selection.items()))
            source_names = getattr(source, "axis_names", tuple("xyz"[: source.ndim]))
            if name not in source_names or not np.isscalar(value):
                raise ValueError(
                    "Slice axis must exist and its position must be scalar"
                )
            axis = source_names.index(name)
            self.origin = np.zeros(source.ndim)
            self.origin[axis] = float(value)
            remaining = [i for i in range(source.ndim) if i != axis]
            self.basis = np.eye(source.ndim)[:, remaining]
            self.axis_names = tuple(source_names[i] for i in remaining)
        if not np.all(np.isfinite(self.origin)):
            raise ValueError("Plane origin must be finite")
        vertices = self.basis.T @ (corners(source) - self.origin[:, None])
        self._inverted = source.inverted
        super().__init__(source.ndim - 1, vertices.min(axis=1), vertices.max(axis=1))

    def _to_parent(self, points):
        points = np.asarray(points, float)
        return (
            self.origin + self.basis @ points
            if points.ndim == 1
            else self.origin[:, None] + self.basis @ points
        )

    def to_world(self, points):
        """Map section coordinates back through any parent sections."""
        parent = self._to_parent(points)
        return (
            self.source.to_world(parent) if isinstance(self.source, Section) else parent
        )

    def to_local(self, points):
        """Orthogonally project world points through any parent sections."""
        points = np.asarray(points, float)
        if isinstance(self.source, Section):
            points = self.source.to_local(points)
        return self.basis.T @ (
            points - self.origin if points.ndim == 1 else points - self.origin[:, None]
        )

    def contains_points(self, points):
        points = np.asarray(xp.to_numpy(_component_major(points, self.ndim)), float)
        return self.source.contains_points(self._to_parent(points))

    def contains(self, coords):
        return self.source.contains(self._to_parent(np.atleast_1d(coords)))

    #: A 3D distance restricted to a plane only bounds the in-plane distance.
    level_set_exact = False

    @property
    def has_level_set(self):
        """Whether the sliced source has a level set."""
        return bool(getattr(self.source, "has_level_set", False))

    def level_set(self, points):
        """The source's level set on the plane: a bound on the in-plane distance."""
        if not self.has_level_set:
            raise NotImplementedError(f"{type(self.source).__name__} has no level set")
        points = np.asarray(xp.to_numpy(_component_major(points, self.ndim)), float)
        return self.source.level_set(self._to_parent(points))

    def labels(self, points):
        if not hasattr(self.source, "labels"):
            return mask(self, points).astype(int)
        return self.source.labels(self._to_parent(points))

    def evaluate(self, *args, quantity=None, **kwargs):
        if quantity is None:
            quantity = "labels" if hasattr(self.source, "labels") else "occupancy"
        return super().evaluate(*args, quantity=quantity, **kwargs)

    @property
    def names(self):
        return getattr(self.source, "names", ("inside",))

    def outlines(self, max_spacing=None, *, resolution=128, bounds=None, world=False):
        """The sampled boundary loops of a plane section, as ``(2, n)`` host arrays.

        Loops follow ``Shape2D.outlines``: no repeated first vertex, and the region on
        their left, so outer boundaries are counter-clockwise and holes clockwise (a
        hollow sphere gives one of each). They come from contouring the section on
        ``resolution`` samples per axis across ``bounds`` (default: the section's
        bounds, padded by 5%), on the level set where the source has one, so vertices
        lie on the boundary up to the sampling accuracy. A loop cut by ``bounds`` stays
        open. ``max_spacing`` (a length, or a function of ``(2, m)`` plane points)
        splits longer edges evenly, as for Boolean outlines. ``world=True`` maps the
        loops onto the plane in 3D, as ``(3, n)`` arrays. Requires the ``mesh`` extra;
        sections of 2D shapes (lines) have no outlines.
        """
        loops = self._sampled_loops(max_spacing, resolution, bounds)
        return [self.to_world(loop) if world else loop for loop, _ in loops]

    def _sampled_loops(self, max_spacing, resolution, bounds):
        """``(loop, closed)`` pairs of :meth:`outlines`, in plane coordinates."""
        from simgeometry.mesh import _find_contours

        if self.ndim != 2:
            raise NotImplementedError("outlines() needs a 2D section of a 3D shape")
        if (
            max_spacing is not None
            and not callable(max_spacing)
            and not max_spacing > 0
        ):
            raise ValueError("max_spacing must be positive")
        contours, _ = _find_contours(self, resolution=resolution, bounds=bounds)
        loops = []
        for contour in contours:
            closed = len(contour) > 3 and np.array_equal(contour[0], contour[-1])
            loop = contour[:-1] if closed else contour
            loop = loop[np.any(np.diff(loop, axis=0, prepend=np.nan) != 0, axis=1)]
            if len(loop) < 2:
                continue
            loop = np.ascontiguousarray(loop.T)
            if max_spacing is not None:
                loop = _refine_loop(loop, max_spacing, closed=closed)
            loops.append((loop, closed))
        return loops

    def show_3d(self, **kwargs):
        """The section in 3D: its source, the cutting plane and the section outline.

        See :func:`simgeometry.visualization.show_section_3d`; returns Matplotlib
        axes, or a PyVista plotter for ``backend='pyvista'``.
        """
        from simgeometry.visualization import show_section_3d

        return show_section_3d(self, **kwargs)
