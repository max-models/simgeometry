"""Boolean composition, affine transformations and queryable plane sections."""

from __future__ import annotations

import copy
from itertools import product

import cunumpy as xp
import numpy as np

from simgeometry.shapes import Shape, _component_major


def corners(shape):
    return np.array(list(product(*zip(shape.lower_bound, shape.upper_bound))), float).T


def mask(shape, points):
    return np.asarray(xp.to_numpy(shape.contains_points(points)), dtype=bool)


class BooleanShape(Shape):
    """A snapshot of a union, intersection, or difference of two shapes.

    Bounds conservatively enclose operands. ``inverted`` indicates the result
    may be unbounded, so sampling requires explicit bounds in that case.
    """

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

    def contains(self, coords):
        return bool(
            xp.to_numpy(self.contains_points(np.asarray(coords).reshape(self.ndim, 1)))[
                0
            ]
        )


class TransformedShape(Shape):
    """Independent affine image of a source, mapping p to matrix @ p + offset."""

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
        flip = np.linalg.det(self.matrix) < 0
        loops = []
        for loop in self.source.outlines(local):
            loop = self.matrix @ loop + self.offset[:, None]
            loops.append(loop[:, ::-1] if flip else loop)
        return loops

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
