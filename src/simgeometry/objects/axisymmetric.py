"""The common frame of the solids of revolution: Cylinder, Tube, Cone and Torus."""

from __future__ import annotations

from abc import abstractmethod

import cunumpy as xp
import numpy as np

from simgeometry._types import Array, BoundsLike
from simgeometry.shapes import Shape3D, _component_major


def _vector(value: BoundsLike, name: str) -> np.ndarray:
    """``value`` as a finite 3-vector."""
    vector = np.asarray(xp.to_numpy(xp.asarray(value, dtype=float)), dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector


class AxisymmetricShape(Shape3D):
    """A solid of revolution about an axis through ``midpoint`` along ``axis``.

    Subclasses describe their profile in the half-plane of the axial coordinate
    ``s`` (along ``axis``, zero at ``midpoint``) and the distance ``rho`` from the
    axis. The level set is the exact signed distance of the profile there, which is
    the exact 3D distance of the solid. A frame along a coordinate axis takes a
    cheaper path than an oblique one.
    """

    has_level_set = True
    level_set_exact = True

    def _set_frame(self, midpoint: np.ndarray, axis: np.ndarray) -> None:
        """Set the centre and the unit axis of the frame."""
        norm = float(np.linalg.norm(axis))
        if norm == 0:
            raise ValueError("The axis must have a nonzero length")
        self._midpoint = np.asarray(midpoint, dtype=float)
        self._axis = np.asarray(axis, dtype=float) / norm
        along = np.flatnonzero(self._axis)
        self._axis_index = (
            int(along[0])
            if along.size == 1 and abs(self._axis[along[0]]) == 1.0
            else None
        )

    @property
    def axis(self) -> tuple[float, float, float]:
        """The unit vector of the symmetry axis."""
        return tuple(float(a) for a in self._axis)

    @property
    def midpoint(self) -> tuple[float, float, float]:
        """The centre of the solid on its axis."""
        return tuple(float(c) for c in self._midpoint)

    def _disc_extent(self, radius: float) -> np.ndarray:
        """Half extent per coordinate of a disc of ``radius`` normal to the axis."""
        return radius * np.sqrt(np.clip(1.0 - self._axis**2, 0.0, None))

    def _axial_radial(self, points: Array, *, squared: bool = False):
        """Axial coordinate ``s`` and distance from the axis (or its square) of points."""
        if self._axis_index is not None:
            k = self._axis_index
            s = (points[k] - float(self._midpoint[k])) * float(self._axis[k])
            others = [i for i in range(3) if i != k]
            d0 = points[others[0]] - float(self._midpoint[others[0]])
            d1 = points[others[1]] - float(self._midpoint[others[1]])
            rho2 = d0 * d0 + d1 * d1
        else:
            relative = points - xp.asarray(self._midpoint)[:, None]
            axis = xp.asarray(self._axis)[:, None]
            s = xp.sum(relative * axis, axis=0)
            across = relative - s[None, :] * axis
            rho2 = xp.sum(across * across, axis=0)
        return s, (rho2 if squared else xp.sqrt(rho2))

    @abstractmethod
    def _inside(self, s: Array, rho2: Array) -> Array:
        """Membership of the geometry from ``s`` and the squared distance ``rho2``."""

    @abstractmethod
    def _profile_distance(self, s: Array, rho: Array) -> Array:
        """Exact signed distance of the profile at ``(s, rho)``."""

    @abstractmethod
    def _bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The lower and upper corners of the bounding box."""

    def _update_bounds(self) -> None:
        """Recompute the bounding box after the frame moved."""
        self.set_bounding_box(*self._bounds())

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        s, rho2 = self._axial_radial(points, squared=True)
        inside = self._inside(s, rho2)
        return ~inside if self.inverted else inside

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the solid.

        Parameters:
        coords (list or tuple): Coordinates (x, y, z).

        Returns:
        bool: True if the point is inside the object, False otherwise.
        """
        point = xp.asarray(coords, dtype=float).reshape(3, 1)
        return bool(xp.to_numpy(self.contains_points(point))[0])

    def _level_set(self, points: Array) -> Array:
        """Exact signed distance to the solid."""
        s, rho = self._axial_radial(points)
        return self._profile_distance(s, rho)

    def translate(self, offset: tuple[float, float, float]) -> None:
        """
        Translate the solid by the given offset.

        Parameters:
            offset (tuple): (dx, dy, dz) translation vector.
        """
        self._midpoint = self._midpoint + _vector(offset, "offset")
        self._update_bounds()

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the solid across the plane where coordinate ``axis`` is ``position``.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis, 2 for z-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis).")
        midpoint, direction = self._midpoint.copy(), self._axis.copy()
        midpoint[axis] = 2 * position - midpoint[axis]
        direction[axis] = -direction[axis]
        self._set_frame(midpoint, direction)
        self._update_bounds()


class CappedAxisymmetricShape(AxisymmetricShape):
    """A solid of revolution between the end points ``start`` and ``end``.

    The axis runs from ``start`` to ``end``; the solid is cut off by the planes
    through them normal to the axis.
    """

    def _set_ends(self, start: BoundsLike, end: BoundsLike) -> None:
        """Set the frame from the two end points of the axis."""
        start, end = _vector(start, "start"), _vector(end, "end")
        length = float(np.linalg.norm(end - start))
        if length == 0:
            raise ValueError("start and end must differ")
        self._half_length = 0.5 * length
        self._set_frame(0.5 * (start + end), end - start)

    @property
    def start(self) -> tuple[float, float, float]:
        """The centre of the end face at the start of the axis."""
        return tuple(float(c) for c in self._midpoint - self._half_length * self._axis)

    @property
    def end(self) -> tuple[float, float, float]:
        """The centre of the end face at the end of the axis."""
        return tuple(float(c) for c in self._midpoint + self._half_length * self._axis)

    @property
    def length(self) -> float:
        """The distance from ``start`` to ``end``."""
        return 2.0 * self._half_length

    @abstractmethod
    def _end_radii(self) -> tuple[float, float]:
        """The outer radius at ``start`` and at ``end``."""

    def _bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The box around the two end discs, which encloses their convex hull.

        A rounded solid is the solid shrunk by its ``fillet``, widened by a ball of
        that radius: the box of the shrunk end discs, widened by the fillet.
        """
        fillet = float(getattr(self, "fillet", 0.0))
        r_start, r_end = (r - fillet for r in self._end_radii())
        start = np.asarray(self.start) + fillet * self._axis
        end = np.asarray(self.end) - fillet * self._axis
        lower = np.minimum(
            start - self._disc_extent(r_start), end - self._disc_extent(r_end)
        )
        upper = np.maximum(
            start + self._disc_extent(r_start), end + self._disc_extent(r_end)
        )
        return (
            tuple(float(v) for v in lower - fillet),
            tuple(float(v) for v in upper + fillet),
        )
