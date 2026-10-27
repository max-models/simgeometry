"""Cuboids, optionally with rounded edges."""

from __future__ import annotations

from operator import add, sub

import cunumpy as xp
import numpy as np

from simgeometry._level_set import box_distance
from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.shapes import Shape3D, _component_major


class Cuboid(Shape3D):
    """Represent a cuboid geometry object.

    ``fillet`` rounds the twelve edges and eight corners with that radius, at most
    half the shortest side, as ``fillet`` rounds the edges of a
    :class:`~simgeometry.Cylinder`. The :meth:`level_set` is the exact signed
    distance, with or without the fillet.

    Example
    -------
    A 2 x 1 x 0.4 block with edges rounded by 0.05::

        block = Cuboid((0, 0, 0), (2.0, 1.0, 0.4), fillet=0.05)
    """

    has_level_set = True
    level_set_exact = True

    def _box_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The lower and upper corners, as host floats on any backend."""
        half_lengths = tuple(float(v) / 2 for v in self.side_lengths)
        midpoint = tuple(float(v) for v in self.midpoint)
        return (
            tuple(map(sub, midpoint, half_lengths)),
            tuple(map(add, midpoint, half_lengths)),
        )

    def __init__(
        self,
        midpoint: BoundsLike,
        side_lengths: BoundsLike,
        bc: ConfigValue = None,
        inverted: bool = False,
        *,
        fillet: float = 0.0,
    ) -> None:
        """
        Initialize a cuboid (rectangular prism) in 3D.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the cuboid.
        side_lengths (list or tuple): Lengths of the sides of the cuboid in each dimension.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the cuboid.
        fillet (float, optional): Radius of the rounded edges and corners.
        """
        assert len(midpoint) == 3, "Length of midpoint must be 3 for 3D Cuboid."
        assert len(side_lengths) == 3, "Length of side_lengths must be 3 for 3D Cuboid."
        if not 0 <= fillet <= 0.5 * min(float(v) for v in side_lengths):
            raise ValueError("fillet must be between 0 and half the shortest side")
        self.fillet = float(fillet)

        self.midpoint = xp.array(midpoint)
        self.side_lengths = xp.array(side_lengths)
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds based on the midpoint and side lengths
        super().__init__(*self._box_bounds())

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the cuboid across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis, 2 for z-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis).")

        # Mirror the midpoint coordinate along the specified axis
        self.midpoint[axis] = 2 * position - self.midpoint[axis]

        self.set_bounding_box(*self._box_bounds())

    def copy(self) -> Cuboid:
        """
        Create a copy of the cuboid.

        Returns:
            Cuboid: A new instance of Cuboid with the same midpoint, side lengths, and boundary conditions.
        """
        return Cuboid(
            midpoint=self.midpoint.copy(),
            side_lengths=self.side_lengths.copy(),
            bc=self._bc,
            inverted=self._inverted,
            fillet=self.fillet,
        )

    def translate(self, offset: tuple[float, float, float]) -> None:
        """
        Translate the cuboid by the given offset.

        Parameters:
            offset (tuple): (dx, dy, dz) translation vector.
        """
        offset_vec = xp.asarray(offset)

        # Apply translation
        self.midpoint = self.midpoint + offset_vec

        self.set_bounding_box(*self._box_bounds())

    def _slice_outline(self, slice_axis: int, slice_position: float) -> np.ndarray:
        """The ``(n, 2)`` outline of a rounded cuboid's slice, a rounded rectangle."""
        from simgeometry.objects.rectangle import Rectangle

        r = self.fillet
        lower = np.asarray(xp.to_numpy(xp.asarray(self.lower_bound)), float)
        upper = np.asarray(xp.to_numpy(xp.asarray(self.upper_bound)), float)
        centre, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
        depth = abs(slice_position - centre[slice_axis]) - (half[slice_axis] - r)
        radius = float(np.sqrt(max(r * r - max(depth, 0.0) ** 2, 0.0)))
        others = [(slice_axis + 1) % 3, (slice_axis + 2) % 3]
        sides = 2 * (half[others] - r + radius)
        section = Rectangle(tuple(centre[others]), tuple(sides), fillet=radius)
        return section.outlines()[0].T

    def get_vertices(self, slice_axis: int, slice_position: float) -> ConfigValue:
        """
        Get the corner vertices of a slice of the cuboid along a given axis.

        Parameters:
            slice_axis (int): Axis along which to slice (0 for x, 1 for y, 2 for z).
            slice_index (float): Position along the slice_axis where the slice is taken.

        Returns:
            vertices (list): List of corner vertices defining the slice polygon;
            with a fillet, the vertices of the rounded rectangle of the slice.
        """
        assert 0 <= slice_axis < 3, "slice_axis must be 0, 1, or 2."
        inverted = self.inverted
        # TODO: replace slice_index with slice_position
        # TODO: Check if the slice_position is in the bounds in that direction

        # Calculate the corners of the slice plane
        # Use the bounds to get the other two axes
        axis1 = (slice_axis + 1) % 3
        axis2 = (slice_axis + 2) % 3

        # List to store the slice corners
        b0 = self.lower_bound
        b1 = self.upper_bound
        if not b0[slice_axis] <= slice_position <= b1[slice_axis]:
            return xp.array([]), inverted
        if self.fillet > 0:
            return xp.array(self._slice_outline(slice_axis, slice_position)), inverted

        vertices = [
            [b0[axis1], b0[axis2]],
            [b1[axis1], b0[axis2]],
            [b1[axis1], b1[axis2]],
            [b0[axis1], b1[axis2]],
        ]
        # TODO: Is this inverted stuff even working?
        return xp.array(vertices), inverted

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the cuboid.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if all coordinates are inside the cuboid, False otherwise.
        """
        if self.fillet > 0:
            point = xp.asarray(coords, dtype=float).reshape(3, 1)
            return bool(xp.to_numpy(self.contains_points(point))[0])
        coords = xp.asarray(coords, dtype=float)
        lower_bound = xp.asarray(self.lower_bound, dtype=float)
        upper_bound = xp.asarray(self.upper_bound, dtype=float)
        inside = bool(xp.all((coords >= lower_bound) & (coords <= upper_bound)))
        return (not inside) if self.inverted else inside

    def _level_set(self, points: Array) -> Array:
        """Exact signed distance to the cuboid, rounded or not."""
        lower = xp.asarray(self.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.upper_bound, dtype=float)[:, None]
        f = self.fillet
        return (
            box_distance(points - 0.5 * (lower + upper), 0.5 * (upper - lower) - f) - f
        )

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        if self.fillet > 0:
            inside = self._level_set(points) <= 0
            return ~inside if self.inverted else inside
        lower = xp.asarray(self.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.upper_bound, dtype=float)[:, None]
        inside = xp.all((points >= lower) & (points <= upper), axis=0)
        return ~inside if self.inverted else inside

    def _inner_sides(self) -> np.ndarray:
        """The side lengths of the box the fillet widens by a ball."""
        sides = np.asarray(xp.to_numpy(xp.asarray(self.side_lengths)), dtype=float)
        return sides - 2 * self.fillet

    def volume(self) -> float:
        """Calculate the volume of the cuboid, less the rounded edges."""
        a, b, c = self._inner_sides()
        r = self.fillet
        faces = 2 * r * (a * b + b * c + c * a)
        return float(
            a * b * c + faces + np.pi * r**2 * (a + b + c) + 4 / 3 * np.pi * r**3
        )

    def surface_area(self) -> float:
        """Calculate the surface area of the cuboid, with its rounded edges."""
        a, b, c = self._inner_sides()
        r = self.fillet
        return float(
            2 * (a * b + b * c + c * a) + 2 * np.pi * r * (a + b + c) + 4 * np.pi * r**2
        )
