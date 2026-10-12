"""Cuboid helpers for geometric object definitions."""

from __future__ import annotations

from operator import add, sub

import cunumpy as xp

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.shapes import Shape3D, _component_major


class Cuboid(Shape3D):
    """Represent a cuboid geometry object."""

    def __init__(
        self,
        midpoint: BoundsLike,
        side_lengths: BoundsLike,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a cuboid (rectangular prism) in 3D.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the cuboid.
        side_lengths (list or tuple): Lengths of the sides of the cuboid in each dimension.
        bc (optional): Boundary condition information.
        """
        assert len(midpoint) == 3, "Length of midpoint must be 3 for 3D Cuboid."
        assert len(side_lengths) == 3, "Length of side_lengths must be 3 for 3D Cuboid."

        self.midpoint = xp.array(midpoint)
        self.side_lengths = xp.array(side_lengths)
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds based on the midpoint and side lengths
        half_lengths = tuple(len / 2 for len in self.side_lengths)
        lower_bound = tuple(map(sub, self.midpoint, half_lengths))
        upper_bound = tuple(map(add, self.midpoint, half_lengths))

        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

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

        half_lengths = tuple(len / 2 for len in self.side_lengths)
        lower_bound = tuple(map(sub, self.midpoint, half_lengths))
        upper_bound = tuple(map(add, self.midpoint, half_lengths))

        self.set_bounding_box(lower_bound, upper_bound)

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

        # Calculate bounds based on the midpoint and side lengths
        half_lengths = tuple(len / 2 for len in self.side_lengths)
        lower_bound = tuple(map(sub, self.midpoint, half_lengths))
        upper_bound = tuple(map(add, self.midpoint, half_lengths))

        self.set_bounding_box(lower_bound, upper_bound)

    def get_vertices(self, slice_axis: int, slice_position: float) -> ConfigValue:
        """
        Get the corner vertices of a slice of the cuboid along a given axis.

        Parameters:
            slice_axis (int): Axis along which to slice (0 for x, 1 for y, 2 for z).
            slice_index (float): Position along the slice_axis where the slice is taken.

        Returns:
            vertices (list): List of corner vertices defining the slice polygon.
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
        coords = xp.asarray(coords, dtype=float)
        lower_bound = xp.asarray(self.lower_bound, dtype=float)
        upper_bound = xp.asarray(self.upper_bound, dtype=float)
        inside = bool(xp.all((coords >= lower_bound) & (coords <= upper_bound)))
        return (not inside) if self.inverted else inside

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        lower = xp.asarray(self.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.upper_bound, dtype=float)[:, None]
        inside = xp.all((points >= lower) & (points <= upper), axis=0)
        return ~inside if self.inverted else inside

    def volume(self) -> float:
        """Calculate the volume of the cuboid."""
        return float(xp.prod(self.side_lengths))

    def surface_area(self) -> float:
        """Calculate the surface area of the cuboid."""
        areas = [
            self.side_lengths[i] * self.side_lengths[j]
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        return float(2 * sum(areas))
