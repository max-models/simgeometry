"""Rectangle helpers for geometric object definitions."""

from __future__ import annotations

from operator import add, sub

import cunumpy as xp

from simgeometry._types import Array, ConfigValue
from simgeometry.point import Point2D
from simgeometry.pointlist import PointList
from simgeometry.shapes import Shape2D, _component_major


class Rectangle(Shape2D):
    """Represent a rectangular geometry object."""

    def __init__(
        self,
        midpoint: tuple | list,
        side_lengths: tuple,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a rectangle in 2D space.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the rectangle.
        side_lengths (list or tuple): Lengths of the sides of the rectangle in each dimension.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the rectangle.
        """
        assert len(midpoint) == 2, (
            f"Length of midpoint must match ndim (2). {len(midpoint) =}"
        )
        assert len(side_lengths) == 2, "Length of side_lengths must match ndim."

        self.midpoint = list(midpoint)
        self.side_lengths = side_lengths
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
        Mirror the rectangle across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 1:
            raise ValueError("Axis must be 0 (x-axis) or 1 (y-axis).")

        # Mirror the midpoint coordinate along the specified axis
        self.midpoint[axis] = 2 * position - self.midpoint[axis]

        half_lengths = tuple(len / 2 for len in self.side_lengths)
        lower_bound = tuple(map(sub, self.midpoint, half_lengths))
        upper_bound = tuple(map(add, self.midpoint, half_lengths))

        self.set_bounding_box(lower_bound, upper_bound)

    def copy(self) -> Rectangle:
        """
        Create a copy of the rectangle.

        Returns:
            Rectangle: A new instance of Rectangle with the same midpoint, side lengths, and boundary conditions.
        """
        return Rectangle(
            midpoint=self.midpoint.copy(),
            side_lengths=self.side_lengths,
            bc=self._bc,
            inverted=self._inverted,
        )

    def translate(self, offset: tuple[float, float]) -> None:
        """
        Translate the xyz by the given offset.

        Parameters:
            offset (tuple): (dx, dy) translation vector.
        """
        offset_vec = xp.asarray(offset)

        # Apply translation
        self.midpoint = list(xp.asarray(self.midpoint) + offset_vec)

        half_lengths = tuple(len / 2 for len in self.side_lengths)
        lower_bound = tuple(map(sub, self.midpoint, half_lengths))
        upper_bound = tuple(map(add, self.midpoint, half_lengths))

        self.set_bounding_box(lower_bound, upper_bound)

    def get_vertices(self) -> PointList:
        """Return the vertices that define the object."""
        xmin, ymin = self.lower_bound
        xmax, ymax = self.upper_bound

        point_list = PointList(
            [
                Point2D(x=xmin, y=ymin),
                Point2D(x=xmax, y=ymin),
                Point2D(x=xmax, y=ymax),
                Point2D(x=xmin, y=ymax),
            ],
        )
        return point_list

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the rectangle.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the rectangle, False otherwise.
        """
        return self._apply_inversion(self.bounding_box.contains(coords))

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(2, n)`` points inside."""
        points = _component_major(points, 2)
        lower = xp.asarray(self.bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.bounding_box.upper_bound, dtype=float)[:, None]
        inside = xp.all((points >= lower) & (points < upper), axis=0)
        return ~inside if self.inverted else inside
