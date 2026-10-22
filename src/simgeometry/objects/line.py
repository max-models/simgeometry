"""Line helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.shapes import Shape1D, _component_major


class Line(Shape1D):
    """Represent a line geometry object.

    Its :meth:`level_set` is the exact signed distance to the interval.
    """

    has_level_set = True
    level_set_exact = True

    def __init__(
        self,
        midpoint: BoundsLike,
        side_lengths: BoundsLike,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a line in 1D space.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the rectangle.
        side_lengths (list or tuple): Lengths of the sides of the rectangle in each dimension.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the line.
        """
        assert len(midpoint) == 1, (
            f"Length of midpoint must match ndim (2). {len(midpoint) =}"
        )
        assert len(side_lengths) == 1, "Length of side_lengths must match ndim."

        self.midpoint = xp.array(midpoint)
        self.side_lengths = xp.array(side_lengths)
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds based on the midpoint and side lengths
        half_lengths = self.side_lengths / 2
        lower_bound = tuple(self.midpoint - half_lengths)
        upper_bound = tuple(self.midpoint + half_lengths)

        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the line across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis).
            position (float): Position along the axis to mirror.
        """
        if axis != 0:
            raise ValueError("Axis must be 0 (x-axis).")

        # Mirror the midpoint coordinate along the specified axis
        self.midpoint[axis] = 2 * position - self.midpoint[axis]

        half_lengths = self.side_lengths / 2
        lower_bound = tuple(self.midpoint - half_lengths)
        upper_bound = tuple(self.midpoint + half_lengths)

        self.set_bounding_box(lower_bound, upper_bound)

    def copy(self) -> Line:
        """
        Create a copy of the line.

        Returns:
            Line: A new instance of Line with the same midpoint, side lengths, and boundary conditions.
        """
        return Line(
            midpoint=self.midpoint.copy(),
            side_lengths=self.side_lengths.copy(),
            bc=self._bc,
            inverted=self._inverted,
        )

    def translate(self, offset: tuple[float]) -> None:
        """
        Translate the line by the given offset.

        Parameters:
            offset (tuple): (dx,) translation vector.
        """
        offset_vec = xp.asarray(offset)

        # Apply translation
        self.midpoint = self.midpoint + offset_vec

        half_lengths = self.side_lengths / 2
        lower_bound = tuple(self.midpoint - half_lengths)
        upper_bound = tuple(self.midpoint + half_lengths)

        self.set_bounding_box(lower_bound, upper_bound)

    def get_vertices(self) -> ConfigValue:
        """Return the vertices that define the object."""
        return [(self.lower_bound[0], self.upper_bound[0])]

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the line.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the line, False otherwise.
        """
        assert len(coords) == 1, f"Expected 1 dimension, got {len(coords)}."

        coords = xp.to_numpy(coords)
        return self._apply_inversion(
            bool(self.lower_bound[0] <= coords[0] <= self.upper_bound[0])
        )

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(1, n)`` points inside."""
        points = _component_major(points, 1)
        inside = (self.lower_bound[0] <= points[0]) & (points[0] <= self.upper_bound[0])
        return ~inside if self.inverted else inside

    def _level_set(self, points: Array) -> Array:
        """Exact signed distance to the interval."""
        lower, upper = float(self.lower_bound[0]), float(self.upper_bound[0])
        return xp.abs(points[0] - 0.5 * (lower + upper)) - 0.5 * (upper - lower)
