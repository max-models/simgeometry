"""Circle helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp
import numpy as np

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.shapes import Shape2D, Spacing, _component_major, _distribute


class Circle(Shape2D):
    """Represent a circular geometry object."""

    def __init__(
        self,
        midpoint: BoundsLike,
        radius: float,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a circle in 2D space.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the circle.
        radius (float): Radius of the circle.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the circle.
        """
        assert len(midpoint) == 2, "Midpoint must have two coordinates."
        assert isinstance(radius, (int, float)) and radius > 0, (
            "Radius must be a positive number."
        )

        self.midpoint = list(midpoint)
        self.radius = radius
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds based on the midpoint and radius
        lower_bound = tuple(x - self.radius for x in self.midpoint)
        upper_bound = tuple(x + self.radius for x in self.midpoint)

        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the circle across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 1:
            raise ValueError("Axis must be 0 (x-axis) or 1 (y-axis).")

        # Mirror the midpoint coordinate along the specified axis
        self.midpoint[axis] = 2 * position - self.midpoint[axis]

        lower_bound = tuple(x - self.radius for x in self.midpoint)
        upper_bound = tuple(x + self.radius for x in self.midpoint)

        self.set_bounding_box(lower_bound, upper_bound)

    def copy(self) -> Circle:
        """
        Create a copy of the circle.

        Returns:
            Circle: A new instance of Circle with the same midpoint, radius, and boundary conditions.
        """
        return Circle(
            midpoint=self.midpoint.copy(),
            radius=self.radius,
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

        lower_bound = tuple(x - self.radius for x in self.midpoint)
        upper_bound = tuple(x + self.radius for x in self.midpoint)

        self.set_bounding_box(lower_bound, upper_bound)

    def _outline_loops(self, max_spacing: Spacing | None) -> list[np.ndarray]:
        """One loop of vertices on the circle, at most ``max_spacing`` apart (100 by default)."""
        x0, y0 = (float(c) for c in self.midpoint)

        def curve(t: np.ndarray) -> np.ndarray:
            theta = 2 * np.pi * t
            return np.stack(
                (x0 + self.radius * np.cos(theta), y0 + self.radius * np.sin(theta))
            )

        if max_spacing is None:
            return [curve(np.arange(100) / 100)]
        perimeter = 2 * np.pi * self.radius
        return [curve(_distribute(curve, perimeter, max_spacing, min_points=3))]

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the circle.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the circle, False otherwise.
        """
        distance_squared = xp.sum(
            (xp.asarray(coords, dtype=float) - xp.asarray(self.midpoint, dtype=float))
            ** 2,
        )
        return self._apply_inversion(bool(distance_squared <= self.radius**2))

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(2, n)`` points inside."""
        points = _component_major(points, 2)
        midpoint = xp.asarray(self.midpoint, dtype=float)[:, None]
        # summed over the axes in order, as xp.sum does for one point
        distance_squared = xp.sum((points - midpoint) ** 2, axis=0)
        inside = distance_squared <= self.radius**2
        return ~inside if self.inverted else inside
