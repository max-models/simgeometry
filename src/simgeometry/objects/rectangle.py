"""Rectangles, optionally with rounded corners."""

from __future__ import annotations

from operator import add, sub

import cunumpy as xp
import numpy as np

from simgeometry._level_set import box_distance
from simgeometry._types import Array, ConfigValue
from simgeometry.shapes import Shape2D, Spacing, _component_major, _distribute

#: Vertices per rounded corner of an outline without ``max_spacing``.
CORNER_POINTS = 25


class Rectangle(Shape2D):
    """Represent a rectangular geometry object.

    ``fillet`` rounds the four corners with that radius, at most half the shorter
    side; at half the shorter side the rectangle is a stadium (a circle for a
    square). Rounded corners keep the surface field of an electrode bounded. The
    :meth:`level_set` is the exact signed distance, with or without the fillet.

    Example
    -------
    A plate 2 wide and 0.4 thick with rounded corners, and a stadium::

        plate = Rectangle((0, 0), (2.0, 0.4), fillet=0.05)
        stadium = Rectangle((0, 0), (2.0, 0.4), fillet=0.2)
    """

    exact_outline = True
    has_level_set = True
    level_set_exact = True

    def __init__(
        self,
        midpoint: tuple | list,
        side_lengths: tuple,
        bc: ConfigValue = None,
        inverted: bool = False,
        *,
        fillet: float = 0.0,
    ) -> None:
        """
        Initialize a rectangle in 2D space.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the rectangle.
        side_lengths (list or tuple): Lengths of the sides of the rectangle in each dimension.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the rectangle.
        fillet (float, optional): Radius of the rounded corners.
        """
        assert len(midpoint) == 2, (
            f"Length of midpoint must match ndim (2). {len(midpoint) =}"
        )
        assert len(side_lengths) == 2, "Length of side_lengths must match ndim."
        if not 0 <= fillet <= 0.5 * min(float(v) for v in side_lengths):
            raise ValueError("fillet must be between 0 and half the shorter side")

        self.fillet = float(fillet)
        # a rounded rectangle's outline is sampled on its arcs
        self.exact_outline = self.fillet == 0
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
            fillet=self.fillet,
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

    def _corner_loops(self) -> list[np.ndarray]:
        """The four corners as one counter-clockwise ``(2, 4)`` loop."""
        (xmin, ymin), (xmax, ymax) = self.lower_bound, self.upper_bound
        return [np.array([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]], float)]

    def _outline_loops(self, max_spacing: Spacing | None) -> list[np.ndarray]:
        """The straight edges and the quarter circles of the corners, counter-clockwise.

        Without ``max_spacing`` each edge is one segment and each corner has
        :data:`CORNER_POINTS` vertices on its arc.
        """
        if self.fillet == 0:
            return super()._outline_loops(max_spacing)
        r = self.fillet
        (xmin, ymin), (xmax, ymax) = self.lower_bound, self.upper_bound
        corners = [(xmax - r, ymin + r), (xmax - r, ymax - r)]
        corners += [(xmin + r, ymax - r), (xmin + r, ymin + r)]
        pieces = []
        for k, (cx, cy) in enumerate(corners):
            # the edge into this corner, from the arc of the previous one
            px, py = corners[k - 1]
            angle = (k - 1) * 0.5 * np.pi
            a = np.array([[px + r * np.cos(angle)], [py + r * np.sin(angle)]])
            b = np.array([[cx + r * np.cos(angle)], [cy + r * np.sin(angle)]])

            def edge(t, a=a, b=b):
                return a + t * (b - a)

            pieces.append((edge, float(np.hypot(*(b - a)[:, 0])), 1))

            def arc(t, cx=cx, cy=cy, angle=angle):
                theta = angle + 0.5 * np.pi * t
                return np.stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))

            pieces.append((arc, 0.5 * np.pi * r, CORNER_POINTS))
        points = []
        for curve, length, default in pieces:
            if length <= 1e-12 * max(xmax - xmin, ymax - ymin):
                continue  # the flat sides of a stadium or a circle
            if max_spacing is None:
                t = np.arange(default) / default
            else:
                t = _distribute(curve, length, max_spacing)
            points.append(curve(t))
        return [np.concatenate(points, axis=1)]

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the rectangle.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the rectangle, False otherwise.
        """
        if self.fillet > 0:
            point = xp.asarray(coords, dtype=float).reshape(2, 1)
            return bool(xp.to_numpy(self.contains_points(point))[0])
        return self._apply_inversion(self.bounding_box.contains(coords))

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(2, n)`` points inside."""
        points = _component_major(points, 2)
        if self.fillet > 0:
            inside = self._level_set(points) <= 0
            return ~inside if self.inverted else inside
        lower = xp.asarray(self.bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.bounding_box.upper_bound, dtype=float)[:, None]
        inside = xp.all((points >= lower) & (points < upper), axis=0)
        return ~inside if self.inverted else inside

    def _level_set(self, points: Array) -> Array:
        """Exact signed distance to the rectangle, rounded or not."""
        lower = xp.asarray(self.bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.bounding_box.upper_bound, dtype=float)[:, None]
        f = self.fillet
        return (
            box_distance(points - 0.5 * (lower + upper), 0.5 * (upper - lower) - f) - f
        )
