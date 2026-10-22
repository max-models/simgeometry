"""Sphere helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp

from simgeometry._types import Array, ConfigValue
from simgeometry.shapes import Shape3D, _component_major


class Sphere(Shape3D):
    """Represent a spherical geometry object.

    Its :meth:`level_set` is the exact signed distance ``|p - midpoint| - radius``.
    """

    has_level_set = True
    level_set_exact = True

    def __init__(
        self,
        midpoint: list | tuple,
        radius: float,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a sphere in 3D.

        Parameters:
        midpoint (list or tuple): Coordinates of the midpoint of the sphere.
        radius (float): Radius of the sphere.
        bc (optional): Boundary condition information.
        """
        assert len(midpoint) == 3, "Length of midpoint must be 3 for 3D Sphere."
        assert isinstance(radius, (int, float)) and radius > 0, (
            "Radius must be a positive number."
        )

        self.midpoint = list(midpoint)
        self.radius = radius
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds based on the midpoint and radius
        lower_bound = tuple(m - self.radius for m in self.midpoint)
        upper_bound = tuple(m + self.radius for m in self.midpoint)

        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the sphere across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis, 2 for z-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis).")

        # Mirror the midpoint coordinate along the specified axis
        self.midpoint[axis] = 2 * position - self.midpoint[axis]

        lower_bound = tuple(m - self.radius for m in self.midpoint)
        upper_bound = tuple(m + self.radius for m in self.midpoint)

        self.set_bounding_box(lower_bound, upper_bound)

    def copy(self) -> Sphere:
        """
        Create a copy of the sphere.

        Returns:
            Sphere: A new instance of Sphere with the same midpoint, radius, and boundary conditions.
        """
        return Sphere(
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

        lower_bound = tuple(m - self.radius for m in self.midpoint)
        upper_bound = tuple(m + self.radius for m in self.midpoint)

        self.set_bounding_box(lower_bound, upper_bound)

    def get_vertices(self, slice_axis: int, slice_position: float) -> ConfigValue:
        """
        Get the vertices of a slice of the sphere along a given axis.

        Parameters:
        slice_axis (int): Axis along which to slice (0 for x, 1 for y, 2 for z).
        slice_position (float): Position along the slice_axis where the slice is taken.

        Returns:
        vertices (ndarray): Array of vertices defining the slice circle.
                            If the slice does not intersect the sphere, returns an empty array.
        """
        assert 0 <= slice_axis < 3, "slice_axis must be 0, 1, or 2."

        inverted = self.inverted

        # Calculate the distance from the sphere center to the slicing plane
        distance = abs(slice_position - self.midpoint[slice_axis])

        # Check if the plane intersects the sphere
        if distance > self.radius:
            # The plane does not intersect the sphere
            return xp.array([]), inverted

        elif distance == self.radius:
            # The plane touches the sphere at a single point
            axis1 = (slice_axis + 1) % 3
            axis2 = (slice_axis + 2) % 3
            point = xp.array([self.midpoint[axis1], self.midpoint[axis2]])
            return xp.array([point]), inverted

        else:
            # The plane intersects the sphere in a circle
            # Compute the radius of the circle
            r = xp.sqrt(self.radius**2 - distance**2)

            # Get the center of the circle in the plane
            axis1 = (slice_axis + 1) % 3
            axis2 = (slice_axis + 2) % 3
            center1 = self.midpoint[axis1]
            center2 = self.midpoint[axis2]

            # Generate points along the circle
            N = 100  # Number of points to approximate the circle
            theta = xp.linspace(0, 2 * xp.pi, N)
            x = center1 + r * xp.cos(theta)
            y = center2 + r * xp.sin(theta)

            # Combine x and y into an array of vertices
            vertices = xp.column_stack((x, y))

            return xp.array(vertices), inverted

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the sphere.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the sphere, False otherwise.
        """
        coords = xp.asarray(coords, dtype=float)
        midpoint = xp.asarray(self.midpoint, dtype=float)
        distance_squared = xp.sum((coords - midpoint) ** 2)
        inside = bool(distance_squared <= self.radius**2)
        return (not inside) if self.inverted else inside

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        midpoint = xp.asarray(self.midpoint, dtype=float)[:, None]
        distance_squared = xp.sum((points - midpoint) ** 2, axis=0)
        inside = distance_squared <= self.radius**2
        return ~inside if self.inverted else inside

    def _level_set(self, points: Array) -> Array:
        """Exact signed distance to the sphere."""
        midpoint = xp.asarray(self.midpoint, dtype=float)[:, None]
        return xp.sqrt(xp.sum((points - midpoint) ** 2, axis=0)) - self.radius

    def volume(self) -> float:
        """Calculate the volume of the sphere."""
        return (4 / 3) * xp.pi * self.radius**3

    def surface_area(self) -> float:
        """Calculate the surface area of the sphere."""
        return 4 * xp.pi * self.radius**2
