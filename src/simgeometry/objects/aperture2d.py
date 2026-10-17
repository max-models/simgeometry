"""Two-dimensional aperture helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp
from numpy import ndarray

from simgeometry._types import ConfigValue
from simgeometry.objects.polygon import Polygon
from simgeometry.pointlist import PointList
from simgeometry.shapes import Shape2D


class Aperture2D(Shape2D):
    """Represent a two-dimensional aperture embedded in a simulation domain."""

    def __init__(
        self,
        axis: int,
        position: float,
        vertices: tuple[tuple[float, float], ...],
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a 2D aperture object.

        Parameters:
            axis (int): Axis along which the aperture is defined (0 or 1). Stored
                only: the plates are always built with the beam along x.
            position (float): The aperture's centre line, ``y = position``.
            vertices (list or tuple): The plate profile as ``(axial, radial)``
                vertices: each plate is the polygon through
                ``(axial, position ± radial)``.
            bc (optional): Boundary condition information.
            inverted (bool, optional): Cover everything except the two plates.
        """

        self.axis = axis
        self._position = position
        self.vertices = vertices
        self._bc = bc
        self._inverted = bool(inverted)

        # Generate two polygons
        self._lower_polygon = Polygon(
            vertices=self.get_vertices_lower(),
            bc=bc,
        )
        self._upper_polygon = Polygon(
            vertices=self.get_vertices_upper(),
            bc=bc,
        )

        # Set the bounds
        super().__init__(
            lower_bound=tuple(
                min(lower, upper)
                for lower, upper in zip(
                    self._lower_polygon.lower_bound,
                    self._upper_polygon.lower_bound,
                )
            ),
            upper_bound=tuple(
                max(lower, upper)
                for lower, upper in zip(
                    self._lower_polygon.upper_bound,
                    self._upper_polygon.upper_bound,
                )
            ),
        )

    @property
    def axis(self) -> int:
        """Return the aperture axis."""
        return self._axis

    @axis.setter
    def axis(self, value: int) -> None:
        """Set the axis."""
        assert value in [0, 1], "Axis must be one of 0 or 1."
        self._axis = value

    def get_vertices(self) -> list[PointList]:
        """Return the vertices that define the object."""
        return [
            self._lower_polygon.get_vertices(),
            self._upper_polygon.get_vertices(),
        ]

    # TODO: simplify these two methods
    # Maybe combine or get rid of them, they are only used in the init
    def get_vertices_lower(self) -> xp.ndarray:
        """
        Return the vertices of the polygon.

        Returns:
        list: List of vertices as (x, y) coordinates.
        """
        axial_positions = self.zvec
        mid_position = self.position
        perp_positions = [mid_position - r for r in self.rvec]

        vertices_polygon = xp.array(
            [(ax, perp) for ax, perp in zip(axial_positions, perp_positions)],
        )
        return vertices_polygon

    def get_vertices_upper(self) -> xp.ndarray:
        """
        Return the vertices of the polygon.

        Returns:
        list: List of vertices as (x, y) coordinates.
        """
        axial_positions = self.zvec
        mid_position = self.position
        perp_positions = [mid_position + r for r in self.rvec]

        vertices_polygon = xp.array(
            [(ax, perp) for ax, perp in zip(axial_positions, perp_positions)],
        )
        return vertices_polygon

    # TODO: The contains method should probably take a tuple as input
    def contains(self, coords: ndarray) -> bool:
        """
        Check if the given coordinates are inside the polygon.

        Parameters:
        coords (list or tuple): Coordinates (x, y).

        Returns:
        bool: True if the point is inside the polygon, False otherwise.
        """

        inside = self._lower_polygon.contains(coords) or self._upper_polygon.contains(
            coords
        )
        return self._apply_inversion(inside)

    def contains_points(self, points: ndarray) -> ndarray:
        """Vectorised :meth:`contains`: a mask of the ``(2, n)`` points inside."""
        inside = self._lower_polygon.contains_points(
            points
        ) | self._upper_polygon.contains_points(points)
        return ~inside if self.inverted else inside

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        """Return object vertices."""
        return self._vertices

    @vertices.setter
    def vertices(self, value: tuple) -> None:
        """Set the vertices."""
        for vertex in value:
            assert len(vertex) == 2, "vertices must be a tuple of length 2."
        self._vertices = value

    @property
    def rvec(self) -> list:
        """Return the radial aperture coordinate vector."""
        return [v[1] for v in self.vertices]

    @property
    def zvec(self) -> list:
        """Return the axial aperture coordinate vector."""
        return [v[0] for v in self.vertices]

    @property
    def position(self) -> float:
        """Return the object position."""
        return self._position


if __name__ == "__main__":
    aperture = Aperture2D(
        axis=0,
        position=10.0,
        vertices=((0, 0), (3, 5), (5, 0)),
    )
    another_pg = Aperture2D(
        vertices=((0, 0), (3, 5), (5, 0)),
        position=10,
        axis=0,
    )
    print("Aperture vertices:", aperture.get_vertices())
