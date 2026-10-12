"""Point helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

from simgeometry._types import Array


class Point:
    """An abstract class representing a point in ND space."""

    def __init__(self, coordinates: tuple) -> None:
        """
        Initialize a point in ND space.

        Parameters:
        coordinates (list or tuple): Coordinates of the point in each dimension.
        """
        self.coordinates = coordinates


class Point1D(Point):
    """A class representing a point in 1D space."""

    def __init__(self, x: Array) -> None:
        """
        Initialize a point in 1D space.

        Parameters:
        x (float): Coordinate of the point.
        """
        super().__init__((x,))


class Point2D(Point):
    """A class representing a point in 2D space."""

    def __init__(self, x: Array, y: Array) -> None:
        """
        Initialize a point in 2D space.

        Parameters:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        """
        super().__init__((x, y))


class Point3D(Point):
    """A class representing a point in 3D space."""

    def __init__(self, x: Array, y: Array, z: Array) -> None:
        """
        Initialize a point in 3D space.

        Parameters:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        z (float): z-coordinate of the point.
        """
        super().__init__((x, y, z))
