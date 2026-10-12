"""Pointlist helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

from simgeometry.point import Point


class PointList:
    """A class representing a list of points in ND space."""

    def __init__(self, points: list, inverted: bool = False) -> None:
        """
        Initialize a list of points in ND space.

        Parameters:
        points (list): List of Point objects.
        """
        self.points = points
        self.inverted = inverted

    def add_point(self, point: Point) -> None:
        """
        Add a point to the list.

        Parameters:
        point (Point): A Point object to add to the list.
        """
        self.points.append(point)

    def get_coordinates(self) -> list[tuple]:
        """
        Get the coordinates of all points in the list.

        Returns:
        list: List of coordinates for each point.
        """
        return [point.coordinates for point in self.points]
