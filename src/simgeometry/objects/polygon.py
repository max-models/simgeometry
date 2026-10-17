"""Polygon helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp
from numpy import ndarray

from simgeometry.kernels.contains_polygon_2d import kernel as contains_polygon_2d
from simgeometry.point import Point2D
from simgeometry.pointlist import PointList
from simgeometry.shapes import Shape2D, _component_major
from simgeometry.utils import vertices_to_bounds


class Polygon(Shape2D):
    """Represent a polygon geometry object."""

    def __init__(
        self,
        vertices: ndarray,
        bc: float | None = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a polygon in 2D space.

        Parameters:
        vertices (list or tuple): List of tuples, each representing a vertex (x, y).
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the polygon.
        """
        assert len(vertices) >= 3, "Polygon must have at least three vertices."
        assert all(len(vertex) == 2 for vertex in vertices), (
            "Each vertex must have two coordinates."
        )

        self.vertices = xp.array(vertices)  # Use setter
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds tuple
        lower_bound, upper_bound = vertices_to_bounds(self.vertices)

        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @property
    def vertices(self) -> ndarray:
        """Return object vertices."""
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: ndarray) -> None:
        """Set the vertices."""
        self._vertices = vertices

    def copy(self) -> Polygon:
        """
        Create a copy of the polygon.

        Returns:
            Polygon: A new instance of Polygon with the same vertices and boundary conditions.
        """
        return Polygon(
            vertices=self.vertices.copy(),
            bc=self._bc,
            inverted=self._inverted,
        )

    def translate(self, offset: tuple[float, float]) -> None:
        """
        Translate the polygon by the given offset.

        Parameters:
            offset (tuple): (dx, dy) translation vector.
        """
        offset_vec = xp.asarray(offset)  # , dtype=self._vertices.dtype)

        # Apply translation
        self.vertices = self.vertices + offset_vec

        lower_bound, upper_bound = vertices_to_bounds(self.vertices)

        self.set_bounding_box(lower_bound, upper_bound)

    # TODO: Enable mirroring along arbitrary axes
    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the polygon across a specified axis at a given position.

        Parameters:
            axis (int): Axis to mirror across (0 for x-axis, 1 for y-axis).
            position (float): Position along the axis to mirror.
        """
        assert axis in (0, 1), "Axis must be 0 (x-axis) or 1 (y-axis)."

        # Create a copy of the vertices to avoid modifying during iteration
        new_vertices = self.vertices.copy()

        # Apply mirroring
        for i in range(len(new_vertices)):
            if axis == 0:  # Mirror across x-axis
                new_vertices[i][1] = 2 * position - new_vertices[i][1]
            else:  # Mirror across y-axis
                new_vertices[i][0] = 2 * position - new_vertices[i][0]

        self.vertices = new_vertices

        lower_bound, upper_bound = vertices_to_bounds(self.vertices)

        self.set_bounding_box(lower_bound, upper_bound)

    def get_vertices(self) -> PointList:
        """
        Return the vertices of the polygon.

        Returns:
        list: List of vertices as (x, y) coordinates.
        """

        point_list = PointList(
            [Point2D(x=vertex[0], y=vertex[1]) for vertex in self._vertices],
        )

        return point_list

    def _contains_polygon(self, coords: ndarray) -> bool:
        """
        Check if the given coordinates are inside the polygon.

        Parameters:
        coords (list or tuple): Coordinates (x, y).

        Returns:
        bool: True if the point is inside the polygon, False otherwise.
        """
        coords = xp.to_numpy(coords)
        vertices = xp.to_numpy(self._vertices)
        x, y = coords

        # First check if the point is inside the bounding box
        if not self.bounding_box.contains(coords):
            return False

        # Ray casting algorithm for point-in-polygon
        num_vertices = len(vertices)
        inside = False
        j = num_vertices - 1
        for i in range(num_vertices):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-16) + xi
            ):
                inside = not inside
            j = i
        return inside
        # return [self._vertices]

    def contains(self, coords: ndarray) -> bool:
        """
        Check if the given coordinates are inside the (possibly inverted) polygon.

        Parameters:
        coords (list or tuple): Coordinates (x, y).

        Returns:
        bool: True if the point is inside the object, False otherwise.
        """
        return self._apply_inversion(self._contains_polygon(coords))

    def contains_points(self, points: ndarray) -> ndarray:
        """Vectorised :meth:`contains`: a mask of the ``(2, n)`` points inside."""
        points = xp.asarray(_component_major(points, 2), dtype=float)
        vertices = xp.kernels.as_kernel_array(self._vertices, like=points, dtype=float)
        lower = xp.asarray(self.bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.bounding_box.upper_bound, dtype=float)[:, None]
        inside = xp.empty(points.shape[1], dtype=bool)
        contains_polygon_2d(
            xp.kernels.as_kernel_array(points, like=points, dtype=float),
            inside,
            vertices,
            0,
        )
        inside &= (points[0] >= lower[0]) & (points[0] < upper[0])
        inside &= (points[1] >= lower[1]) & (points[1] < upper[1])
        return ~inside if self.inverted else inside
