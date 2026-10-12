"""Shapes helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import cunumpy as xp
import numpy as np

from simgeometry._types import Array
from simgeometry.boundingbox import BoundingBox
from simgeometry.grid import Grid


def _component_major(points: Array, ndim: int) -> Array:
    """Return ``points`` as a ``(ndim, n)`` float array, checking its layout."""
    points = xp.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] != ndim:
        raise ValueError(f"Expected points with shape ({ndim}, n), got {points.shape}")
    return points


class Shape(ABC):
    """Define the base interface for geometric shapes.

    A shape created with ``inverted=True`` covers everything *except* its
    geometry (e.g. a chamber wall around a circular cavity): ``contains`` is
    negated and the shape extends over the whole simulation domain.  The
    bounding box still describes the geometry itself, so consumers must not
    use it to skip points for inverted shapes.
    """

    _inverted: bool = False

    def __init__(
        self,
        ndim: int,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
    ) -> None:
        """
        Initialize a shape.

        Parameters:
        ndim (int): Number of dimensions.
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
        """
        assert isinstance(ndim, int) and ndim > 0, "ndim must be a positive integer."
        assert ndim in [1, 2, 3]
        self.ndim = ndim
        self.set_bounding_box(lower_bound, upper_bound)

    def set_bounding_box(
        self,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
    ) -> None:
        """Set the bounding box for the shape."""
        self._bounding_box = BoundingBox(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        assert self.ndim == self.bounding_box.ndim

    @abstractmethod
    def contains(self, coords: tuple[float, ...]) -> bool:
        """
        Check if the given coordinates are inside the shape.

        Parameters:
        coords (list or tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the shape, False otherwise.
        """

    def contains_points(self, points: Array) -> Array:
        """Return a mask of the ``(ndim, n)`` points inside the shape.

        The default calls :meth:`contains` point by point; shapes override it with a
        vectorised test.
        """
        points = _component_major(points, self.ndim)
        return xp.asarray([bool(self.contains(p)) for p in points.T], dtype=bool)

    def node_mask(self, grid: Grid) -> np.ndarray:
        """Return a host boolean array of the grid's shape, set at the nodes inside.

        One vectorised :meth:`contains_points` over :meth:`Grid.node_points`, so
        the cost is that of a few array passes over the grid.
        """
        inside = self.contains_points(grid.node_points())
        shape = tuple(int(n) for n in grid.num_gridpoints)
        return np.asarray(xp.to_numpy(inside), dtype=bool).reshape(shape)

    def get_indices_in_object(self, grid: Grid) -> list[int] | list[tuple[int, ...]]:
        """
        Get the indices of grid points that are inside the shape.

        Parameters:
        grid (object): Grid object containing indices and cell sizes.

        Returns:
        list: The :attr:`Grid.indices` of the nodes inside, in their order.
        """
        if hasattr(grid, "indices_from_mask"):
            return grid.indices_from_mask(self.node_mask(grid))
        # any object with ``indices`` and ``node_coordinates`` (e.g. a few nodes)
        indices = list(grid.indices)
        if not indices:
            return []
        points = np.stack(
            [np.atleast_1d(grid.node_coordinates(index)) for index in indices], axis=1
        )
        inside = np.asarray(xp.to_numpy(self.contains_points(points)), dtype=bool)
        return [index for index, keep in zip(indices, inside) if keep]

    def get_bound_vertices(self, slice_axis: int) -> xp.ndarray:
        """Return vertices that bound the shape."""
        b0 = self.lower_bound
        b1 = self.upper_bound
        # print(b0, b1)
        axis1 = (slice_axis + 1) % 3
        axis2 = (slice_axis + 2) % 3
        bound_vertices = xp.array(
            [
                [b0[axis1], b0[axis2]],
                [b1[axis1], b0[axis2]],
                [b1[axis1], b1[axis2]],
                [b0[axis1], b1[axis2]],
            ],
        )
        return bound_vertices

    @property
    def inverted(self) -> bool:
        """Return whether this object represents an inverted region."""
        return self._inverted

    def _apply_inversion(self, inside: bool) -> bool:
        """Return ``inside`` for the geometry, negated for an inverted shape."""
        return (not inside) if self.inverted else bool(inside)

    @property
    def bounding_box(self) -> BoundingBox:
        """Return the bounding box."""
        return self._bounding_box

    @property
    def lower_bound(self) -> tuple[float, ...]:
        """Return the lower coordinate bounds."""
        return self.bounding_box.lower_bound

    @property
    def upper_bound(self) -> tuple[float, ...]:
        """Return the upper coordinate bounds."""
        return self.bounding_box.upper_bound


class Shape1D(Shape):
    """Represent a one-dimensional shape."""

    def __init__(
        self,
        lower_bound: tuple[float, float],
        upper_bound: tuple[float, float],
    ) -> None:
        """
        Initialize a 1D shape.

        Parameters:
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
        """
        super().__init__(ndim=1, lower_bound=lower_bound, upper_bound=upper_bound)

    @abstractmethod
    def get_vertices(self) -> list[list[float]]:
        """
        Get the vertices of the 1D shape.

        Returns:
        list: List of vertices as [x] coordinates.
        """

    @abstractmethod
    def contains(self, coords: float) -> bool:
        """
        Check if the given coordinates are inside the shape.

        Parameters:
        coords (float): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the shape, False otherwise.
        """


class Shape2D(Shape):
    """Represent a two-dimensional shape."""

    def __init__(
        self,
        lower_bound: tuple[float, float],
        upper_bound: tuple[float, float],
    ) -> None:
        """
        Initialize a 2D shape.

        Parameters:
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
        """
        super().__init__(ndim=2, lower_bound=lower_bound, upper_bound=upper_bound)

    @abstractmethod
    def get_vertices(self) -> list[list[float]]:
        """
        Get the vertices of the 2D shape.

        Returns:
        list: List of vertices as [x, y] coordinates.
        """

    @abstractmethod
    def contains(self, coords: tuple[float, float]) -> bool:
        """
        Check if the given coordinates are inside the shape.

        Parameters:
        coords (tuple): Coordinates for each dimension.

        Returns:
        bool: True if the coordinates are inside the shape, False otherwise.
        """

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Draw the shape as filled polygons into ``ax`` and return the axes.

        Keyword arguments style the polygons (``facecolor``, ``alpha``, ...).
        """
        from simgeometry.plotting import draw_shape

        return draw_shape(self, ax=ax, **kwargs)


class Shape3D(Shape):
    """Represent a three-dimensional shape."""

    def __init__(
        self,
        lower_bound: tuple[float, float, float],
        upper_bound: tuple[float, float, float],
    ) -> None:
        """
        Initialize a 3D shape.

        Parameters:
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
        """

        super().__init__(ndim=3, lower_bound=lower_bound, upper_bound=upper_bound)

    @abstractmethod
    def volume(self) -> float:
        """
        Calculate the volume of the 3D shape.

        Returns:
        float: Volume of the shape.
        """

    @abstractmethod
    def surface_area(self) -> float:
        """
        Calculate the surface area of the 3D shape.

        Returns:
        float: Surface area of the shape.
        """
