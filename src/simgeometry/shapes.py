"""Shapes helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import cunumpy as xp
import numpy as np

from simgeometry._types import Array
from simgeometry.api import GeometryAPI
from simgeometry.boundingbox import BoundingBox
from simgeometry.grid import Grid


def _component_major(points: Array, ndim: int) -> Array:
    """Return ``points`` as a ``(ndim, n)`` float array, checking its layout."""
    points = xp.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] != ndim:
        raise ValueError(f"Expected points with shape ({ndim}, n), got {points.shape}")
    return points


def _signed_area(loop: np.ndarray) -> float:
    """Shoelace area of a ``(2, n)`` loop: positive when counter-clockwise."""
    x, y = loop
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


#: A spacing: a length, or a function of ``(2, m)`` points returning ``(m,)`` lengths.
Spacing = float | Callable[[np.ndarray], np.ndarray]


def _pieces(length: float, max_spacing: float) -> int:
    """Number of equal pieces that keeps each one at most ``max_spacing`` long.

    The tolerance keeps ``length / (length / n)`` from rounding up to ``n + 1``.
    """
    return max(1, int(np.ceil(length / max_spacing * (1 - 1e-12))))


def _distribute(
    curve: Callable[[np.ndarray], np.ndarray],
    length: float,
    max_spacing: Spacing,
    *,
    min_points: int = 1,
    samples: int = 256,
) -> np.ndarray:
    """Parameters in ``[0, 1)`` of points along ``curve`` at about ``max_spacing`` apart.

    ``curve`` maps parameters ``(m,)`` in ``[0, 1]`` to ``(2, m)`` points at uniform
    arc length, ``length`` long in total. A constant spacing gives equal pieces of at most
    that length; a callable one places points at equal steps of the integral of
    ``1 / spacing`` along the curve, sampled at ``samples`` intervals.
    """
    if not callable(max_spacing):
        n = max(min_points, _pieces(length, max_spacing))
        return np.arange(n) / n
    t = np.linspace(0.0, 1.0, samples + 1)
    spacing = np.broadcast_to(np.asarray(max_spacing(curve(t)), dtype=float), t.shape)
    if not np.all(spacing > 0):
        raise ValueError("max_spacing must be positive")
    density = length / spacing
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(t)))
    )
    n = max(min_points, _pieces(cumulative[-1], 1.0))
    return np.interp(np.arange(n) / n * cumulative[-1], cumulative, t)


def _refine_loop(loop: np.ndarray, max_spacing: Spacing) -> np.ndarray:
    """Split each edge of a closed ``(2, n)`` loop into pieces of about ``max_spacing``."""
    pieces = []
    for a, b in zip(loop.T, np.roll(loop, -1, axis=1).T):
        length = float(np.hypot(*(b - a)))
        a, b = a[:, None], b[:, None]
        t = _distribute(lambda t: a + t * (b - a), length, max_spacing)
        pieces.append(a + t * (b - a))
    return np.concatenate(pieces, axis=1)


class Shape(GeometryAPI, ABC):
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

    def outlines(self, max_spacing: Spacing | None = None) -> list[np.ndarray]:
        """Return the closed boundary loops of the shape as ``(2, n)`` host arrays.

        Each loop is counter-clockwise, without repeating its first vertex. A shape
        made of several pieces (such as :class:`~simgeometry.Aperture2D`'s two
        plates) has one loop per piece. With ``max_spacing``, edges longer than it
        are split evenly so that no two consecutive vertices are further apart;
        curved shapes then place every vertex on the curve. ``max_spacing`` may
        also be a function of ``(2, m)`` points returning ``(m,)`` spacings, to grade
        the vertices (finer near a corner, say); the spacing then holds
        approximately. Inversion does not change the outline.
        """
        if (
            max_spacing is not None
            and not callable(max_spacing)
            and not max_spacing > 0
        ):
            raise ValueError("max_spacing must be positive")
        return [
            loop[:, ::-1] if _signed_area(loop) < 0.0 else loop
            for loop in self._outline_loops(max_spacing)
        ]

    def _outline_loops(self, max_spacing: Spacing | None) -> list[np.ndarray]:
        """The loops through :meth:`get_vertices`, refined; curved shapes override this."""
        pointlists = self.get_vertices()
        if not isinstance(pointlists, list):
            pointlists = [pointlists]
        loops = [
            np.array(
                [
                    [float(v[0]) for v in pl.get_coordinates()],
                    [float(v[1]) for v in pl.get_coordinates()],
                ]
            )
            for pl in pointlists
        ]
        if max_spacing is None:
            return loops
        return [_refine_loop(loop, max_spacing) for loop in loops]

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Draw the shape as filled polygons into ``ax`` and return the axes.

        Keyword arguments style the polygons (``facecolor``, ``alpha``, ...).
        """
        if any(
            key in kwargs
            for key in ("x", "y", "z", "grid", "nodes", "normals", "resolution")
        ):
            return self.show(ax=ax, display=False, **kwargs)
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
