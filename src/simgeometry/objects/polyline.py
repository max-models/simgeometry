"""Open polylines: curves of zero thickness, such as thin plates and wires."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from numpy import ndarray

from simgeometry._types import ConfigValue
from simgeometry.shapes import Shape2D, _component_major
from simgeometry.utils import vertices_to_bounds


class Polyline(Shape2D):
    """An open curve through ``vertices``: a thin plate, wire or grid of zero thickness.

    It encloses no area, so :meth:`contains` is always False and it cannot be
    inverted or combined in Boolean shapes. Its geometry is its outline, an open
    polyline from the first vertex to the last (``closed`` is False), which
    :meth:`~simgeometry.shapes.Shape2D.boundary_panels` and ``crossings()`` use
    directly.
    """

    exact_outline = True
    closed = False

    def __init__(self, vertices: ndarray, bc: ConfigValue = None) -> None:
        """
        Initialize a polyline in 2D space.

        Parameters:
        vertices (list or tuple): At least two ``(x, y)`` vertices, in order along the
            curve; consecutive vertices must differ.
        bc (optional): Boundary condition information.
        """
        points = np.asarray(xp.to_numpy(xp.asarray(vertices, dtype=float)), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError("A polyline needs at least two (x, y) vertices")
        if np.any(np.hypot(*np.diff(points, axis=0).T) == 0):
            raise ValueError("Consecutive polyline vertices must differ")
        self._vertices = points
        self._bc = bc
        lower_bound, upper_bound = vertices_to_bounds(points)
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound)

    @property
    def vertices(self) -> ndarray:
        """The ``(n, 2)`` vertices, one row per point, as given."""
        return self._vertices

    @property
    def length(self) -> float:
        """The length of the curve."""
        return float(np.hypot(*np.diff(self._vertices, axis=0).T).sum())

    def _corner_loops(self) -> list[np.ndarray]:
        """The vertices as one open ``(2, n)`` polyline."""
        return [self._vertices.T.copy()]

    def contains(self, coords: tuple[float, float]) -> bool:
        """Always False: a curve encloses no area."""
        return False

    def contains_points(self, points: ndarray) -> ndarray:
        """An all-False mask of the ``(2, n)`` points: a curve encloses no area."""
        points = _component_major(points, 2)
        return xp.zeros(points.shape[1], dtype=bool)
