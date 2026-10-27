"""Two-dimensional aperture helpers for geometric object definitions."""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from simgeometry._types import ConfigValue
from simgeometry.objects.polygon import Polygon
from simgeometry.shapes import Shape2D


class Aperture2D(Shape2D):
    """Two plates mirrored about a beam axis: a slit, or the 2D cut of a round aperture.

    The beam runs along coordinate ``axis`` (0 for x, 1 for y), on the centre line
    where the other coordinate equals ``position``. ``vertices`` are the plate profile
    as ``(axial, radial)`` pairs: the coordinate along the beam and the distance from
    the centre line. One plate is the polygon at ``position + radial``, the other its
    mirror image at ``position - radial``. :meth:`slit` builds the common rectangular
    plates.

    Its :meth:`level_set` is the signed distance to the plates' edges, exact while the
    plates do not touch (every radial coordinate positive).
    """

    exact_outline = True
    has_level_set = True

    def _plate_vertices(self) -> tuple[ndarray, ndarray]:
        """The ``(n, 2)`` vertices of the plate below and the plate above the centre line."""
        profile = np.asarray(self.vertices, dtype=float)
        axial, radial = profile[:, 0], profile[:, 1]
        plates = []
        for across in (self.position - radial, self.position + radial):
            points = (axial, across) if self.axis == 0 else (across, axial)
            plates.append(np.stack(points, axis=1))
        return plates[0], plates[1]

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
            axis (int): The beam axis, 0 (along x) or 1 (along y).
            position (float): The centre line: the coordinate across the beam,
                ``y`` for ``axis=0`` and ``x`` for ``axis=1``.
            vertices (list or tuple): The plate profile as ``(axial, radial)``
                vertices.
            bc (optional): Boundary condition information.
            inverted (bool, optional): Cover everything except the two plates.
        """

        self.axis = axis
        self._position = position
        self.vertices = vertices
        self._bc = bc
        self._inverted = bool(inverted)

        lower, upper = self._plate_vertices()
        self._lower_polygon = Polygon(vertices=lower, bc=bc)
        self._upper_polygon = Polygon(vertices=upper, bc=bc)

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

    @classmethod
    def slit(
        cls,
        axial_position: float,
        half_width: float,
        thickness: float,
        length: float,
        *,
        axis: int = 0,
        position: float = 0.0,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> Aperture2D:
        """A slit: two rectangular plates with a gap of ``2 * half_width`` between them.

        The plates are ``thickness`` thick along the beam, centred on
        ``axial_position``, and extend ``length`` outward from the gap.
        """
        if not (half_width >= 0 and thickness > 0 and length > 0):
            raise ValueError(
                "A slit needs half_width >= 0 and positive thickness and length"
            )
        front, back = axial_position - thickness / 2, axial_position + thickness / 2
        tip, outer = half_width, half_width + length
        return cls(
            axis=axis,
            position=position,
            vertices=((front, tip), (back, tip), (back, outer), (front, outer)),
            bc=bc,
            inverted=inverted,
        )

    @property
    def axis(self) -> int:
        """Return the beam axis."""
        return self._axis

    @axis.setter
    def axis(self, value: int) -> None:
        """Set the beam axis."""
        assert value in [0, 1], "Axis must be one of 0 or 1."
        self._axis = value

    def _corner_loops(self) -> list[np.ndarray]:
        """One loop per plate: the lower plate, then the upper one."""
        return [
            loop
            for plate in (self._lower_polygon, self._upper_polygon)
            for loop in plate._corner_loops()
        ]

    def contains(self, coords: ndarray) -> bool:
        """
        Check if the given coordinates are inside one of the plates.

        Parameters:
        coords (list or tuple): Coordinates (x, y).

        Returns:
        bool: True if the point is inside a plate, False otherwise.
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

    def _level_set(self, points: ndarray) -> ndarray:
        """Signed distance to the edges of both plates."""
        return self._edge_level_set(points)

    @property
    def level_set_exact(self) -> bool:
        """Whether the plates are apart, so that the level set is exact."""
        return all(float(v[1]) > 0 for v in self.vertices)

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        """Return the plate profile as ``(axial, radial)`` vertices."""
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
        """Return the centre line's coordinate across the beam."""
        return self._position
