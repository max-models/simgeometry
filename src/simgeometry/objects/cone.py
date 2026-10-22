"""Cones and frustums of revolution along any axis."""

from __future__ import annotations

import cunumpy as xp
import numpy as np

from simgeometry._level_set import capped_cone_distance
from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.objects.axisymmetric import CappedAxisymmetricShape


class Cone(CappedAxisymmetricShape):
    """A frustum with radius ``start_radius`` at ``start`` and ``end_radius`` at ``end``.

    One radius may be zero, which gives a cone with its tip there. The
    :meth:`level_set` is the exact signed distance.

    Example
    -------
    A cone of base radius 0.5 at z = 0 with its tip at z = 1::

        tip = Cone((0, 0, 0), (0, 0, 1), 0.5, 0.0)
    """

    def __init__(
        self,
        start: BoundsLike,
        end: BoundsLike,
        start_radius: float,
        end_radius: float,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a cone or frustum in 3D.

        Parameters:
        start (list or tuple): Centre of the end face at the start of the axis.
        end (list or tuple): Centre of the end face at the end of the axis.
        start_radius (float): Radius at ``start``, zero or positive.
        end_radius (float): Radius at ``end``, zero or positive.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the cone.
        """
        self._set_ends(start, end)
        if not (
            start_radius >= 0 and end_radius >= 0 and start_radius + end_radius > 0
        ):
            raise ValueError("A cone needs radii >= 0, at least one of them positive")
        self.start_radius = float(start_radius)
        self.end_radius = float(end_radius)
        self._bc = bc
        self._inverted = bool(inverted)
        super().__init__(*self._bounds())

    def _end_radii(self) -> tuple[float, float]:
        return self.start_radius, self.end_radius

    def copy(self) -> Cone:
        """Create a copy of the cone."""
        return Cone(
            self.start,
            self.end,
            self.start_radius,
            self.end_radius,
            bc=self._bc,
            inverted=self._inverted,
        )

    def _inside(self, s: Array, rho2: Array) -> Array:
        h = self._half_length
        radius = self.start_radius + (self.end_radius - self.start_radius) * (
            (s + h) / (2 * h)
        )
        return (xp.abs(s) <= h) & (rho2 <= radius * radius)

    def _profile_distance(self, s: Array, rho: Array) -> Array:
        return capped_cone_distance(
            rho, s, self._half_length, self.start_radius, self.end_radius
        )

    def volume(self) -> float:
        """Calculate the volume of the frustum."""
        r1, r2 = self.start_radius, self.end_radius
        return float(np.pi * self.length / 3 * (r1**2 + r1 * r2 + r2**2))

    def surface_area(self) -> float:
        """Calculate the surface area of the frustum, with its end discs."""
        r1, r2 = self.start_radius, self.end_radius
        slant = np.hypot(self.length, r1 - r2)
        return float(np.pi * (r1 + r2) * slant + np.pi * (r1**2 + r2**2))
