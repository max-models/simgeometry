"""Solid tori about any axis."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from methodcard import Rule, method

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.objects.axisymmetric import (
    _AXIS,
    _CENTRE,
    AxisymmetricShape,
    _frame,
    _vector,
)
from simgeometry.shapes import _level_set_at


def _torus_rules() -> dict:
    """The SymPy rule of the `Torus` level set on an oblique axis (for the tests)."""
    import sympy

    x, y, z, major, minor = sympy.symbols("x y z major minor", real=True)

    def level_set(x, y, z, major, minor):
        return _level_set_at(Torus(_CENTRE, major, minor, axis=_AXIS), x, y, z)

    s, rho = _frame((x, y, z), _CENTRE, _AXIS)
    rule = sympy.sqrt((rho - major) ** 2 + s**2) - minor
    ranges = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-2.0, 2.0)}
    ranges |= {"major": (1.0, 1.5), "minor": (0.2, 0.9)}
    return {"level_set": Rule(rule, implementation=level_set, ranges=ranges)}


@method(
    equations={
        "inside": r"p \in S \iff (\rho - R)^2 + s^2 \le r^2",
        "level_set": r"\phi = \sqrt{(\rho - R)^2 + s^2} - r",
    },
    assumptions=(
        r"major radius $R$, minor radius $0 < r \le R$, centre circle in the plane $s = 0$ of the frame of `AxisymmetricShape`",
        r"$\phi$ is the exact signed distance, negative inside",
    ),
    references=(
        "[Quílez, distance functions (torus)](https://iquilezles.org/articles/distfunctions/)",
    ),
    components=(AxisymmetricShape,),
    symbolic=_torus_rules,
)
class Torus(AxisymmetricShape):
    """A solid torus: the points within ``minor_radius`` of a circle of ``major_radius``.

    The circle is centred at ``midpoint`` in the plane normal to ``axis``. The
    `level_set` is the exact signed distance.

    Example
    -------
    A torus with major radius 3 and tube radius 1 about the z axis::

        ring = Torus((0, 0, 0), 3.0, 1.0)
    """

    def _bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The centre circle's box, widened by the tube radius."""
        extent = self._disc_extent(self.major_radius) + self.minor_radius
        lower = self._midpoint - extent
        upper = self._midpoint + extent
        return tuple(float(v) for v in lower), tuple(float(v) for v in upper)

    def __init__(
        self,
        midpoint: BoundsLike,
        major_radius: float,
        minor_radius: float,
        axis: BoundsLike = (0.0, 0.0, 1.0),
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a torus in 3D.

        Parameters:
        midpoint (list or tuple): Centre of the torus.
        major_radius (float): Radius of the centre circle of the tube.
        minor_radius (float): Radius of the tube, at most ``major_radius``.
        axis (list or tuple, optional): Symmetry axis, normal to the centre circle.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the torus.
        """
        if not 0 < minor_radius <= major_radius:
            raise ValueError("A torus needs 0 < minor_radius <= major_radius")
        self._set_frame(_vector(midpoint, "midpoint"), _vector(axis, "axis"))
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self._bc = bc
        self._inverted = bool(inverted)
        super().__init__(*self._bounds())

    def copy(self) -> Torus:
        """Create a copy of the torus."""
        return Torus(
            self.midpoint,
            self.major_radius,
            self.minor_radius,
            axis=self.axis,
            bc=self._bc,
            inverted=self._inverted,
        )

    def _inside(self, s: Array, rho2: Array) -> Array:
        across = xp.sqrt(rho2) - self.major_radius
        return across * across + s * s <= self.minor_radius**2

    def _profile_distance(self, s: Array, rho: Array) -> Array:
        across = rho - self.major_radius
        return xp.sqrt(across * across + s * s) - self.minor_radius

    def volume(self) -> float:
        """Calculate the volume of the torus."""
        return float(2 * np.pi**2 * self.major_radius * self.minor_radius**2)

    def surface_area(self) -> float:
        """Calculate the surface area of the torus."""
        return float(4 * np.pi**2 * self.major_radius * self.minor_radius)
