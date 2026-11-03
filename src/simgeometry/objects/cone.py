"""Cones and frustums of revolution along any axis."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from methodcard import Rule, method

from simgeometry._level_set import capped_cone_distance
from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.objects.axisymmetric import (
    _AXIS,
    _CENTRE,
    AxisymmetricShape,
    CappedAxisymmetricShape,
    _frame,
)
from simgeometry.shapes import _level_set_at


def _cone_rules() -> dict:
    """The SymPy rule of the `Cone` level set on an oblique axis (for the tests)."""
    import sympy

    x, y, z, r1, r2, half = sympy.symbols("x y z r1 r2 half", real=True)

    def level_set(x, y, z, r1, r2, half):
        start = np.subtract(_CENTRE, np.multiply(half, _AXIS))
        end = np.add(_CENTRE, np.multiply(half, _AXIS))
        return _level_set_at(Cone(start, end, r1, r2), x, y, z)

    s, rho = _frame((x, y, z), _CENTRE, _AXIS)
    cap = sympy.Piecewise((r1, s < 0), (r2, True))
    cap_x, cap_y = rho - sympy.Min(rho, cap), sympy.Abs(s) - half
    side_x, side_y = r1 - r2, -2 * half
    t = ((rho - r2) * side_x + (s - half) * side_y) / (side_x**2 + side_y**2)
    t = sympy.Min(sympy.Max(t, 0), 1)
    slant_x, slant_y = rho - (r2 + t * side_x), s - (half + t * side_y)
    distance = sympy.sqrt(sympy.Min(cap_x**2 + cap_y**2, slant_x**2 + slant_y**2))
    inside = sympy.And(slant_x < 0, cap_y < 0)
    rule = sympy.Piecewise((-distance, inside), (distance, True))
    ranges = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-2.0, 2.0)}
    ranges |= {"r1": (0.0, 1.0), "r2": (0.2, 1.2), "half": (0.5, 1.5)}
    return {"level_set": Rule(rule, implementation=level_set, ranges=ranges)}


@method(
    equations={
        "inside": r"p \in S \iff |s| \le L,\; \rho \le r_1 + (r_2 - r_1)\,\frac{s + L}{2L}",
        "level_set": r"\phi = \pm \min\Big(\big|(\rho - \min(\rho, r_\mathrm{cap}),\ |s| - L)\big|,\ \big|(\rho, s) - q\big|\Big), \qquad q = (r_2, L) + \operatorname{clip}(\tau, 0, 1)\,(r_1 - r_2, -2L)",
        "slant": r"\tau = \frac{(\rho - r_2)(r_1 - r_2) - 2L\,(s - L)}{(r_1 - r_2)^2 + 4L^2}",
    },
    assumptions=(
        r"radius $r_1$ at $s = -L$ (`start`), $r_2$ at $s = L$ (`end`), one of them may be zero; frame $(s, \rho)$ of `AxisymmetricShape`",
        r"$r_\mathrm{cap}$ is the radius of the nearer cap; the minus sign where the point is on the inner side of the slant and $|s| < L$",
        r"$\phi$ is the exact signed distance",
    ),
    references=(
        "[Quílez, distance functions (capped cone)](https://iquilezles.org/articles/distfunctions/)",
    ),
    components=(AxisymmetricShape,),
    symbolic=_cone_rules,
)
class Cone(CappedAxisymmetricShape):
    """A frustum with radius ``start_radius`` at ``start`` and ``end_radius`` at ``end``.

    One radius may be zero, which gives a cone with its tip there. The
    `level_set` is the exact signed distance.

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
