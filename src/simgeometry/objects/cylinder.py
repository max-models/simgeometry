"""Solid cylinders along any axis, optionally with rounded edges."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from methodcard import Rule, method

from simgeometry._level_set import rectangle_distance
from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.objects.axisymmetric import (
    _AXIS,
    _CENTRE,
    AxisymmetricShape,
    CappedAxisymmetricShape,
    _frame,
    _rectangle_distance,
)
from simgeometry.shapes import _level_set_at


def _cylinder_rules() -> dict:
    """The SymPy rule of the `Cylinder` level set on an oblique axis (for the tests)."""
    import sympy

    x, y, z, radius, half, f = sympy.symbols("x y z radius half f", real=True)

    def level_set(x, y, z, radius, half, f):
        start = np.subtract(_CENTRE, np.multiply(half, _AXIS))
        end = np.add(_CENTRE, np.multiply(half, _AXIS))
        return _level_set_at(Cylinder(start, end, radius, fillet=f), x, y, z)

    s, rho = _frame((x, y, z), _CENTRE, _AXIS)
    rule = _rectangle_distance(rho, s, radius - f, half - f) - f
    ranges = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-2.0, 2.0)}
    ranges |= {"radius": (0.5, 1.5), "half": (0.5, 1.5), "f": (0.0, 0.25)}
    return {"level_set": Rule(rule, implementation=level_set, ranges=ranges)}


@method(
    equations={
        "inside": r"p \in S \iff |s| \le L,\; \rho^2 \le R^2 \quad (f = 0), \qquad p \in S \iff \phi(p) \le 0 \quad (f > 0)",
        "level_set": r"\phi = \phi_\mathrm{rect}(\rho, s;\ R - f, L - f) - f, \qquad \phi_\mathrm{rect}(x, y; h_x, h_y) = \big|\max(d, 0)\big| + \min(\max(d_x, d_y), 0), \quad d = (|x| - h_x, |y| - h_y)",
    },
    assumptions=(
        r"half length $L$, radius $R$, in the frame $(s, \rho)$ of `AxisymmetricShape`",
        r"the fillet $f \le \min(R, L)$ rounds the two circular edges",
        r"$\phi$ is the exact signed distance, negative inside",
    ),
    components=(AxisymmetricShape,),
    symbolic=_cylinder_rules,
)
class Cylinder(CappedAxisymmetricShape):
    """A solid circular cylinder with its axis from ``start`` to ``end``.

    ``fillet`` rounds the two circular edges with that radius (at most the radius
    and half the length). The `level_set` is the exact signed distance, with
    or without the fillet.

    Example
    -------
    A rod of radius 0.1 along z, from z = -1 to z = 1::

        rod = Cylinder((0, 0, -1), (0, 0, 1), 0.1)
    """

    def __init__(
        self,
        start: BoundsLike,
        end: BoundsLike,
        radius: float,
        bc: ConfigValue = None,
        inverted: bool = False,
        *,
        fillet: float = 0.0,
    ) -> None:
        """
        Initialize a cylinder in 3D.

        Parameters:
        start (list or tuple): Centre of the end face at the start of the axis.
        end (list or tuple): Centre of the end face at the end of the axis.
        radius (float): Radius of the cylinder.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the cylinder.
        fillet (float, optional): Radius of the rounded circular edges.
        """
        self._set_ends(start, end)
        if not radius > 0:
            raise ValueError("Radius must be a positive number.")
        if not 0 <= fillet <= min(radius, self._half_length):
            raise ValueError("fillet must be between 0 and min(radius, length / 2)")
        self.radius = float(radius)
        self.fillet = float(fillet)
        self._bc = bc
        self._inverted = bool(inverted)
        super().__init__(*self._bounds())

    def _end_radii(self) -> tuple[float, float]:
        return self.radius, self.radius

    def copy(self) -> Cylinder:
        """Create a copy of the cylinder."""
        return Cylinder(
            self.start,
            self.end,
            self.radius,
            bc=self._bc,
            inverted=self._inverted,
            fillet=self.fillet,
        )

    def _profile_distance(self, s: Array, rho: Array) -> Array:
        f = self.fillet
        return rectangle_distance(rho, s, self.radius - f, self._half_length - f) - f

    def _inside(self, s: Array, rho2: Array) -> Array:
        if self.fillet > 0:
            return self._profile_distance(s, xp.sqrt(rho2)) <= 0
        return (xp.abs(s) <= self._half_length) & (rho2 <= self.radius**2)

    def volume(self) -> float:
        """Calculate the volume of the cylinder, less the rounded edges."""
        r, h, f = self.radius, self._half_length, self.fillet
        removed = 4 * np.pi * f**2 * ((1 - np.pi / 4) * (r - f) + f / 6)
        return float(2 * np.pi * r**2 * h - removed)

    def surface_area(self) -> float:
        """Calculate the surface area of the cylinder, with its rounded edges."""
        r, h, f = self.radius, self._half_length, self.fillet
        caps = 2 * np.pi * (r - f) ** 2
        side = 4 * np.pi * r * (h - f)
        rounds = 2 * np.pi**2 * f * (r - f) + 4 * np.pi * f**2
        return float(caps + side + rounds)
