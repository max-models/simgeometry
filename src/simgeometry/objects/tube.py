"""Hollow cylinders (tubes, ring electrodes) along any axis, optionally rounded."""

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


def _tube_rules() -> dict:
    """The SymPy rule of the `Tube` level set on an oblique axis (for the tests)."""
    import sympy

    x, y, z, inner, outer, half, f = sympy.symbols(
        "x y z inner outer half f", real=True
    )

    def level_set(x, y, z, inner, outer, half, f):
        start = np.subtract(_CENTRE, np.multiply(half, _AXIS))
        end = np.add(_CENTRE, np.multiply(half, _AXIS))
        return _level_set_at(Tube(start, end, inner, outer, fillet=f), x, y, z)

    s, rho = _frame((x, y, z), _CENTRE, _AXIS)
    middle, thickness = (inner + outer) / 2, (outer - inner) / 2
    rule = _rectangle_distance(rho - middle, s, thickness - f, half - f) - f
    ranges = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-2.0, 2.0)}
    ranges |= {"inner": (0.3, 0.6), "outer": (0.8, 1.5), "half": (0.5, 1.5)}
    ranges |= {"f": (0.0, 0.1)}
    return {"level_set": Rule(rule, implementation=level_set, ranges=ranges)}


@method(
    equations={
        "inside": r"p \in S \iff |s| \le L,\; r_i^2 \le \rho^2 \le r_o^2 \quad (f = 0), \qquad p \in S \iff \phi(p) \le 0 \quad (f > 0)",
        "level_set": r"\phi = \phi_\mathrm{rect}\!\left(\rho - \tfrac{r_i + r_o}{2}, s;\ \tfrac{r_o - r_i}{2} - f, L - f\right) - f",
    },
    assumptions=(
        r"half length $L$, $0 < r_i < r_o$, in the frame $(s, \rho)$ of `AxisymmetricShape`; $\phi_\mathrm{rect}$ the 2D box distance as for `Cylinder`",
        r"the fillet $f \le \min(\tfrac{r_o - r_i}{2}, L)$ rounds the four circular edges",
        r"$\phi$ is the exact signed distance, negative inside",
    ),
    components=(AxisymmetricShape,),
    symbolic=_tube_rules,
)
class Tube(CappedAxisymmetricShape):
    """A hollow cylinder between ``inner_radius`` and ``outer_radius``, from ``start`` to ``end``.

    A short tube is a ring electrode; a long one a drift tube or a chamber wall.
    ``fillet`` rounds its four circular edges (at most half the wall thickness and
    half the length). The `level_set` is the exact signed distance, with or
    without the fillet.

    Example
    -------
    A ring of bore radius 1 and outer radius 1.5, 0.4 long, centred at z = 0::

        ring = Tube((0, 0, -0.2), (0, 0, 0.2), 1.0, 1.5)
    """

    def __init__(
        self,
        start: BoundsLike,
        end: BoundsLike,
        inner_radius: float,
        outer_radius: float,
        bc: ConfigValue = None,
        inverted: bool = False,
        *,
        fillet: float = 0.0,
    ) -> None:
        """
        Initialize a tube in 3D.

        Parameters:
        start (list or tuple): Centre of the end face at the start of the axis.
        end (list or tuple): Centre of the end face at the end of the axis.
        inner_radius (float): Radius of the bore, positive.
        outer_radius (float): Outer radius, larger than ``inner_radius``.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the tube.
        fillet (float, optional): Radius of the rounded circular edges.
        """
        self._set_ends(start, end)
        if not 0 < inner_radius < outer_radius:
            raise ValueError("A tube needs 0 < inner_radius < outer_radius")
        thickness = outer_radius - inner_radius
        if not 0 <= fillet <= min(0.5 * thickness, self._half_length):
            raise ValueError(
                "fillet must be between 0 and min(wall thickness, length) / 2"
            )
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.fillet = float(fillet)
        self._bc = bc
        self._inverted = bool(inverted)
        super().__init__(*self._bounds())

    def _end_radii(self) -> tuple[float, float]:
        return self.outer_radius, self.outer_radius

    def copy(self) -> Tube:
        """Create a copy of the tube."""
        return Tube(
            self.start,
            self.end,
            self.inner_radius,
            self.outer_radius,
            bc=self._bc,
            inverted=self._inverted,
            fillet=self.fillet,
        )

    def _profile_distance(self, s: Array, rho: Array) -> Array:
        f = self.fillet
        middle = 0.5 * (self.inner_radius + self.outer_radius)
        half_thickness = 0.5 * (self.outer_radius - self.inner_radius)
        return (
            rectangle_distance(
                rho - middle, s, half_thickness - f, self._half_length - f
            )
            - f
        )

    def _inside(self, s: Array, rho2: Array) -> Array:
        if self.fillet > 0:
            return self._profile_distance(s, xp.sqrt(rho2)) <= 0
        return (
            (xp.abs(s) <= self._half_length)
            & (rho2 >= self.inner_radius**2)
            & (rho2 <= self.outer_radius**2)
        )

    def volume(self) -> float:
        """Calculate the volume of the tube, less the rounded edges."""
        ri, ro, h, f = (
            self.inner_radius,
            self.outer_radius,
            self._half_length,
            self.fillet,
        )
        removed = 4 * np.pi * f**2 * (1 - np.pi / 4) * (ro + ri)
        return float(2 * np.pi * h * (ro**2 - ri**2) - removed)

    def surface_area(self) -> float:
        """Calculate the surface area of the tube, inside and out, with its rounded edges."""
        ri, ro, h, f = (
            self.inner_radius,
            self.outer_radius,
            self._half_length,
            self.fillet,
        )
        sides = 4 * np.pi * (ro + ri) * (h - f)
        caps = 2 * np.pi * ((ro - f) ** 2 - (ri + f) ** 2)
        rounds = 2 * np.pi**2 * f * (ro + ri)
        return float(sides + caps + rounds)
