"""Helical wires: the points within a radius of a helix, e.g. the winding of an RF coil."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from methodcard import method

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.objects.axisymmetric import _vector
from simgeometry.shapes import Shape3D, _component_major


def _reference(axis: np.ndarray) -> np.ndarray:
    """The unit vector normal to ``axis`` closest to x (to y for an axis along x)."""
    helper = np.eye(3)[0] if abs(axis[0]) < 0.9 else np.eye(3)[1]
    first = helper - (helper @ axis) * axis
    return first / np.linalg.norm(first)


@method(
    equations={
        "centreline": r"\mathbf c(t) = \mathbf p_0 + R\,(\cos t\,\mathbf u + \sin t\,\mathbf v) + \frac{p\,t}{2\pi}\,\mathbf a, \qquad 0 \le t \le 2\pi N",
        "level_set": r"\phi(\mathbf x) = \min_{0 \le t \le 2\pi N} |\mathbf x - \mathbf c(t)| - r_w",
        "descent": r"t \leftarrow t - \frac{f'(t)}{f''(t) \text{ or } 2\rho R + 2b^2}, \qquad f(t) = \rho^2 + R^2 - 2\rho R\cos(t - \varphi) + (s - b t)^2, \quad b = \frac{p}{2\pi}",
        "bound": r"|\mathbf x - \mathbf c(t)| \ge \sqrt{(\rho - R)^2 + \max(-s, s - pN, 0)^2}",
    },
    assumptions=(
        r"$\mathbf a$ the unit axis, $\mathbf u$ the direction of the start point from the axis, $\mathbf v = \pm\mathbf a\times\mathbf u$ (right- or left-handed); $(s, \rho, \varphi)$ the point's cylindrical coordinates in that frame",
        r"the wire's ends are hemispheres (the points within $r_w$ of the end points of the centreline)",
        r"within $2 r_w$ of the cylinder the centreline lies on: the minimum over $t$ from four starts, the two ends and the linearised minima of the two turns around $s / b$, by six Newton steps (descent steps on the majorant $f'' \le 2\rho R + 2b^2$ where $f''$ is not clearly positive), clipped to $[0, 2\pi N]$; exact there (tested against dense sampling)",
        r"farther away $\phi$ is the bound (the distance from the cylinder, minus $r_w$): a level set that never exceeds the distance",
    ),
)
class Helix(Shape3D):
    """A helical wire of round cross-section: the points within ``wire_radius`` of a helix.

    The centreline winds ``turns`` times around ``axis`` at ``radius``, advancing
    ``pitch`` along the axis per turn, from its start point at
    ``start + radius * u`` (``u`` the unit vector normal to the axis in the
    direction of ``start_direction``, by default the one closest to x) to its end
    ``pitch * turns`` further along the axis. Right-handed by default (counter-
    clockwise seen from the end the axis points to); ``left_handed`` reverses it. The
    wire's ends are hemispherical caps. The `level_set` is the signed distance to
    the centreline minus the wire's radius, found by Newton steps on the squared
    distance along the centreline, within a wire radius of the wire; farther away it is
    a lower bound (``level_set_exact`` is False).

    Example
    -------
    The six windings of an RF coil, radius 0.14 m, 2 cm pitch, a copper tube of 6 mm
    outer diameter, along z from z = 0.02::

        coil = Helix((0, 0, 0.02), 0.14, 0.02, 6, 0.003)
    """

    has_level_set = True
    level_set_exact = False

    def _bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The box of the cylinder of radius ``radius + wire_radius`` around the wire."""
        extent = (self.radius + self.wire_radius) * np.sqrt(
            np.clip(1.0 - self._axis**2, 0.0, None)
        )
        first = self._start - self.wire_radius * self._axis
        last = self._start + (self.length + self.wire_radius) * self._axis
        lower = np.minimum(first, last) - extent
        upper = np.maximum(first, last) + extent
        return tuple(float(c) for c in lower), tuple(float(c) for c in upper)

    def __init__(
        self,
        start: BoundsLike,
        radius: float,
        pitch: float,
        turns: float,
        wire_radius: float,
        axis: BoundsLike = (0.0, 0.0, 1.0),
        start_direction: BoundsLike | None = None,
        left_handed: bool = False,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a helical wire in 3D.

        Parameters:
        start (list or tuple): The point on the axis level with the wire's start.
        radius (float): The radius of the centreline.
        pitch (float): The advance along the axis per turn, positive.
        turns (float): The number of turns, positive (need not be whole).
        wire_radius (float): The wire's radius, below ``radius``.
        axis (list or tuple, optional): The direction the helix advances along.
        start_direction (list or tuple, optional): The direction from the axis to the
            start point (its part normal to the axis); default the one closest to x.
        left_handed (bool, optional): Wind clockwise seen from the end ``axis``
            points to.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the wire.
        """
        axis = _vector(axis, "axis")
        if np.linalg.norm(axis) == 0:
            raise ValueError("The axis must have a nonzero length")
        axis = axis / np.linalg.norm(axis)
        if not 0 < wire_radius < radius:
            raise ValueError("A helix needs 0 < wire_radius < radius")
        if pitch <= 0 or turns <= 0:
            raise ValueError("A helix needs pitch > 0 and turns > 0")
        if start_direction is None:
            u = _reference(axis)
        else:
            direction = _vector(start_direction, "start_direction")
            u = direction - (direction @ axis) * axis
            if np.linalg.norm(u) < 1e-12:
                raise ValueError("start_direction must not be along the axis")
            u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        self._start = _vector(start, "start")
        self._axis = axis
        self._u = u
        self._v = -v if left_handed else v
        self.radius = float(radius)
        self.pitch = float(pitch)
        self.turns = float(turns)
        self.wire_radius = float(wire_radius)
        self._bc = bc
        self._inverted = bool(inverted)
        super().__init__(*self._bounds())

    @property
    def start(self) -> tuple[float, float, float]:
        """The point on the axis level with the wire's start."""
        return tuple(float(c) for c in self._start)

    @property
    def axis(self) -> tuple[float, float, float]:
        """The unit vector the helix advances along."""
        return tuple(float(c) for c in self._axis)

    @property
    def start_direction(self) -> tuple[float, float, float]:
        """The unit vector from the axis to the start point."""
        return tuple(float(c) for c in self._u)

    @property
    def left_handed(self) -> bool:
        """Whether the helix winds clockwise seen from the end the axis points to."""
        return bool(np.cross(self._axis, self._u) @ self._v < 0)

    @property
    def length(self) -> float:
        """The axial length of the centreline, ``pitch * turns``."""
        return self.pitch * self.turns

    @property
    def wire_length(self) -> float:
        """The length of the centreline, ``turns * sqrt((2 pi radius)^2 + pitch^2)``."""
        return float(self.turns * np.hypot(2 * np.pi * self.radius, self.pitch))

    def centreline(self, t: Array) -> Array:
        """The points ``(3, n)`` of the centreline at the angles ``t`` (``0`` to ``2 pi N``)."""
        t = xp.asarray(t, dtype=float)
        b = self.pitch / (2 * np.pi)
        return (
            xp.asarray(self._start)[:, None]
            + self.radius * xp.cos(t)[None, :] * xp.asarray(self._u)[:, None]
            + self.radius * xp.sin(t)[None, :] * xp.asarray(self._v)[:, None]
            + b * t[None, :] * xp.asarray(self._axis)[:, None]
        )

    def copy(self) -> Helix:
        """Create a copy of the helix."""
        return Helix(
            self.start,
            self.radius,
            self.pitch,
            self.turns,
            self.wire_radius,
            axis=self.axis,
            start_direction=self.start_direction,
            left_handed=self.left_handed,
            bc=self._bc,
            inverted=self._inverted,
        )

    def _frame_coordinates(self, points: Array):
        """Axial position ``s``, distance ``rho`` from the axis and angle ``phi``."""
        relative = points - xp.asarray(self._start)[:, None]
        s = xp.sum(relative * xp.asarray(self._axis)[:, None], axis=0)
        x = xp.sum(relative * xp.asarray(self._u)[:, None], axis=0)
        y = xp.sum(relative * xp.asarray(self._v)[:, None], axis=0)
        return s, xp.sqrt(x * x + y * y), xp.arctan2(y, x)

    def _bound(self, s: Array, rho: Array) -> Array:
        """A lower bound of the distance from the centreline: from its cylinder."""
        beyond = xp.maximum(xp.maximum(-s, s - self.length), 0.0)
        return xp.sqrt((rho - self.radius) ** 2 + beyond * beyond)

    def _nearest(self, s: Array, rho: Array, phi: Array) -> Array:
        """The distance from the centreline, by descent along it from four starts."""
        radius, b = self.radius, self.pitch / (2 * np.pi)
        top, turn = 2 * np.pi * self.turns, 2 * np.pi
        coupling = rho * radius
        majorant = 2 * coupling + 2 * b * b
        k = xp.floor((s / b - phi) / turn)
        starts = []
        for j in (0, 1):
            # the minimum of the cosine's basin around phi + 2 pi (k + j), linearised:
            # rho R sin(tau) = b (s - b t) with sin(tau) = tau
            base = phi + turn * (k + j)
            # moved by whole turns into the range (points beyond the ends)
            base = xp.where(
                base > top, base - turn * xp.ceil((base - top) / turn), base
            )
            base = xp.where(base < 0, base + turn * xp.ceil(-base / turn), base)
            starts.append(base + b * (s - b * base) / (coupling + b * b))

        def descend(t, s, phi, coupling, majorant):
            t = xp.clip(t, 0.0, top)
            for _ in range(5):
                angle = t - phi
                slope = 2 * coupling * xp.sin(angle) - 2 * b * (s - b * t)
                curvature = 2 * coupling * xp.cos(angle) + 2 * b * b
                # Newton where the curvature is clearly positive, else a descent
                # step on the majorant f'' <= 2 rho R + 2 b^2
                divisor = xp.where(curvature > 0.05 * majorant, curvature, majorant)
                t = xp.clip(t - slope / divisor, 0.0, top)
            return t

        def squared(t, s, rho, phi):
            return (
                rho * rho
                + radius**2
                - 2 * rho * radius * xp.cos(t - phi)
                + (s - b * t) ** 2
            )

        best = None
        for t in starts:
            t = descend(t, s, phi, coupling, majorant)
            value = squared(t, s, rho, phi)
            best = value if best is None else xp.minimum(best, value)
        # the two ends, for the points within a pitch of them (axially)
        margin = self.pitch + self.wire_radius
        for end in (0.0, top):
            near = xp.flatnonzero(xp.abs(s - b * end) < margin)
            if near.shape[0]:
                sub = (s[near], phi[near], coupling[near], majorant[near])
                t = descend(xp.full(near.shape, end), *sub)
                best[near] = xp.minimum(
                    best[near], squared(t, s[near], rho[near], phi[near])
                )
        return xp.sqrt(xp.maximum(best, 0.0))

    def _distance(self, points: Array, reach: float) -> Array:
        """The distance from the centreline: exact within ``reach`` of its cylinder.

        Farther away the lower bound of `_bound` (the distance from the cylinder the
        centreline lies on).
        """
        s, rho, phi = self._frame_coordinates(points)
        distance = self._bound(s, rho)
        near = xp.flatnonzero(distance < reach)
        if near.shape[0]:
            distance[near] = self._nearest(s[near], rho[near], phi[near])
        return distance

    def _level_set(self, points: Array) -> Array:
        """The distance from the centreline minus the wire's radius (a bound beyond it)."""
        return self._distance(points, 2 * self.wire_radius) - self.wire_radius

    def contains_points(self, points: Array) -> Array:
        """Vectorised `contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        inside = self._distance(points, self.wire_radius) <= self.wire_radius
        return ~inside if self.inverted else inside

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the wire.

        Parameters:
        coords (list or tuple): Coordinates (x, y, z).

        Returns:
        bool: True if the point is inside the object, False otherwise.
        """
        point = xp.asarray(coords, dtype=float).reshape(3, 1)
        return bool(xp.to_numpy(self.contains_points(point))[0])

    def translate(self, offset: tuple[float, float, float]) -> None:
        """
        Translate the helix by the given offset.

        Parameters:
            offset (tuple): (dx, dy, dz) translation vector.
        """
        self._start = self._start + _vector(offset, "offset")
        self.set_bounding_box(*self._bounds())

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the helix across the plane where coordinate ``axis`` is ``position``.

        A mirror image of a right-handed helix is left-handed.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis, 2 for z-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis).")
        flip = np.ones(3)
        flip[axis] = -1.0
        self._start = self._start * flip
        self._start[axis] += 2 * position
        self._axis = self._axis * flip
        self._u = self._u * flip
        self._v = self._v * flip
        self.set_bounding_box(*self._bounds())

    def volume(self) -> float:
        """The volume: the swept disc along the centreline and the two hemispheres.

        Exact (Pappus for a curve) as long as the wire does not overlap itself
        (``pitch >= 2 * wire_radius`` and ``wire_radius`` below the radius of
        curvature).
        """
        disc = np.pi * self.wire_radius**2
        return float(disc * self.wire_length + 4 / 3 * np.pi * self.wire_radius**3)

    def surface_area(self) -> float:
        """The surface area: the swept circle and the two hemispheres (no overlap)."""
        r = self.wire_radius
        return float(2 * np.pi * r * self.wire_length + 4 * np.pi * r**2)
