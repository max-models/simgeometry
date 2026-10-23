"""Tubes (and rods) along any axis whose far end is cut by a slanted plane."""

from __future__ import annotations

import cunumpy as xp
import numpy as np

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.objects.axisymmetric import _vector
from simgeometry.shapes import Shape3D, _component_major


def _orthonormal(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors normal to ``axis`` and to each other."""
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


class SlantedTube(Shape3D):
    """A tube from ``start`` to ``end`` whose end at ``end`` is cut by a slanted plane.

    The wall lies between ``inner_radius`` and ``outer_radius`` about the axis from
    ``start`` to ``end``; ``inner_radius = 0`` gives a solid rod. The end at
    ``start`` is the flat face normal to the axis. The end at ``end`` is the plane
    through ``end`` with the outward normal ``cut_normal``, tilted from the axis by
    :attr:`cut_angle`, like the mouth of an oven nozzle cut at 45°. ``length`` is
    measured along the axis to the centre of the cut, and the cut may not reach the
    start face (``outer_radius * tan(cut_angle) < length``).

    The :meth:`level_set` is the larger of the signed distance to the tube without
    the cut and to the cut plane: exact on the faces and inside away from the cut's
    rim, a bound elsewhere (``level_set_exact`` is False). :func:`~simgeometry.crossings`
    are analytic.

    Example
    -------
    A duct along z, 0.1 long, bore radius 0.01, wall 2 mm, cut at 45° so that its
    mouth faces down (-y)::

        c = np.cos(np.pi / 4)
        nozzle = SlantedTube((0, 0, 0), (0, 0, 0.1), 0.01, 0.012, (0, -c, c))
    """

    has_level_set = True
    level_set_exact = False

    def __init__(
        self,
        start: BoundsLike,
        end: BoundsLike,
        inner_radius: float,
        outer_radius: float,
        cut_normal: BoundsLike,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a slanted tube in 3D.

        Parameters:
        start (list or tuple): Centre of the flat end face.
        end (list or tuple): Centre of the slanted cut, on the axis.
        inner_radius (float): Radius of the bore, zero (a rod) or positive.
        outer_radius (float): Outer radius, larger than ``inner_radius``.
        cut_normal (list or tuple): Outward normal of the cut plane; its component
            along the axis (from ``start`` to ``end``) must be positive.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the tube.
        """
        start, end = _vector(start, "start"), _vector(end, "end")
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length == 0:
            raise ValueError("start and end must differ")
        if not 0 <= inner_radius < outer_radius:
            raise ValueError("A slanted tube needs 0 <= inner_radius < outer_radius")
        normal = _vector(cut_normal, "cut_normal")
        if np.linalg.norm(normal) == 0:
            raise ValueError("cut_normal must have a nonzero length")
        self._start = start
        self._length = length
        self._axis = axis / length
        self._normal = normal / np.linalg.norm(normal)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self._check_cut()
        self._bc = bc
        self._inverted = bool(inverted)
        super().__init__(*self._bounds())

    def _check_cut(self) -> None:
        cosine = float(self._normal @ self._axis)
        if cosine <= 0:
            raise ValueError("cut_normal must point along the axis, away from start")
        tangent = np.sqrt(max(1.0 - cosine**2, 0.0)) / cosine
        if self.outer_radius * tangent >= self._length:
            raise ValueError(
                "The cut reaches the start face: need outer_radius * tan(angle) < length"
            )

    @property
    def start(self) -> tuple[float, float, float]:
        """The centre of the flat end face."""
        return tuple(float(c) for c in self._start)

    @property
    def end(self) -> tuple[float, float, float]:
        """The centre of the slanted cut, on the axis."""
        return tuple(float(c) for c in self._start + self._length * self._axis)

    @property
    def axis(self) -> tuple[float, float, float]:
        """The unit vector from ``start`` to ``end``."""
        return tuple(float(a) for a in self._axis)

    @property
    def cut_normal(self) -> tuple[float, float, float]:
        """The unit outward normal of the cut plane."""
        return tuple(float(a) for a in self._normal)

    @property
    def cut_angle(self) -> float:
        """The angle between the cut plane and the cross-section, in degrees."""
        cosine = float(np.clip(self._normal @ self._axis, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @property
    def length(self) -> float:
        """The distance from ``start`` to ``end`` along the axis."""
        return self._length

    def _bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The box around the start disc and the elliptic rim of the cut."""
        radius = self.outer_radius
        first, second = _orthonormal(self._axis)
        disc = radius * np.sqrt(np.clip(1.0 - self._axis**2, 0.0, None))
        # rim: end + R (cos f u + sin f v) + s(f) a with s chosen to stay on the plane
        along = self._normal @ self._axis
        u = first - self._axis * (self._normal @ first) / along
        v = second - self._axis * (self._normal @ second) / along
        rim = radius * np.sqrt(u**2 + v**2)
        end = np.asarray(self.end)
        lower = np.minimum(self._start - disc, end - rim)
        upper = np.maximum(self._start + disc, end + rim)
        return tuple(float(c) for c in lower), tuple(float(c) for c in upper)

    def _update_bounds(self) -> None:
        self.set_bounding_box(*self._bounds())

    def copy(self) -> SlantedTube:
        """Create a copy of the slanted tube."""
        return SlantedTube(
            self.start,
            self.end,
            self.inner_radius,
            self.outer_radius,
            self.cut_normal,
            bc=self._bc,
            inverted=self._inverted,
        )

    def _coordinates(self, points: Array):
        """Axial distance from start, distance from the axis and height above the cut."""
        relative = points - xp.asarray(self._start)[:, None]
        axis = xp.asarray(self._axis)[:, None]
        s = xp.sum(relative * axis, axis=0)
        across = relative - s[None, :] * axis
        rho = xp.sqrt(xp.sum(across * across, axis=0))
        above = xp.sum(relative * xp.asarray(self._normal)[:, None], axis=0) - (
            self._length * float(self._normal @ self._axis)
        )
        return s, rho, above

    def contains_points(self, points: Array) -> Array:
        """Vectorised :meth:`contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        s, rho, above = self._coordinates(points)
        inside = (
            (s >= 0)
            & (above <= 0)
            & (rho <= self.outer_radius)
            & (rho >= self.inner_radius)
        )
        return ~inside if self.inverted else inside

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the slanted tube.

        Parameters:
        coords (list or tuple): Coordinates (x, y, z).

        Returns:
        bool: True if the point is inside the object, False otherwise.
        """
        point = xp.asarray(coords, dtype=float).reshape(3, 1)
        return bool(xp.to_numpy(self.contains_points(point))[0])

    def _level_set(self, points: Array) -> Array:
        """Larger of the distance to the uncut half-infinite tube and to the cut plane."""
        s, rho, above = self._coordinates(points)
        if self.inner_radius > 0:
            radial = xp.maximum(self.inner_radius - rho, rho - self.outer_radius)
        else:
            radial = rho - self.outer_radius
        axial = -s
        outside = xp.sqrt(xp.maximum(radial, 0.0) ** 2 + xp.maximum(axial, 0.0) ** 2)
        tube = outside + xp.minimum(xp.maximum(radial, axial), 0.0)
        return xp.maximum(tube, above)

    def translate(self, offset: tuple[float, float, float]) -> None:
        """
        Translate the slanted tube by the given offset.

        Parameters:
            offset (tuple): (dx, dy, dz) translation vector.
        """
        self._start = self._start + _vector(offset, "offset")
        self._update_bounds()

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the slanted tube across the plane where coordinate ``axis`` is ``position``.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis, 2 for z-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis).")
        self._start = self._start.copy()
        self._start[axis] = 2 * position - self._start[axis]
        self._axis = self._axis.copy()
        self._axis[axis] = -self._axis[axis]
        self._normal = self._normal.copy()
        self._normal[axis] = -self._normal[axis]
        self._update_bounds()

    def volume(self) -> float:
        """Calculate the volume: a cut through the axis point leaves that of a straight tube."""
        return float(
            np.pi * (self.outer_radius**2 - self.inner_radius**2) * self._length
        )

    def surface_area(self) -> float:
        """Calculate the surface area: both walls, the flat end and the elliptic cut."""
        ri, ro = self.inner_radius, self.outer_radius
        annulus = np.pi * (ro**2 - ri**2)
        cosine = float(self._normal @ self._axis)
        walls = 2 * np.pi * (ro + ri) * self._length
        return float(walls + annulus + annulus / cosine)

    def _first_crossing(self, a: Array, d: Array, start_tolerance: float):
        """Fraction ``(N,)`` (inf without a hit) and outward normal of the first crossing.

        Candidates are the roots on the four surfaces (outer and inner cylinder,
        start plane, cut plane) in ``(start_tolerance, 1]``; a root counts where the
        point lies within the other three constraints. Tangent roots are not
        crossings. The normal is that of the solid, before any inversion.
        """
        n = a.shape[1]
        origin = xp.asarray(self._start)[:, None]
        axis = xp.asarray(self._axis)[:, None]
        normal = xp.asarray(self._normal)[:, None]
        w = a - origin
        w_s = xp.sum(w * axis, axis=0)
        d_s = xp.sum(d * axis, axis=0)
        w_perp = w - w_s[None, :] * axis
        d_perp = d - d_s[None, :] * axis
        w_n = xp.sum(w * normal, axis=0) - self._length * float(
            self._normal @ self._axis
        )
        d_n = xp.sum(d * normal, axis=0)
        scale = max(self._length, self.outer_radius)
        slack = 1e-12 * scale

        def _within(t: Array, skip: str) -> Array:
            s = w_s + t * d_s
            rho2 = xp.sum((w_perp + t[None, :] * d_perp) ** 2, axis=0)
            above = w_n + t * d_n
            ok = xp.ones(t.shape, dtype=bool)
            if skip != "start":
                ok &= s >= -slack
            if skip != "cut":
                ok &= above <= slack
            if skip != "outer":
                ok &= rho2 <= (self.outer_radius + slack) ** 2
            if skip != "inner" and self.inner_radius > 0:
                ok &= rho2 >= (self.inner_radius - slack) ** 2
            return ok

        best = xp.full(n, xp.inf)
        kind = xp.full(n, -1, dtype=int)

        def _offer(t: Array, valid: Array, label: int, skip: str) -> None:
            nonlocal best, kind
            valid = valid & (t > start_tolerance) & (t <= 1.0)
            t_safe = xp.where(valid, t, 0.0)
            valid = valid & _within(t_safe, skip)
            better = valid & (t < best)
            best = xp.where(better, t, best)
            kind = xp.where(better, label, kind)

        for t, valid, label, skip in _plane_roots(w_s, d_s, w_n, d_n):
            _offer(t, valid, label, skip)
        aa = xp.sum(d_perp * d_perp, axis=0)
        bb = xp.sum(w_perp * d_perp, axis=0)
        ww = xp.sum(w_perp * w_perp, axis=0)
        radii = [(self.outer_radius, 2, "outer")]
        if self.inner_radius > 0:
            radii.append((self.inner_radius, 3, "inner"))
        for radius, label, skip in radii:
            disc = bb * bb - aa * (ww - radius**2)
            valid = (aa > 0) & (disc > 0)
            root = xp.sqrt(xp.where(valid, disc, 0.0))
            safe = xp.where(aa > 0, aa, 1.0)
            for sign in (-1.0, 1.0):
                _offer((-bb + sign * root) / safe, valid, label, skip)

        point_rel = w_perp + xp.where(xp.isfinite(best), best, 0.0)[None, :] * d_perp
        rho = xp.sqrt(xp.sum(point_rel * point_rel, axis=0))
        radial = point_rel / xp.where(rho > 0, rho, 1.0)[None, :]
        out = xp.zeros((3, n))
        out = xp.where((kind == 0)[None, :], -axis, out)
        out = xp.where((kind == 1)[None, :], normal, out)
        out = xp.where((kind == 2)[None, :], radial, out)
        out = xp.where((kind == 3)[None, :], -radial, out)
        return best, out


def _plane_roots(w_s, d_s, w_n, d_n):
    """Crossings with the start plane (label 0) and the cut plane (label 1)."""
    moving_s = d_s != 0
    t_start = -w_s / xp.where(moving_s, d_s, 1.0)
    moving_n = d_n != 0
    t_cut = -w_n / xp.where(moving_n, d_n, 1.0)
    return [(t_start, moving_s, 0, "start"), (t_cut, moving_n, 1, "cut")]
