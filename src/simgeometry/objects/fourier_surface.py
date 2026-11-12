"""Toroidal surfaces as Fourier series: a stellarator's plasma boundary or first wall.

`FourierSurface` is VMEC's and ConStellaration's ``SurfaceRZFourier``:

    R(theta, phi) = sum_mn r_cos[m, n] cos(m theta - nfp n phi) (+ r_sin[m, n] sin(...))
    Z(theta, phi) = sum_mn z_sin[m, n] sin(m theta - nfp n phi) (+ z_cos[m, n] cos(...))

with ``m = 0 ... M`` along the first axis, ``n = -N ... N`` along the second, ``nfp``
field periods and ``phi`` the cylindrical toroidal angle, so the point is
``(R cos phi, R sin phi, Z)``.

Membership and the search for hits run on a triangulation of the surface
(`TriangleMesh`, ``resolution`` points per poloidal turn and per field period).
The hits are then moved onto the exact surface by Newton steps on
``a + t d = X(theta, phi)``, from the triangle's angles, so their position and normal
are exact to round-off; whether a segment hits at all is decided by the triangles,
which deviate from the surface by `triangulation_error`, second order in the spacing.

`FourierSurface.offset` builds a wall around a plasma boundary: every point moved
by a distance along the outward normal, the result fitted again as a Fourier series.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from methodcard import ConvergenceProblem, method

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry._types import Array, ConfigValue
from simgeometry.kernels.triangle_hits import triangle_fractions
from simgeometry.objects.triangle_mesh import (
    SurfaceHits,
    TriangleMesh,
    _point_major,
)
from simgeometry.shapes import Shape3D, _component_major

#: Points evaluated at once by the Fourier series.
_CHUNK = 16384
#: Newton steps for the hits and the closest points.
_NEWTON_STEPS = 30
#: Most poloidal modes of an offset surface (toroidal: half as many).
MAX_OFFSET_MODES = 64
#: The numbers of modes `FourierSurface.offset` tries, fewest first.
_MODE_STEPS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _circular_torus(major: float, minor: float, nfp: int = 1) -> FourierSurface:
    """A circular torus as a Fourier surface (for the checks)."""
    r_cos = np.zeros((2, 1))
    z_sin = np.zeros((2, 1))
    r_cos[0, 0], r_cos[1, 0], z_sin[1, 0] = major, minor, minor
    return FourierSurface(r_cos, z_sin, nfp)


def _triangulation_problem() -> ConvergenceProblem:
    """Fractions of segments hitting a circular torus, from the triangles only."""
    rng = np.random.default_rng(3)
    n = 400
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    target = np.stack(
        (
            (3.0 + 0.5 * np.cos(theta)) * np.cos(phi),
            (3.0 + 0.5 * np.cos(theta)) * np.sin(phi),
            0.5 * np.sin(theta),
        )
    )
    centre = np.stack((3.0 * np.cos(phi), 3.0 * np.sin(phi), np.zeros(n)))
    start = centre
    end = centre + 2.0 * (target - centre)  # from the tube's centre, through the wall

    def error(spacing: float) -> float:
        points = round(2 * np.pi / spacing)
        torus = _circular_torus(3.0, 0.5)
        torus = FourierSurface(
            torus.r_cos, torus.z_sin, 1, resolution=(points, 6 * points), refine=False
        )
        found = torus.crossings(start, end)
        return float(np.max(np.abs(np.asarray(xp.to_numpy(found.fraction)) - 0.5)))

    return ConvergenceProblem(
        error, steps=[2 * np.pi / 16, 2 * np.pi / 32, 2 * np.pi / 64]
    )


def _as_boundary(boundary) -> dict:
    """ConStellaration's boundary as a dict: a dict, its JSON, or an object."""
    if isinstance(boundary, (str, bytes)):
        boundary = json.loads(boundary)
    keys = ("r_cos", "z_sin", "r_sin", "z_cos", "n_field_periods")
    if not isinstance(boundary, dict):
        boundary = {key: getattr(boundary, key, None) for key in keys}
    missing = [
        key
        for key in ("r_cos", "z_sin", "n_field_periods")
        if boundary.get(key) is None
    ]
    if missing:
        raise ValueError(f"the boundary lacks {missing}")
    return boundary


def _fourier_fit(values: np.ndarray, big_m: int, big_n: int):
    """Coefficients of ``cos`` and ``sin(m theta - n zeta)`` of a periodic grid.

    ``values`` is ``(n_theta, n_zeta)`` on uniform grids of ``theta`` and ``zeta``
    (``nfp phi``) over ``[0, 2 pi)``, with more than ``2M`` and ``2N`` points. Returns
    the ``(M + 1, 2N + 1)`` cosine and sine coefficients (``m = 0`` only for
    ``n >= 0``, as in VMEC's layout) and the largest misfit at the grid points.
    """
    n_theta, n_zeta = values.shape
    spectrum = np.fft.fft2(values) / values.size
    m = np.arange(big_m + 1)[:, None]
    n = np.arange(-big_n, big_n + 1)[None, :]
    entry = spectrum[m % n_theta, (-n) % n_zeta]
    keep = (m > 0) | (n >= 0)
    cos = np.where(keep, 2.0 * entry.real, 0.0)
    sin = np.where(keep & ((m > 0) | (n > 0)), -2.0 * entry.imag, 0.0)
    cos[0, big_n] = entry[0, big_n].real
    # the misfit: the grid's values against the truncated spectrum
    k = np.rint(np.fft.fftfreq(n_theta) * n_theta)[:, None]
    l_ = np.rint(np.fft.fftfreq(n_zeta) * n_zeta)[None, :]
    truncated = np.where((np.abs(k) <= big_m) & (np.abs(l_) <= big_n), spectrum, 0.0)
    fitted = np.fft.ifft2(truncated * values.size).real
    return cos, sin, float(np.max(np.abs(fitted - values)))


@method(
    equations={
        "surface": r"X(\theta, \varphi) = (R\cos\varphi, R\sin\varphi, Z), \quad R = \sum_{m,n} r^c_{mn}\cos(m\theta - N_\mathrm{fp} n\varphi) + r^s_{mn}\sin(\cdot), \quad Z = \sum_{m,n} z^s_{mn}\sin(m\theta - N_\mathrm{fp} n\varphi) + z^c_{mn}\cos(\cdot)",
        "normal": r"n = \sigma\,\frac{X_\theta \times X_\varphi}{|X_\theta \times X_\varphi|}, \qquad \sigma = \operatorname{sign} \oint X \cdot (X_\theta \times X_\varphi)\, d\theta\, d\varphi",
        "hit": r"a + t d = X(\theta, \varphi): \quad \begin{pmatrix} d & -X_\theta & -X_\varphi \end{pmatrix} \delta = -(a + t d - X), \quad (t, \theta, \varphi) \mathrel{+}= \delta",
        "closest": r"\min_{\theta, \varphi} \tfrac12 |X - p|^2: \quad H \delta = -\bigl((X - p)\cdot X_\theta, (X - p)\cdot X_\varphi\bigr), \quad H_{ij} = X_i \cdot X_j + (X - p)\cdot X_{ij}",
        "offset": r"X' = X + \delta\, n, \qquad (R', Z') \text{ at } \varphi' = \varphi_j \text{ refitted by least squares}",
        "volume": r"V = \tfrac13 \oint X \cdot n\, dA, \qquad A = \oint |X_\theta \times X_\varphi|\, d\theta\, d\varphi \quad \text{(trapezoidal rule)}",
    },
    order=2,
    assumptions=(
        r"VMEC's and ConStellaration's `SurfaceRZFourier`: $m = 0 \dots M$, $n = -N \dots N$, $\varphi$ the cylindrical angle; each cross-section at constant $\varphi$ a simple closed curve",
        r"membership and the hit search on the triangulation of `resolution` $(n_\theta, n_\varphi)$ points per poloidal turn and per field period, exact for the triangles (`TriangleMesh`)",
        r"`refine`: hits moved onto the surface by Newton steps on $(t, \theta, \varphi)$ from the triangle's angles, to $10^{-12}$ of the surface's size; kept on the triangle if a step fails or leaves $(10^{-12}, 1]$",
        "whether a segment hits is decided by the triangles: grazing segments within `triangulation_error` of the surface may differ from the exact surface",
        r"offset: the normal offset is a surface at distance $|\delta|$ while $|\delta|$ is below the smallest radius of curvature of the concave parts; refitted with $2M$ poloidal and $2N$ toroidal modes by default",
        r"the order is that of the hit fractions from the triangles alone in the angular spacing",
    ),
    components=("simgeometry.objects.triangle_mesh.TriangleMesh",),
    problem=_triangulation_problem,
)
class FourierSurface(Shape3D):
    """A closed toroidal surface as a Fourier series, with ``nfp`` field periods.

    Example
    -------
    A wall 10 cm outside a ConStellaration plasma boundary, and where beam lines meet
    it::

        plasma = FourierSurface.from_boundary(json.loads(row["boundary.json"]))
        wall = plasma.offset(0.10)
        hits = wall.first_hit(starts, ends)  # (..., 3) arrays
    """

    def _series(self, theta: np.ndarray, phi: np.ndarray, order: int) -> dict:
        """R, Z and their derivatives up to ``order`` at flat angles ``(n,)``.

        With ``E = exp(i (m theta + k phi))``, ``k = -nfp n``, a series
        ``c.A + s.B`` is ``Re(E.(A - iB))`` and each derivative in the angle
        multiplies the coefficients by ``i``: one complex product for all of them.
        """
        names = ["", "t", "p"] + (["tt", "tp", "pp"] if order > 1 else [])
        m, k = self._m, self._k
        factors = {"": 1.0, "t": 1j * m, "p": 1j * k}
        factors |= {"tt": -m * m, "tp": -m * k, "pp": -k * k}
        r_coef = self._rc - 1j * self._rs
        z_coef = self._zc - 1j * self._zs
        keys = [f"{q}{n}" for q in "RZ" for n in names]
        matrix = np.stack(
            [factors[n] * r_coef for n in names] + [factors[n] * z_coef for n in names],
            axis=1,
        )
        values = np.empty((theta.shape[0], len(keys)))
        m_values = np.arange(self.r_cos.shape[0])
        k_values = -self.nfp * (np.arange(self.r_cos.shape[1]) - self.n_toroidal)
        # sum over n first, (2N + 1) by (M + 1) * outputs, then over m
        shape = (m_values.shape[0], k_values.shape[0], len(keys))
        matrix = matrix.reshape(shape).transpose(1, 0, 2).reshape(shape[1], -1)
        for lo in range(0, theta.shape[0], _CHUNK):
            sl = slice(lo, lo + _CHUNK)
            poloidal = np.exp(1j * theta[sl, None] * m_values)
            toroidal = np.exp(1j * phi[sl, None] * k_values)
            partial = (toroidal @ matrix).reshape(-1, shape[0], shape[2])
            values[sl] = np.einsum("pm,pmq->pq", poloidal, partial).real
        return {key: values[:, i] for i, key in enumerate(keys)}

    def _frame(self, theta: np.ndarray, phi: np.ndarray, order: int = 1) -> tuple:
        """Points ``(n, 3)`` and their derivatives in theta and phi (and second ones)."""
        f = self._series(theta, phi, order)
        cos, sin = np.cos(phi), np.sin(phi)
        r, rt, rp = f["R"], f["Rt"], f["Rp"]
        x = np.stack((r * cos, r * sin, f["Z"]), axis=1)
        xt = np.stack((rt * cos, rt * sin, f["Zt"]), axis=1)
        xp_ = np.stack((rp * cos - r * sin, rp * sin + r * cos, f["Zp"]), axis=1)
        if order < 2:
            return x, xt, xp_
        rtt, rtp, rpp = f["Rtt"], f["Rtp"], f["Rpp"]
        xtt = np.stack((rtt * cos, rtt * sin, f["Ztt"]), axis=1)
        xtp = np.stack((rtp * cos - rt * sin, rtp * sin + rt * cos, f["Ztp"]), axis=1)
        xpp = np.stack(
            (
                rpp * cos - 2 * rp * sin - r * cos,
                rpp * sin + 2 * rp * cos - r * sin,
                f["Zpp"],
            ),
            axis=1,
        )
        return x, xt, xp_, xtt, xtp, xpp

    def __init__(
        self,
        r_cos: Array,
        z_sin: Array,
        n_field_periods: int,
        *,
        r_sin: Array | None = None,
        z_cos: Array | None = None,
        resolution: tuple[int, int] = (64, 32),
        refine: bool = True,
        cells: int | tuple[int, int, int] | None = None,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a Fourier surface.

        Parameters:
        r_cos (array): ``(M + 1, 2N + 1)`` cosine coefficients of R.
        z_sin (array): ``(M + 1, 2N + 1)`` sine coefficients of Z.
        n_field_periods (int): Number of field periods ``nfp``.
        r_sin (array, optional): Sine coefficients of R (not stellarator-symmetric).
        z_cos (array, optional): Cosine coefficients of Z (not stellarator-symmetric).
        resolution (tuple, optional): Points of the triangulation per poloidal turn
            and per field period.
        refine (bool, optional): Move hits onto the exact surface by Newton steps.
        cells (int or tuple, optional): Cells of the triangulation's grid per axis.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the inside.
        """
        self.r_cos = np.asarray(r_cos, dtype=float)
        self.z_sin = np.asarray(z_sin, dtype=float)
        if (
            self.r_cos.ndim != 2
            or self.r_cos.shape[1] % 2 != 1
            or self.z_sin.shape != self.r_cos.shape
        ):
            raise ValueError(
                "r_cos and z_sin must be (M + 1, 2N + 1) arrays of one shape, "
                f"got {self.r_cos.shape} and {self.z_sin.shape}"
            )
        self.r_sin = None if r_sin is None else np.asarray(r_sin, dtype=float)
        self.z_cos = None if z_cos is None else np.asarray(z_cos, dtype=float)
        for name, value in (("r_sin", self.r_sin), ("z_cos", self.z_cos)):
            if value is not None and value.shape != self.r_cos.shape:
                raise ValueError(f"{name} must have the shape of r_cos")
        self.nfp = int(n_field_periods)
        if self.nfp < 1:
            raise ValueError("n_field_periods must be positive")
        n_theta, n_phi = (int(v) for v in resolution)
        if n_theta < 3 or n_phi < 1 or n_theta * n_phi * self.nfp < 9:
            raise ValueError("resolution is too coarse for a closed triangulation")
        self.resolution = (n_theta, n_phi)
        self.refine = bool(refine)
        self._cells = cells
        self._bc = bc
        self._inverted = bool(inverted)
        m = np.arange(self.r_cos.shape[0])
        n = np.arange(self.r_cos.shape[1]) - self.n_toroidal
        self._m = np.repeat(m, n.shape[0]).astype(float)
        self._k = np.tile(-self.nfp * n, m.shape[0]).astype(float)  # d(angle)/d(phi)
        zeros = np.zeros(self._m.shape[0])  # the m-major order of r_cos.ravel()
        self._rc = self.r_cos.ravel()
        self._rs = zeros if self.r_sin is None else self.r_sin.ravel()
        self._zs = self.z_sin.ravel()
        self._zc = zeros if self.z_cos is None else self.z_cos.ravel()
        # orientation, volume, area and box from one field period (the rule is exact
        # for the products of three series on more than 3M and 3 (2N + 1) points; at
        # least 64 by 32 points keep the box's sampling error below its padding)
        n_theta = max(3 * self.r_cos.shape[0] + 16, 64)
        n_phi = max(3 * self.r_cos.shape[1] + 8, 32)
        theta = np.repeat(2 * np.pi * np.arange(n_theta) / n_theta, n_phi)
        phi = np.tile(2 * np.pi * np.arange(n_phi) / (n_phi * self.nfp), n_theta)
        x, xt, xp_ = self._frame(theta, phi)
        cross = np.cross(xt, xp_)
        weight = (2 * np.pi) ** 2 / theta.size  # nfp periods of 2 pi / nfp each
        signed = float(np.sum(x * cross) / 3.0 * weight)
        if not abs(signed) > 0.0:
            raise ValueError("the surface encloses no volume")
        self._orientation = 1.0 if signed > 0 else -1.0
        self._volume = abs(signed)
        self._area = float(np.sum(np.linalg.norm(cross, axis=1)) * weight)
        r = np.hypot(x[:, 0], x[:, 1])
        self._scale = float(max(r.max(), np.abs(x[:, 2]).max()))
        pad = 1e-3 * self._scale
        r_max = float(r.max()) + pad
        lower = (-r_max, -r_max, float(x[:, 2].min()) - pad)
        upper = (r_max, r_max, float(x[:, 2].max()) + pad)
        self._mesh: TriangleMesh | None = None
        self._face_angles: np.ndarray | None = None
        self._triangulation_error: float | None = None
        self.offset_residual: float | None = None
        super().__init__(lower_bound=lower, upper_bound=upper)

    @property
    def n_poloidal(self) -> int:
        """The largest poloidal mode number ``M``."""
        return self.r_cos.shape[0] - 1

    @property
    def n_toroidal(self) -> int:
        """The largest toroidal mode number ``N``."""
        return (self.r_cos.shape[1] - 1) // 2

    @property
    def stellarator_symmetric(self) -> bool:
        """Whether the surface has only ``r_cos`` and ``z_sin`` terms."""
        return self.r_sin is None and self.z_cos is None

    @classmethod
    def from_boundary(cls, boundary: Any, **kwargs: Any) -> FourierSurface:
        """Return the surface of ConStellaration's ``boundary.json``.

        ``boundary`` is the dict, its JSON string, or an object with the attributes
        ``r_cos``, ``z_sin``, ``r_sin``, ``z_cos`` and ``n_field_periods`` (such as
        ``constellaration``'s ``SurfaceRZFourier``). Other arguments go to
        `FourierSurface`.
        """
        boundary = _as_boundary(boundary)
        return cls(
            boundary["r_cos"],
            boundary["z_sin"],
            boundary["n_field_periods"],
            r_sin=boundary.get("r_sin"),
            z_cos=boundary.get("z_cos"),
            **kwargs,
        )

    def to_boundary(self) -> dict:
        """Return the coefficients in ConStellaration's ``boundary.json`` layout."""
        return {
            "r_cos": self.r_cos.tolist(),
            "z_sin": self.z_sin.tolist(),
            "r_sin": None if self.r_sin is None else self.r_sin.tolist(),
            "z_cos": None if self.z_cos is None else self.z_cos.tolist(),
            "n_field_periods": self.nfp,
            "is_stellarator_symmetric": self.stellarator_symmetric,
        }

    def copy(self) -> FourierSurface:
        """Create a copy of the surface."""
        return FourierSurface(
            self.r_cos,
            self.z_sin,
            self.nfp,
            r_sin=self.r_sin,
            z_cos=self.z_cos,
            resolution=self.resolution,
            refine=self.refine,
            cells=self._cells,
            bc=self._bc,
            inverted=self._inverted,
        )

    def _grid(self, n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
        """Flat angles of a periodic grid: ``n_theta`` by ``n_phi`` per field period."""
        theta = 2 * np.pi * np.arange(n_theta) / n_theta
        phi = 2 * np.pi * np.arange(n_phi * self.nfp) / (n_phi * self.nfp)
        theta, phi = np.meshgrid(theta, phi, indexing="ij")
        return theta.ravel(), phi.ravel()

    def rz(self, theta: Array, phi: Array) -> tuple[np.ndarray, np.ndarray]:
        """Return ``R`` and ``Z`` at broadcast angles ``theta`` and ``phi``."""
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
        f = self._series(theta.ravel(), phi.ravel(), 1)
        return f["R"].reshape(theta.shape), f["Z"].reshape(theta.shape)

    def point(self, theta: Array, phi: Array) -> np.ndarray:
        """Return the Cartesian points ``(..., 3)`` at broadcast angles."""
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
        x = self._frame(theta.ravel(), phi.ravel())[0]
        return x.reshape(*theta.shape, 3)

    def _unit_normal(self, xt: np.ndarray, xp_: np.ndarray) -> np.ndarray:
        cross = self._orientation * np.cross(xt, xp_)
        length = np.linalg.norm(cross, axis=-1, keepdims=True)
        return cross / np.where(length > 0, length, np.nan)

    def normal(self, theta: Array, phi: Array) -> np.ndarray:
        """Return the outward unit normals ``(..., 3)`` at broadcast angles."""
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
        _, xt, xp_ = self._frame(theta.ravel(), phi.ravel())
        return self._unit_normal(xt, xp_).reshape(*theta.shape, 3)

    def triangulate(
        self, n_theta: int, n_phi: int, *, cells=None
    ) -> tuple[TriangleMesh, np.ndarray]:
        """Return a closed mesh of the surface and the angles of its faces' corners.

        The vertices are a grid of ``n_theta`` poloidal angles by ``n_phi`` toroidal
        angles per field period, each grid quad two triangles oriented outward.
        The angles are ``(F, 3, 2)``, ``(theta, phi)`` per corner, continuous within
        a face (not wrapped to ``[0, 2 pi)``).
        """
        n_total = n_phi * self.nfp
        theta, phi = self._grid(n_theta, n_phi)
        vertices = self._frame(theta, phi)[0]
        i, j = np.meshgrid(np.arange(n_theta), np.arange(n_total), indexing="ij")
        i, j = i.ravel(), j.ravel()
        ip, jp = (i + 1) % n_theta, (j + 1) % n_total

        def index(a, b):
            return a * n_total + b

        dt, dp = 2 * np.pi / n_theta, 2 * np.pi / n_total
        t0, p0 = i * dt, j * dp
        corners = [
            (index(i, j), t0, p0),
            (index(ip, j), t0 + dt, p0),
            (index(ip, jp), t0 + dt, p0 + dp),
            (index(i, jp), t0, p0 + dp),
        ]
        order = [(0, 1, 2), (0, 2, 3)]
        if self._orientation < 0:
            order = [(0, 2, 1), (0, 3, 2)]
        faces = np.concatenate(
            [np.stack([corners[c][0] for c in tri], axis=1) for tri in order]
        )
        angles = np.concatenate(
            [
                np.stack([np.stack(corners[c][1:], axis=1) for c in tri], axis=1)
                for tri in order
            ]
        )
        mesh = TriangleMesh(vertices, faces, closed=True, cells=cells)
        return mesh, angles

    @property
    def mesh(self) -> TriangleMesh:
        """The triangulation at `resolution`, which answers membership and hit searches."""
        if self._mesh is None:
            self._mesh, self._face_angles = self.triangulate(
                *self.resolution, cells=self._cells
            )
        return self._mesh

    @property
    def triangulation_error(self) -> float:
        """The largest distance between the faces and the surface at the same angles.

        Measured at the centroids and the midpoints of the edges of all faces.

        An estimate of how far the triangles deviate from the surface (second
        order in the spacing); membership differs from the exact surface only
        within about this distance of it.
        """
        if self._triangulation_error is None:
            mesh = self.mesh
            corners = mesh.vertices[mesh.faces]  # (F, 3, 3)
            error = 0.0
            # the centroid and the midpoints of the edges, against the surface there
            for weights in ([1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]):
                w = np.asarray(weights, float)[None, :, None] / sum(weights)
                points = np.sum(w * corners, axis=1)
                angles = np.sum(w * self._face_angles, axis=1)
                exact = self._frame(angles[:, 0], angles[:, 1])[0]
                error = max(
                    error, float(np.max(np.linalg.norm(exact - points, axis=1)))
                )
            self._triangulation_error = error
        return self._triangulation_error

    def contains(self, coords: tuple[float, float, float]) -> bool:
        """Return whether the point is inside (see `contains_points`)."""
        point = np.asarray(coords, dtype=float).reshape(3, 1)
        return bool(np.asarray(xp.to_numpy(self.contains_points(point)))[0])

    def contains_points(self, points: Array) -> Array:
        """Return a mask of the ``(3, n)`` points inside the surface's triangulation.

        Exact for the triangles; within `triangulation_error` of the surface the
        result may differ from the exact surface's.
        """
        inside = self.mesh.contains_points(_component_major(points, 3))
        return ~inside if self.inverted else inside

    def inside_at(self, points: Array) -> Array:
        """Return whether the points ``(..., 3)`` are inside, as ``(...)``."""
        points = _point_major(points, "points")
        flat = points.reshape(-1, 3).T
        return xp.reshape(self.contains_points(flat), points.shape[:-1])

    @sp.profile("simgeometry.fourier_surface_refine", tags=_PROFILE_TAGS)
    def _refined(self, a, d, fraction, face, t_min):
        """Newton steps from the triangles' hits onto the surface.

        Returns the fractions, angles and normals ``(3, n)`` of the hits, and the
        mask of those refined (the others keep the triangle's values).
        """
        mesh = self.mesh
        hit = face >= 0
        rows = np.flatnonzero(hit)
        f = face[rows]
        a_h, d_h = a[:, rows], d[:, rows]
        _, u, v = triangle_fractions(mesh.vertices, mesh.faces, f, a_h, d_h)
        corners = self._face_angles[f]  # (h, 3, 2)
        weights = np.stack((1 - u - v, u, v), axis=1)[:, :, None]
        angles = np.sum(weights * corners, axis=1)
        t = fraction[rows].copy()
        theta, phi = angles[:, 0].copy(), angles[:, 1].copy()
        tolerance = 1e-12 * self._scale
        done = np.zeros(rows.shape[0], dtype=bool)
        normal = mesh.face_normals[f].copy()  # (h, 3); the surface's where refined
        if self.refine and rows.shape[0]:
            active = np.arange(rows.shape[0])
            for _ in range(_NEWTON_STEPS):
                x, xt, xp_ = self._frame(theta[active], phi[active])
                g = (a_h[:, active] + t[active] * d_h[:, active]).T - x
                small = np.linalg.norm(g, axis=1) <= tolerance
                done[active[small]] = True
                normal[active[small]] = self._unit_normal(xt[small], xp_[small])
                active, g, xt, xp_ = active[~small], g[~small], xt[~small], xp_[~small]
                if not active.shape[0]:
                    break
                jac = np.stack((d_h[:, active].T, -xt, -xp_), axis=2)
                singular = np.abs(np.linalg.det(jac)) < 1e-300
                jac[singular] = np.eye(3)
                step = np.linalg.solve(jac, -g[:, :, None])[:, :, 0]
                step[singular] = 0.0
                step[:, 1:] = np.clip(step[:, 1:], -0.2, 0.2)
                t[active] += step[:, 0]
                theta[active] += step[:, 1]
                phi[active] += step[:, 2]
            # a refined hit must stay on the segment and near the triangle's, across
            # the surface (along a grazing segment it may move much further)
            across = np.abs(np.sum(normal.T * d_h, axis=0))
            moved = np.abs(t - fraction[rows]) * across
            near = moved <= 10.0 * self.triangulation_error + tolerance
            on_start = done & near & (t <= t_min)
            done &= (t > t_min) & (t <= 1.0) & near
        t = np.where(done, t, fraction[rows])
        theta = np.where(done, theta, angles[:, 0])
        phi = np.where(done, phi, angles[:, 1])
        normal = np.where(done[:, None], normal, mesh.face_normals[f]).T
        n = a.shape[1]
        out_t = fraction.copy()
        out_t[rows] = t
        out_theta, out_phi = np.full(n, np.nan), np.full(n, np.nan)
        out_theta[rows] = np.mod(theta, 2 * np.pi)
        out_phi[rows] = np.mod(phi, 2 * np.pi)
        out_normal = np.full((3, n), np.nan)
        out_normal[:, rows] = normal
        refined = np.zeros(n, dtype=bool)
        refined[rows] = done
        at_start = np.zeros(n, dtype=bool)
        if self.refine and rows.shape[0]:
            at_start[rows] = on_start
        return out_t, out_theta, out_phi, out_normal, refined, at_start

    def _search(self, a: np.ndarray, d: np.ndarray, t_min: float):
        """First hits of ``a + t d``: fractions, faces, angles, normals and refined.

        A segment that starts on the surface (a hit refined back to its start) is
        searched again past the triangle it met there, so a line can be followed
        from one hit to the next.
        """
        fraction, face = self.mesh._hits(a, a + d, t_min)
        t, theta, phi, normal, refined, at_start = self._refined(
            a, d, fraction, face, t_min
        )
        for _ in range(3):
            rows = np.flatnonzero(at_start)
            if not rows.shape[0]:
                break
            skip = fraction[rows] + 1e-9
            a_rest = a[:, rows] + skip * d[:, rows]
            d_rest = (1.0 - skip) * d[:, rows]
            f_rest, face_rest = self.mesh._hits(a_rest, a_rest + d_rest, t_min)
            t_rest, th_rest, ph_rest, n_rest, r_rest, s_rest = self._refined(
                a_rest, d_rest, f_rest, face_rest, t_min
            )
            fraction[rows] = skip + (1.0 - skip) * f_rest
            t[rows] = skip + (1.0 - skip) * t_rest
            face[rows] = face_rest
            theta[rows], phi[rows] = th_rest, ph_rest
            normal[:, rows] = n_rest
            refined[rows] = r_rest
            at_start[:] = False
            at_start[rows] = s_rest
        return t, face, theta, phi, normal, refined

    def _first_crossing(self, a: Array, d: Array, t_min: float):
        """Fraction (inf for none), normal and exactness of the first crossings.

        The interface of `simgeometry.crossings.crossings`; exact when every hit
        was refined onto the surface.
        """
        a_host = np.asarray(xp.to_numpy(a), dtype=float)
        d_host = np.asarray(xp.to_numpy(d), dtype=float)
        t, face, _, _, normal, refined = self._search(a_host, d_host, t_min)
        hit = face >= 0
        exact = bool(np.all(refined[hit])) if self.refine else False
        return xp.asarray(np.where(hit, t, np.inf)), xp.asarray(normal), exact

    def first_hit(self, starts: Array, ends: Array) -> SurfaceHits:
        """Return the first hit of each segment ``starts[...] -> ends[...]``, ``(..., 3)``.

        Point-major `SurfaceHits` with the face of the triangulation, the angles
        ``theta`` and ``phi`` of the hit and whether it was ``refined`` onto the
        exact surface. A segment starting at a refined hit (on the surface) finds
        the next one, so ``first_hit(hits.point, ends)`` follows a line through
        the surface.
        """
        from simgeometry.crossings import START_TOLERANCE

        a_full = _point_major(starts, "starts")
        b_full = _point_major(ends, "ends")
        if a_full.shape != b_full.shape:
            raise ValueError(
                f"starts {a_full.shape} and ends {b_full.shape} must have the same shape"
            )
        shape = a_full.shape[:-1]
        a = a_full.reshape(-1, 3).T
        b = b_full.reshape(-1, 3).T
        t, face, theta, phi, normal, refined = self._search(a, b - a, START_TOLERANCE)
        hits = self.mesh._result(
            a, b, t, face, shape, normal=normal, theta=theta, phi=phi, refined=refined
        )
        if self.inverted:  # _result flips the normal of the (never inverted) mesh only
            hits = replace(hits, normal=-hits.normal)
        return hits

    def closest(self, points: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the angles ``theta``, ``phi`` and the distance of the closest points.

        For points ``(..., 3)``: Newton steps on the squared distance from the
        nearest vertex of the triangulation (Gauss–Newton where the Hessian is not
        positive). The cost grows with the number of vertices; it is meant for
        checks, not for every particle of a run.
        """
        points = _point_major(points, "points")
        shape = points.shape[:-1]
        p = points.reshape(-1, 3)
        mesh = self.mesh
        n_theta, n_phi = self.resolution
        theta_grid, phi_grid = self._grid(n_theta, n_phi)
        nearest = np.empty(p.shape[0], dtype=np.int64)
        size = max(1, 4_000_000 // mesh.vertices.shape[0])
        for lo in range(0, p.shape[0], size):
            delta = p[lo : lo + size, None, :] - mesh.vertices[None, :, :]
            nearest[lo : lo + size] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        theta, phi = theta_grid[nearest].copy(), phi_grid[nearest].copy()
        for _ in range(_NEWTON_STEPS):
            x, xt, xp_, xtt, xtp, xpp = self._frame(theta, phi, order=2)
            r = x - p
            g = np.stack((np.sum(r * xt, 1), np.sum(r * xp_, 1)), axis=1)
            gn = np.stack(
                (
                    np.stack((np.sum(xt * xt, 1), np.sum(xt * xp_, 1)), axis=1),
                    np.stack((np.sum(xt * xp_, 1), np.sum(xp_ * xp_, 1)), axis=1),
                ),
                axis=1,
            )
            curvature = np.stack(
                (
                    np.stack((np.sum(r * xtt, 1), np.sum(r * xtp, 1)), axis=1),
                    np.stack((np.sum(r * xtp, 1), np.sum(r * xpp, 1)), axis=1),
                ),
                axis=1,
            )
            hessian = gn + curvature
            positive = (np.linalg.det(hessian) > 0) & (hessian[:, 0, 0] > 0)
            hessian = np.where(positive[:, None, None], hessian, gn)
            step = np.linalg.solve(hessian, -g[:, :, None])[:, :, 0]
            step = np.clip(step, -0.2, 0.2)
            theta += step[:, 0]
            phi += step[:, 1]
            if np.max(np.abs(step)) < 1e-14:
                break
        distance = np.linalg.norm(self._frame(theta, phi)[0] - p, axis=1)
        return (
            np.mod(theta, 2 * np.pi).reshape(shape),
            np.mod(phi, 2 * np.pi).reshape(shape),
            distance.reshape(shape),
        )

    def offset(
        self,
        distance: float,
        *,
        tolerance: float | None = None,
        poloidal_modes: int | None = None,
        toroidal_modes: int | None = None,
        **kwargs: Any,
    ) -> FourierSurface:
        """Return the surface moved by ``distance`` along its outward normal.

        Every point of a fine grid is moved, ``X + distance n``; at the grid's
        toroidal angles the moved surface's ``R`` and ``Z`` are found by fixed-point
        steps on the toroidal angle of the source point and transformed by an FFT.
        The series is cut at the fewest modes whose largest misfit on the grid is
        below ``tolerance`` (default: a hundredth of ``|distance|``), with up to
        `MAX_OFFSET_MODES` poloidal and half as many toroidal modes, or at
        ``poloidal_modes`` and ``toroidal_modes`` if given. The misfit is kept as
        `offset_residual`; a warning says when the tolerance is not reached.

        A negative distance moves inward. The result is the surface at the given
        distance as long as ``|distance|`` is below the smallest radius of
        curvature of the concave parts (`closest` checks it). Other arguments go
        to `FourierSurface` (default: this surface's `resolution` and `refine`).
        """
        cap_m = MAX_OFFSET_MODES if poloidal_modes is None else int(poloidal_modes)
        cap_n = MAX_OFFSET_MODES // 2 if toroidal_modes is None else int(toroidal_modes)
        if cap_m < 1 or cap_n < 0:
            raise ValueError("poloidal_modes must be >= 1 and toroidal_modes >= 0")
        if tolerance is None:
            tolerance = 1e-2 * abs(distance) if distance else 1e-12 * self._scale
        n_theta, n_phi = 3 * (cap_m + 1) + 1, 3 * (cap_n + 1) + 1
        theta = 2 * np.pi * np.arange(n_theta) / n_theta
        target = 2 * np.pi * np.arange(n_phi) / (n_phi * self.nfp)
        theta, target = (a.ravel() for a in np.meshgrid(theta, target, indexing="ij"))
        source = target.copy()
        for _ in range(100):
            x, xt, xp_ = self._frame(theta, source)
            moved = x + distance * self._unit_normal(xt, xp_)
            angle = np.arctan2(moved[:, 1], moved[:, 0])
            change = np.angle(np.exp(1j * (angle - target)))
            source -= change
            if np.max(np.abs(change)) < 1e-13:
                break
        x, xt, xp_ = self._frame(theta, source)
        moved = x + distance * self._unit_normal(xt, xp_)
        r_new = np.hypot(moved[:, 0], moved[:, 1]).reshape(n_theta, n_phi)
        z_new = moved[:, 2].reshape(n_theta, n_phi)
        # the fewest modes within the tolerance, from the same samples
        fixed = poloidal_modes is not None and toroidal_modes is not None
        choices = (
            [(cap_m, cap_n)]
            if fixed
            else sorted(
                {
                    (min(m_, cap_m), min(n_, cap_n))
                    for m_ in _MODE_STEPS
                    for n_ in (0, *_MODE_STEPS)
                    if n_ <= m_
                }
                | {(cap_m, cap_n)},
                key=lambda mn: ((mn[0] + 1) * (2 * mn[1] + 1), mn),
            )
        )
        for big_m, big_n in choices:
            r_cos, r_sin, r_residual = _fourier_fit(r_new, big_m, big_n)
            z_cos, z_sin, z_residual = _fourier_fit(z_new, big_m, big_n)
            if max(r_residual, z_residual) <= tolerance:
                break
        else:
            if not fixed:
                warnings.warn(
                    f"the offset surface misses the tolerance {tolerance:.3g} by "
                    f"{max(r_residual, z_residual):.3g} with {cap_m} poloidal and "
                    f"{cap_n} toroidal modes",
                    stacklevel=2,
                )
        symmetric = self.stellarator_symmetric
        options = {"resolution": self.resolution, "refine": self.refine} | kwargs
        surface = FourierSurface(
            r_cos,
            z_sin,
            self.nfp,
            r_sin=None if symmetric else r_sin,
            z_cos=None if symmetric else z_cos,
            **options,
        )
        surface.offset_residual = max(r_residual, z_residual)
        return surface

    def volume(self) -> float:
        """Return the enclosed volume (trapezoidal rule on the exact surface)."""
        return self._volume

    def surface_area(self) -> float:
        """Return the area of the surface (trapezoidal rule on the exact surface)."""
        return self._area

    def to_mesh(self, **kwargs: Any):
        """Return the triangulation as a `SurfaceMesh` (``resolution`` and ``bounds`` are ignored)."""
        return self.mesh.to_mesh()
