"""Surfaces given as triangle meshes: a first wall from CAD, or a sampled surface.

`TriangleMesh` takes vertices ``(V, 3)`` and faces ``(F, 3)``. A closed mesh (every
edge shared by exactly two faces, consistently oriented) encloses a volume: it has an
inside test and outward normals. An open one (a set of wall panels with gaps) is a
surface without an inside, like a `Polyline` in 2D, and still stops segments.

Queries go through a uniform grid of cells listing the faces that overlap them
(`simgeometry._triangle_grid`) and two compiled kernels: ``mesh_first_hits`` walks
each segment through the cells and tests their triangles (Möller–Trumbore), and
``mesh_inside`` compares a point with the nearest node of its cell, whose state is
known. Both are exact for the triangles: no sampling along the segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from methodcard import ConvergenceProblem, method

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry._triangle_grid import TriangleGrid, build_triangle_grid
from simgeometry._types import Array, ConfigValue
from simgeometry.shapes import Shape3D, _component_major


@dataclass(frozen=True)
class SurfaceHits:
    """First hits of segments with a surface, point-major: ``(...)`` and ``(..., 3)``.

    ``hit``, ``fraction`` (of the segment, in ``(0, 1]``), ``distance`` (from the
    start), ``face`` (the triangle hit, -1 for none) and, for a `FourierSurface`,
    ``theta`` and ``phi`` (the surface's angles at the hit) have the segments'
    shape ``(...)``; ``point`` and ``normal`` (unit, out of the region where
    ``contains`` is true; facing the segment for an open mesh) are ``(..., 3)``.
    Where ``hit`` is false the floats are NaN. ``refined`` (a `FourierSurface`'s
    hits) is true where the hit was moved onto the exact surface by Newton steps.
    """

    hit: Any
    fraction: Any
    distance: Any
    point: Any
    normal: Any
    face: Any
    theta: Any = None
    phi: Any = None
    refined: Any = None


def _point_major(points: Array, name: str) -> np.ndarray:
    """``points`` as a host float array with a last axis of length 3."""
    points = np.asarray(xp.to_numpy(xp.asarray(points, dtype=float)), dtype=float)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {points.shape}")
    return points


def _c_contiguous(array: np.ndarray, dtype=float) -> np.ndarray:
    """A C-ordered copy of ``array`` (Pyccel checks the strides)."""
    array = np.asarray(array, dtype=dtype)
    return np.array(array, dtype=dtype, order="C", copy=True)


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Triangle soup ``(3F, 3)`` vertices and ``(F, 3)`` faces of an STL file."""
    data = path.read_bytes()
    if len(data) >= 84:
        count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
        if 84 + 50 * count == len(data):
            record = np.dtype(
                [("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")]
            )
            facets = np.frombuffer(data, dtype=record, count=count, offset=84)
            corners = facets["corners"].astype(float).reshape(-1, 3)
            return corners, np.arange(corners.shape[0]).reshape(-1, 3)
    words = data.decode("ascii", errors="replace").split()
    values = [
        float(words[i + j])
        for i, word in enumerate(words)
        if word == "vertex"
        for j in (1, 2, 3)
    ]
    corners = np.asarray(values, dtype=float).reshape(-1, 3)
    return corners, np.arange(corners.shape[0]).reshape(-1, 3)


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles (polygons fanned) of a Wavefront OBJ file."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(v) for v in parts[1:4]])
        elif parts[0] == "f":
            corners = [int(p.split("/")[0]) for p in parts[1:]]
            corners = [c - 1 if c > 0 else len(vertices) + c for c in corners]
            faces.extend(
                [corners[0], corners[i], corners[i + 1]]
                for i in range(1, len(corners) - 1)
            )
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _weld(vertices: np.ndarray, faces: np.ndarray, tolerance: float):
    """Merge vertices closer than ``tolerance`` (on a lattice of that spacing)."""
    if tolerance <= 0.0 or vertices.shape[0] == 0:
        return vertices, faces
    keys = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return vertices[first], inverse.reshape(-1)[faces]


def _sphere_problem() -> ConvergenceProblem:
    """The volume inside a UV-sphere mesh of the unit sphere, against 4 pi / 3."""

    def error(spacing: float) -> float:
        n = round(np.pi / spacing)
        mesh = uv_sphere(1.0, n, 2 * n)
        return abs(mesh.volume() - 4.0 * np.pi / 3.0)

    return ConvergenceProblem(error, steps=[np.pi / 8, np.pi / 16, np.pi / 32])


def uv_sphere(radius: float, n_theta: int, n_phi: int) -> TriangleMesh:
    """A closed mesh of a sphere: ``n_theta`` rings of ``n_phi`` vertices and two poles."""
    theta = np.pi * np.arange(1, n_theta) / n_theta
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    ring = np.stack(
        np.broadcast_arrays(
            np.sin(theta)[:, None] * np.cos(phi),
            np.sin(theta)[:, None] * np.sin(phi),
            np.cos(theta)[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    vertices = radius * np.concatenate(([[0.0, 0.0, 1.0]], ring, [[0.0, 0.0, -1.0]]))
    south = vertices.shape[0] - 1
    faces = []
    for j in range(n_phi):
        k = (j + 1) % n_phi
        faces.append([0, 1 + j, 1 + k])
        last = 1 + (n_theta - 2) * n_phi
        faces.append([south, last + k, last + j])
        for i in range(n_theta - 2):
            a, b = 1 + i * n_phi + j, 1 + i * n_phi + k
            c, d = a + n_phi, b + n_phi
            faces.extend([[a, c, d], [a, d, b]])
    return TriangleMesh(vertices, np.asarray(faces))


def _topology(faces: np.ndarray, n_vertices: int) -> tuple[bool, bool]:
    """Whether every edge has two faces, and every directed edge appears once."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = directed[:, 0] * n_vertices + directed[:, 1]
    lo = np.minimum(directed[:, 0], directed[:, 1])
    hi = np.maximum(directed[:, 0], directed[:, 1])
    _, counts = np.unique(lo * n_vertices + hi, return_counts=True)
    watertight = bool(np.all(counts == 2))
    consistent = np.unique(key).shape[0] == key.shape[0]
    return watertight, bool(consistent)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """The volume enclosed by the faces, positive when they are oriented outward."""
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    return float(np.sum(v0 * np.cross(v1, v2)) / 6.0)


@method(
    equations={
        "volume": r"V = \frac16 \sum_f v_0 \cdot (v_1 \times v_2)",
        "normal": r"n_f = \frac{(v_1 - v_0) \times (v_2 - v_0)}{|(v_1 - v_0) \times (v_2 - v_0)|}",
        "inside": r"p \in S \iff s(q) \oplus \bigl(\#\{f : [p, q] \cap f \ne \emptyset\} \bmod 2\bigr), \qquad q \text{ the node of } p\text{'s cell nearest to } p",
        "nodes": r"s(q) = \operatorname{maj}_{k = x, y, z}\Bigl(\#\{f : \text{the grid line along } k \text{ meets } f \text{ before } q\} \bmod 2\Bigr)",
    },
    order=2,
    assumptions=(
        r"closed: every edge in exactly two faces, each directed edge once (consistent orientation); faces are reoriented so that $V > 0$ and $n_f$ points out",
        "open meshes have no inside (`contains` is False everywhere) and their normals face the incoming segment",
        "a uniform grid of about four cells per face over the mesh's box, shifted by irrational fractions of a cell; every cell lists the faces whose box (widened by $10^{-9}$ cells) overlaps it",
        r"first hits walk the segment's cells in order (Amanatides and Woo) and test their faces exactly (Möller–Trumbore), stopping once the nearest hit lies before the cell's exit; hits at $t \le 10^{-12}$ are ignored",
        "exact for the triangles, except for a segment through an edge or vertex exactly, where a crossing may count twice",
        r"the order is that of the volume of a sphere's mesh in the angular spacing",
    ),
    references=(
        "[Amanatides and Woo, Eurographics (1987)](http://www.cse.yorku.ca/~amana/research/grid.pdf)",
        "[Möller and Trumbore, J. Graphics Tools 2, 21 (1997)](https://doi.org/10.1080/10867651.1997.10487468)",
    ),
    components=("simgeometry.kernels.triangle_hits.triangle_fractions",),
    problem=_sphere_problem,
)
class TriangleMesh(Shape3D):
    """A surface made of triangles: closed (a solid) or open (panels).

    Example
    -------
    A wall exported from CAD as STL, and the first wall hit of a few segments::

        wall = TriangleMesh.read("first_wall.stl")
        hits = wall.first_hit(starts, ends)  # (..., 3) arrays
        hits.point, hits.normal, hits.face
    """

    def __init__(
        self,
        vertices: Array,
        faces: Array,
        *,
        closed: bool | None = None,
        cells: int | tuple[int, int, int] | None = None,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a triangle mesh.

        Parameters:
        vertices (array): ``(V, 3)`` vertex coordinates.
        faces (array): ``(F, 3)`` vertex indices of the triangles.
        closed (bool, optional): Whether the mesh encloses a volume; by default
            whether it is watertight and consistently oriented. True raises if it
            is not.
        cells (int or tuple, optional): Cells of the acceleration grid per axis;
            by default about four cells per face.
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the inside.
        """
        vertices = np.asarray(xp.to_numpy(xp.asarray(vertices)), dtype=float)
        faces = np.asarray(xp.to_numpy(xp.asarray(faces)), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
            raise ValueError(f"vertices must be (V, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            raise ValueError(f"faces must be (F, 3), got {faces.shape}")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if faces.min() < 0 or faces.max() >= vertices.shape[0]:
            raise ValueError("faces refer to vertices that do not exist")
        watertight, consistent = _topology(faces, vertices.shape[0])
        if closed is None:
            closed = watertight and consistent
        elif closed and not watertight:
            raise ValueError(
                "the mesh is not closed: some edges are not shared by exactly two faces"
            )
        elif closed and not consistent:
            raise ValueError(
                "the faces of the closed mesh are not consistently oriented"
            )
        self.closed = bool(closed)
        if self.closed and _signed_volume(vertices, faces) < 0.0:
            faces = faces[:, ::-1]
        self._vertices = _c_contiguous(vertices)
        self._faces = _c_contiguous(faces, np.int64)
        self._vertices.flags.writeable = False
        self._faces.flags.writeable = False
        self._bc = bc
        self._inverted = bool(inverted)
        self._cells = cells
        self.grid: TriangleGrid = build_triangle_grid(
            self._vertices, self._faces, cells=cells, closed=self.closed
        )
        cross = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        length = np.linalg.norm(cross, axis=1)
        self._face_areas = 0.5 * length
        self._face_normals = cross / np.where(length > 0, length, np.nan)[:, None]
        super().__init__(
            lower_bound=tuple(float(v) for v in vertices.min(axis=0)),
            upper_bound=tuple(float(v) for v in vertices.max(axis=0)),
        )

    @classmethod
    def read(cls, path: str | Path, *, merge_tolerance: float | None = None, **kwargs):
        """Read a mesh from STL (ASCII or binary), OBJ, NPZ, or any format meshio reads.

        STL stores every triangle on its own, so vertices closer than
        ``merge_tolerance`` (default ``1e-9`` of the mesh's extent) are merged to
        recover the topology; pass 0 to keep them apart. Other arguments go to
        `TriangleMesh`. Formats other than STL, OBJ and NPZ (``vertices`` and
        ``cells``, as written by `SurfaceMesh.save`) need ``meshio``.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".stl":
            vertices, faces = _read_stl(path)
        elif suffix == ".obj":
            vertices, faces = _read_obj(path)
        elif suffix == ".npz":
            with np.load(path) as data:
                vertices, faces = data["vertices"], data["cells"]
        else:
            try:
                meshio = import_module("meshio")  # optional, not a dependency
            except ImportError as exc:
                raise ImportError(
                    f"reading {suffix} files needs meshio (pip install meshio)"
                ) from exc
            data = meshio.read(path)
            vertices = np.asarray(data.points, dtype=float)[:, :3]
            faces = np.concatenate(
                [block.data for block in data.cells if block.type == "triangle"]
            )
        vertices = np.asarray(vertices, dtype=float)
        if merge_tolerance is None:
            extent = np.ptp(vertices, axis=0).max() if vertices.size else 0.0
            merge_tolerance = 1e-9 * float(extent)
        vertices, faces = _weld(
            vertices, np.asarray(faces, dtype=np.int64), merge_tolerance
        )
        return cls(vertices, faces, **kwargs)

    def save(self, path: str | Path) -> Path:
        """Write the mesh as ``.npz``, ``.obj`` or ASCII ``.stl`` (see `SurfaceMesh.save`)."""
        return self.to_mesh().save(path)

    @property
    def vertices(self) -> np.ndarray:
        """The ``(V, 3)`` vertices (read-only)."""
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        """The ``(F, 3)`` faces (read-only); outward-oriented for a closed mesh."""
        return self._faces

    @property
    def n_faces(self) -> int:
        """The number of triangles."""
        return int(self._faces.shape[0])

    @property
    def face_normals(self) -> np.ndarray:
        """Unit normals ``(F, 3)`` of the faces, outward for a closed mesh."""
        return self._face_normals

    @property
    def face_areas(self) -> np.ndarray:
        """Areas ``(F,)`` of the faces."""
        return self._face_areas

    @property
    def edge_length(self) -> float:
        """The mean length of the faces' edges."""
        corners = self._vertices[self._faces]
        edges = corners - np.roll(corners, 1, axis=1)
        return float(np.linalg.norm(edges, axis=2).mean())

    def copy(self) -> TriangleMesh:
        """Create a copy of the mesh (sharing the read-only arrays)."""
        return TriangleMesh(
            self._vertices,
            self._faces,
            closed=self.closed,
            cells=self._cells,
            bc=self._bc,
            inverted=self._inverted,
        )

    def contains(self, coords: tuple[float, float, float]) -> bool:
        """Return whether the point is inside (never for an open mesh)."""
        point = np.asarray(coords, dtype=float).reshape(3, 1)
        return bool(np.asarray(xp.to_numpy(self.contains_points(point)))[0])

    @sp.profile("simgeometry.mesh_inside", tags=_PROFILE_TAGS)
    def contains_points(self, points: Array) -> Array:
        """Return a mask of the ``(3, n)`` points inside the closed mesh.

        Exact for the triangles (see the module docstring); an open mesh contains
        nothing, so the mask is False (True if inverted).
        """
        points = _component_major(points, 3)
        host = _c_contiguous(xp.to_numpy(points))
        inside = np.zeros(host.shape[1], dtype=bool)
        if self.closed and host.shape[1]:
            from simgeometry.kernels.mesh_inside import kernel

            grid = self.grid
            kernel(
                host,
                self._vertices,
                self._faces,
                grid.lower,
                grid.spacing,
                grid.cells,
                grid.cell_start,
                grid.cell_faces,
                grid.node_inside,
                inside,
            )
        if self.inverted:
            inside = ~inside
        return xp.asarray(inside)

    def inside_at(self, points: Array) -> Array:
        """Return whether the points ``(..., 3)`` are inside, as ``(...)``.

        The point-major form of `contains_points`, as freeofcharge's
        ``inside_wall_at`` takes its points.
        """
        points = _point_major(points, "points")
        flat = points.reshape(-1, 3).T
        return xp.reshape(self.contains_points(flat), points.shape[:-1])

    @sp.profile("simgeometry.mesh_first_hits", tags=_PROFILE_TAGS)
    def _hits(self, a: np.ndarray, b: np.ndarray, t_min: float):
        """Fractions ``(n,)`` (2 for none) and faces of the first hits of ``a -> b``."""
        from simgeometry.kernels.mesh_first_hits import kernel

        n = a.shape[1]
        fraction = np.full(n, 2.0)
        face = np.full(n, -1, dtype=np.int64)
        if n:
            grid = self.grid
            kernel(
                _c_contiguous(a),
                _c_contiguous(b),
                self._vertices,
                self._faces,
                grid.lower,
                grid.spacing,
                grid.cells,
                grid.cell_start,
                grid.cell_faces,
                float(t_min),
                fraction,
                face,
            )
        return fraction, face

    def _oriented_normals(self, face: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Normals ``(3, n)`` of the faces hit: outward, or facing ``d`` if open."""
        normal = self._face_normals[np.maximum(face, 0)].T
        if not self.closed:
            facing = np.sum(normal * d, axis=0)
            normal = np.where(facing > 0, -normal, normal)
        return normal

    def _first_crossing(self, a: Array, d: Array, t_min: float):
        """Fraction (inf for none), normal and exactness of the first crossings.

        The interface of `simgeometry.crossings.crossings` for component-major
        ``(3, n)`` starts ``a`` and steps ``d`` on the active backend.
        """
        a_host = np.asarray(xp.to_numpy(a), dtype=float)
        d_host = np.asarray(xp.to_numpy(d), dtype=float)
        fraction, face = self._hits(a_host, a_host + d_host, t_min)
        t = np.where(face >= 0, fraction, np.inf)
        normal = self._oriented_normals(face, d_host)
        return xp.asarray(t), xp.asarray(normal), True

    def _result(self, a, b, fraction, face, shape, normal=None, **extra) -> SurfaceHits:
        """Pack hits of ``(3, n)`` segments into point-major `SurfaceHits`."""
        d = b - a
        hit = face >= 0
        t = np.where(hit, fraction, np.nan)
        point = a + np.where(hit, fraction, 0.0) * d
        if normal is None:
            normal = self._oriented_normals(face, d)
        if self.inverted:
            normal = -normal
        point = np.where(hit, point, np.nan)
        normal = np.where(hit, normal, np.nan)
        return SurfaceHits(
            hit=hit.reshape(shape),
            fraction=t.reshape(shape),
            distance=(t * np.linalg.norm(d, axis=0)).reshape(shape),
            point=point.T.reshape(*shape, 3),
            normal=normal.T.reshape(*shape, 3),
            face=face.reshape(shape),
            **{key: value.reshape(shape) for key, value in extra.items()},
        )

    def first_hit(self, starts: Array, ends: Array) -> SurfaceHits:
        """Return the first hit of each segment ``starts[...] -> ends[...]``, ``(..., 3)``.

        The point-major form of `crossings` with the face hit; a hit at the
        very start of a segment (a fraction below ``1e-12``) is not counted, so a
        segment that starts on the wall leaves it.
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
        fraction, face = self._hits(a, b, START_TOLERANCE)
        return self._result(a, b, fraction, face, shape)

    def volume(self) -> float:
        """Return the enclosed volume of a closed mesh."""
        if not self.closed:
            raise ValueError("an open mesh encloses no volume")
        return float(_signed_volume(self._vertices, self._faces))

    def surface_area(self) -> float:
        """Return the total area of the faces."""
        return float(self._face_areas.sum())

    def to_mesh(self, **kwargs: Any):
        """Return the triangles as a `SurfaceMesh` (``resolution`` and ``bounds`` are ignored)."""
        from simgeometry.mesh import SurfaceMesh

        spacing = (self.edge_length,) * 3
        return SurfaceMesh(self._vertices.copy(), self._faces.copy(), spacing)
