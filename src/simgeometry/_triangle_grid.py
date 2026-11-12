"""A uniform grid of cells over a triangle mesh, for its ray and inside queries.

Every cell lists the triangles whose (slightly widened) bounding box overlaps it, so a
segment only tests the triangles of the cells it passes through, and a point only those
of its own cell. For a closed mesh the grid also stores whether each of its nodes is
inside: a point is then inside when the node nearest to it, a corner of its own cell,
is inside and the short segment between them crosses the surface an even number of
times. The node states come from crossings counted along the grid lines of all three
axes, by majority, so one miscounted crossing cannot flip a whole line of nodes.

The grid is shifted against the mesh by irrational fractions of a cell, so its lines
do not run through the vertices and edges of symmetric meshes (a torus whose vertices
lie in the planes ``y = 0`` or ``z = 0``), where a crossing could be counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Fractions of a cell by which the grid is shifted against the mesh, per axis.
_SHIFT = np.array([0.3819660112501051, 0.4142135623730951, 0.2360679774997897])
#: Default number of cells per triangle; the grid has about this many cells per face.
CELLS_PER_FACE = 4.0
#: Largest number of cells along one axis of the default grid.
MAX_CELLS_PER_AXIS = 512


@dataclass(frozen=True)
class TriangleGrid:
    """The cells of a triangle mesh, as arrays for the kernels.

    ``lower`` ``(3,)`` is the grid's corner, ``spacing`` ``(3,)`` the cell size and
    ``cells`` ``(3,)`` the number of cells per axis. Cell ``(i, j, k)`` is number
    ``(i * cells[1] + j) * cells[2] + k`` and lists the faces
    ``cell_faces[cell_start[c]:cell_start[c + 1]]``. ``node_inside`` holds one
    entry per node ``(i, j, k)``, ``i <= cells[0]`` and so on, in the same order:
    true inside the closed mesh; it is empty for an open mesh.
    """

    lower: np.ndarray
    spacing: np.ndarray
    cells: np.ndarray
    cell_start: np.ndarray
    cell_faces: np.ndarray
    node_inside: np.ndarray

    @property
    def upper(self) -> np.ndarray:
        """The grid's upper corner."""
        return self.lower + self.spacing * self.cells

    @property
    def faces_per_cell(self) -> float:
        """The mean number of faces of the cells that have any."""
        counts = np.diff(self.cell_start)
        occupied = counts[counts > 0]
        return float(occupied.mean()) if occupied.size else 0.0


def _ragged(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The owner and the position within its run of every entry of runs ``counts``."""
    owner = np.repeat(np.arange(counts.shape[0]), counts)
    starts = np.cumsum(counts) - counts
    return owner, np.arange(owner.shape[0]) - starts[owner]


def _grid_shape(lower, upper, n_faces: int, cells) -> np.ndarray:
    """Cells per axis: the given ones, or about ``CELLS_PER_FACE`` per face."""
    if cells is not None:
        shape = np.broadcast_to(np.asarray(cells, dtype=np.int64), (3,)).copy()
        if np.any(shape < 1):
            raise ValueError("cells must be positive")
        return shape
    extent = upper - lower
    total = max(1.0, CELLS_PER_FACE * n_faces)
    size = (np.prod(extent) / total) ** (1.0 / 3.0)
    shape = np.ceil(extent / size).astype(np.int64)
    return np.clip(shape, 1, MAX_CELLS_PER_AXIS)


def _node_states_along(vertices, faces, lower, spacing, cells, axis: int):
    """Node states ``(cells + 1)`` from the crossings counted along lines of ``axis``.

    Returns the states and a mask of the lines (as nodes) whose total count was odd,
    a sign of a crossing counted twice or missed.
    """
    b, c = (axis + 1) % 3, (axis + 2) % 3
    nodes = cells + 1
    corners = vertices[faces]  # (F, 3, 3)
    width = 1e-9 * spacing
    low = (corners.min(axis=1) - width - lower) / spacing
    high = (corners.max(axis=1) + width - lower) / spacing
    jb0 = np.clip(np.ceil(low[:, b]).astype(np.int64), 0, nodes[b])
    jb1 = np.clip(np.floor(high[:, b]).astype(np.int64), -1, nodes[b] - 1)
    jc0 = np.clip(np.ceil(low[:, c]).astype(np.int64), 0, nodes[c])
    jc1 = np.clip(np.floor(high[:, c]).astype(np.int64), -1, nodes[c] - 1)
    nb = np.maximum(jb1 - jb0 + 1, 0)
    nc = np.maximum(jc1 - jc0 + 1, 0)
    toggles = np.zeros((nodes[b], nodes[c], nodes[axis] + 1), dtype=np.int64)
    face, offset = _ragged(nb * nc)
    jb = jb0[face] + offset // nc[face]
    jc = jc0[face] + offset % nc[face]
    yb = lower[b] + jb * spacing[b]
    yc = lower[c] + jc * spacing[c]
    p = corners[face]
    # barycentric weights of the line's foot in the triangle projected on (b, c)
    w = []
    for i in range(3):
        v1, v2 = p[:, (i + 1) % 3], p[:, (i + 2) % 3]
        w.append(
            (v2[:, b] - v1[:, b]) * (yc - v1[:, c])
            - (v2[:, c] - v1[:, c]) * (yb - v1[:, b])
        )
    w = np.stack(w, axis=1)
    area = w.sum(axis=1)
    inside = (area != 0) & (np.all(w >= 0, axis=1) | np.all(w <= 0, axis=1))
    safe = np.where(area != 0, area, 1.0)
    x = np.sum(w * p[:, :, axis], axis=1) / safe
    first = np.floor((x - lower[axis]) / spacing[axis]).astype(np.int64) + 1
    keep = inside & (first <= nodes[axis] - 1)
    first = np.clip(first, 0, nodes[axis])
    np.add.at(toggles, (jb[keep], jc[keep], first[keep]), 1)
    counts = np.cumsum(toggles, axis=2)
    parity = (counts[:, :, : nodes[axis]] % 2).astype(np.int8)
    odd = (np.sum(toggles, axis=2) % 2) == 1
    order = np.argsort((b, c, axis))
    states = np.transpose(parity, order)
    odd_lines = np.transpose(np.broadcast_to(odd[:, :, None], parity.shape), order)
    return states, odd_lines


def build_triangle_grid(vertices, faces, *, cells=None, closed: bool) -> TriangleGrid:
    """Bin the ``(F, 3)`` faces of the ``(V, 3)`` vertices into a uniform grid.

    ``cells`` is the number of cells per axis (one or three integers); by default
    the grid has about `CELLS_PER_FACE` cells per face over the mesh's box, at most
    `MAX_CELLS_PER_AXIS` per axis. With ``closed`` the node states are computed.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    vmin, vmax = vertices.min(axis=0), vertices.max(axis=0)
    extent = vmax - vmin
    scale = float(extent.max()) if extent.max() > 0 else 1.0
    extent = np.maximum(extent, 1e-3 * scale)
    shape = _grid_shape(vmin, vmin + extent, faces.shape[0], cells)
    # a margin of at least a quarter of a cell on both sides, shifted by irrational
    # fractions of a cell: (0.25 + shift / 2) below, (0.75 - shift / 2) above
    spacing = np.where(shape > 1, extent / np.maximum(shape - 1, 1), 2.5 * extent)
    lower = vmin - spacing * (0.25 + 0.5 * _SHIFT)
    cells_shape = shape.astype(np.int64)
    # the faces of every cell, from their widened bounding boxes
    corners = vertices[faces]
    width = 1e-9 * spacing
    i0 = np.floor((corners.min(axis=1) - width - lower) / spacing).astype(np.int64)
    i1 = np.floor((corners.max(axis=1) + width - lower) / spacing).astype(np.int64)
    i0 = np.clip(i0, 0, cells_shape - 1)
    i1 = np.clip(i1, 0, cells_shape - 1)
    span = i1 - i0 + 1
    face, offset = _ragged(np.prod(span, axis=1))
    s = span[face]
    i = i0[face, 0] + offset // (s[:, 1] * s[:, 2])
    j = i0[face, 1] + (offset // s[:, 2]) % s[:, 1]
    k = i0[face, 2] + offset % s[:, 2]
    cell = (i * cells_shape[1] + j) * cells_shape[2] + k
    order = np.argsort(cell, kind="stable")
    n_cells = int(np.prod(cells_shape))
    cell_start = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=n_cells), out=cell_start[1:])
    cell_faces = np.ascontiguousarray(face[order], dtype=np.int64)
    node_inside = np.zeros(0, dtype=bool)
    if closed:
        votes = []
        for axis in range(3):
            states, _ = _node_states_along(
                vertices, faces, lower, spacing, cells_shape, axis
            )
            votes.append(states.astype(np.int64))
        node_inside = (sum(votes) >= 2).reshape(-1)
    return TriangleGrid(
        lower=np.ascontiguousarray(lower),
        spacing=np.ascontiguousarray(spacing),
        cells=cells_shape,
        cell_start=cell_start,
        cell_faces=cell_faces,
        node_inside=node_inside,
    )
