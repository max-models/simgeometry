"""The NumPy version of the ``mesh_inside`` kernel, run when Pyccel is unavailable.

Same arguments and result as ``mesh_inside_pyccel.py``, vectorized over the points
(all faces of each point's cell at once), in chunks of points.
"""

from __future__ import annotations

import numpy as np

from simgeometry.kernels.triangle_hits import triangle_fractions

#: Points handled at once.
CHUNK = 65536


def _chunk(
    points, vertices, faces, lower, spacing, cells, cell_start, cell_faces, node_inside
):
    """The mask of one chunk of ``(3, n)`` points."""
    n = points.shape[1]
    g = (points - lower[:, None]) / spacing[:, None]
    in_box = np.all((g >= 0.0) & (g <= cells[:, None]), axis=0)
    rows = np.flatnonzero(in_box)
    g = g[:, rows]
    index = np.minimum(np.floor(g).astype(np.int64), (cells - 1)[:, None])
    corner = index + (g - index >= 0.5)
    node = (corner[0] * (cells[1] + 1) + corner[1]) * (cells[2] + 1) + corner[2]
    state = node_inside[node]
    cell = (index[0] * cells[1] + index[1]) * cells[2] + index[2]
    count = cell_start[cell + 1] - cell_start[cell]
    owner = np.repeat(np.arange(rows.shape[0]), count)
    offset = np.arange(owner.shape[0]) - (np.cumsum(count) - count)[owner]
    candidate = cell_faces[cell_start[cell[owner]] + offset]
    a = points[:, rows[owner]]
    q = lower[:, None] + corner[:, owner] * spacing[:, None]
    t, _, _ = triangle_fractions(vertices, faces, candidate, a, q - a)
    crossings = np.bincount(owner, (t > 0.0) & (t <= 1.0), rows.shape[0])
    inside = np.zeros(n, dtype=bool)
    inside[rows] = state ^ (crossings.astype(np.int64) % 2 == 1)
    return inside


def mesh_inside(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    lower: np.ndarray,
    spacing: np.ndarray,
    cells: np.ndarray,
    cell_start: np.ndarray,
    cell_faces: np.ndarray,
    node_inside: np.ndarray,
    inside: np.ndarray,
) -> None:
    """Set ``inside[i]`` to whether point ``i`` lies inside the closed mesh."""
    n = points.shape[1]
    for lo in range(0, n, CHUNK):
        sl = slice(lo, min(n, lo + CHUNK))
        inside[sl] = _chunk(
            points[:, sl],
            vertices,
            faces,
            lower,
            spacing,
            cells,
            cell_start,
            cell_faces,
            node_inside,
        )
