"""The NumPy version of the ``mesh_first_hits`` kernel, run when Pyccel is unavailable.

Same arguments and results as ``mesh_first_hits_pyccel.py``, vectorized over the
segments: the cells of each segment come from the fractions where it crosses the
grid planes, and all their faces are tested (no early stop), in chunks of segments.
"""

from __future__ import annotations

import numpy as np

from simgeometry.kernels.triangle_hits import triangle_fractions

#: Segments handled at once.
CHUNK = 2048


def _ragged(counts):
    """The owner and the position within its run of every entry of runs ``counts``."""
    owner = np.repeat(np.arange(counts.shape[0]), counts)
    starts = np.cumsum(counts) - counts
    return owner, np.arange(owner.shape[0]) - starts[owner]


def segment_cells(g, gd, cells):
    """``(segment, cell)`` pairs of the cells the segments ``g + t gd`` pass through.

    ``g`` and ``gd`` are ``(3, n)`` in grid units (cell ``i`` spans ``[i, i + 1)``);
    the cells are numbered as in `simgeometry._triangle_grid.TriangleGrid`.
    """
    n = g.shape[1]
    t0 = np.zeros(n)
    t1 = np.ones(n)
    missed = np.zeros(n, dtype=bool)
    for k in range(3):
        moving = gd[k] != 0.0
        safe = np.where(moving, gd[k], 1.0)
        ta = -g[k] / safe
        tb = (cells[k] - g[k]) / safe
        t0 = np.where(moving, np.maximum(t0, np.minimum(ta, tb)), t0)
        t1 = np.where(moving, np.minimum(t1, np.maximum(ta, tb)), t1)
        missed |= ~moving & ((g[k] < 0.0) | (g[k] > cells[k]))
    rows = np.flatnonzero(~missed & (t0 <= t1))
    t0, t1 = t0[rows], t1[rows]
    # the fractions where each segment crosses a grid plane between t0 and t1
    owners, fractions = [rows, rows], [t0, t1]
    for k in range(3):
        x0 = g[k, rows] + t0 * gd[k, rows]
        x1 = g[k, rows] + t1 * gd[k, rows]
        lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
        first = np.floor(lo).astype(np.int64) + 1
        count = np.maximum(np.ceil(hi).astype(np.int64) - first, 0)
        owner, offset = _ragged(count)
        plane = first[owner] + offset
        segment = rows[owner]
        fractions.append((plane - g[k, segment]) / gd[k, segment])
        owners.append(segment)
    owner = np.concatenate(owners)
    t = np.concatenate(fractions)
    order = np.lexsort((t, owner))
    owner, t = owner[order], t[order]
    same = owner[1:] == owner[:-1]
    middle = 0.5 * (t[1:] + t[:-1])[same]
    segment = owner[1:][same]
    index = np.floor(g[:, segment] + middle * gd[:, segment]).astype(np.int64)
    for k in range(3):
        index[k] = np.clip(index[k], 0, cells[k] - 1)
    cell = (index[0] * cells[1] + index[1]) * cells[2] + index[2]
    return segment, cell


def _chunk(a, d, vertices, faces, lower, spacing, cells, cell_start, cell_faces, t_min):
    """First hits of the segments ``a + t d`` of one chunk: fractions and faces."""
    n = a.shape[1]
    g = (a - lower[:, None]) / spacing[:, None]
    gd = d / spacing[:, None]
    segment, cell = segment_cells(g, gd, cells)
    count = cell_start[cell + 1] - cell_start[cell]
    pair, offset = _ragged(count)
    candidate = cell_faces[cell_start[cell[pair]] + offset]
    row = segment[pair]
    t, _, _ = triangle_fractions(vertices, faces, candidate, a[:, row], d[:, row])
    valid = (t > t_min) & (t <= 1.0)
    row, t, candidate = row[valid], t[valid], candidate[valid]
    order = np.lexsort((candidate, t, row))
    row, t, candidate = row[order], t[order], candidate[order]
    first = np.ones(row.shape[0], dtype=bool)
    first[1:] = row[1:] != row[:-1]
    fraction = np.full(n, 2.0)
    face = np.full(n, -1, dtype=np.int64)
    fraction[row[first]] = t[first]
    face[row[first]] = candidate[first]
    return fraction, face


def mesh_first_hits(
    starts: np.ndarray,
    ends: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    lower: np.ndarray,
    spacing: np.ndarray,
    cells: np.ndarray,
    cell_start: np.ndarray,
    cell_faces: np.ndarray,
    t_min: float,
    fraction: np.ndarray,
    face: np.ndarray,
) -> None:
    """Set ``fraction[i]`` and ``face[i]`` of the first hit of segment ``i``.

    Without a hit ``fraction[i]`` is 2 and ``face[i]`` is -1.
    """
    n = starts.shape[1]
    for lo in range(0, n, CHUNK):
        sl = slice(lo, min(n, lo + CHUNK))
        a = starts[:, sl]
        d = ends[:, sl] - a
        t, f = _chunk(
            a, d, vertices, faces, lower, spacing, cells, cell_start, cell_faces, t_min
        )
        fraction[sl] = t
        face[sl] = f
