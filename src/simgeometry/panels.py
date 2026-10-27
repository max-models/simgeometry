"""Boundary panels: the straight edges of a 2D outline, for boundary-element solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Panels:
    """Straight panels along the boundary of a 2D shape, as component-major host arrays.

    ``start`` and ``end`` are ``(2, n)``; ``piece`` ``(n,)`` is the index of the
    outline loop (see ``outlines()``) each panel belongs to, so the panels of one loop
    are contiguous and in order. ``closed`` is False for an open curve.

    :attr:`normal` is the unit normal to the right of each panel's direction. Loops
    keep the shape on their left, so for a closed shape it points out of the
    geometry, at outer boundaries and holes alike (inversion is ignored, as in
    ``outlines()``). For an open curve it is the side to the right of the curve's
    direction, from its first vertex to its last.
    """

    start: np.ndarray
    end: np.ndarray
    piece: np.ndarray
    closed: bool = True

    @property
    def midpoint(self) -> np.ndarray:
        """Panel midpoints, ``(2, n)``."""
        return 0.5 * (self.start + self.end)

    @property
    def length(self) -> np.ndarray:
        """Panel lengths, ``(n,)``."""
        return np.hypot(*(self.end - self.start))

    @property
    def tangent(self) -> np.ndarray:
        """Unit tangents from start to end, ``(2, n)``."""
        return (self.end - self.start) / self.length

    @property
    def normal(self) -> np.ndarray:
        """Unit normals to the right of the tangents, ``(2, n)``."""
        tx, ty = self.tangent
        return np.stack((ty, -tx))

    def __len__(self) -> int:
        return int(self.start.shape[1])


def boundary_panels(shape, max_spacing=None) -> Panels:
    """The edges of ``shape.outlines(max_spacing)`` as :class:`Panels`.

    Each edge of a closed loop is a panel, including the one back to its first
    vertex; an open curve has one panel fewer than vertices. Zero-length edges are
    dropped.
    """
    closed = bool(getattr(shape, "closed", True))
    starts, ends, pieces = [], [], []
    for index, loop in enumerate(shape.outlines(max_spacing)):
        loop = np.asarray(loop, dtype=float)
        start = loop if closed else loop[:, :-1]
        end = np.roll(loop, -1, axis=1) if closed else loop[:, 1:]
        keep = np.hypot(*(end - start)) > 0
        starts.append(start[:, keep])
        ends.append(end[:, keep])
        pieces.append(np.full(int(keep.sum()), index))
    if not starts:
        empty = np.zeros((2, 0))
        return Panels(empty, empty.copy(), np.zeros(0, dtype=int), closed)
    return Panels(
        np.concatenate(starts, axis=1),
        np.concatenate(ends, axis=1),
        np.concatenate(pieces),
        closed,
    )
