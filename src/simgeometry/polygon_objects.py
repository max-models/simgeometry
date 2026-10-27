"""2D objects made of polygons, packed for the swept absorption kernel.

:func:`polygon_objects` packs the objects the ``absorb_polygons_2d`` kernel can
take (a :class:`~simgeometry.objects.polygon.Polygon`, an
:class:`~simgeometry.objects.aperture2d.Aperture2D`: the union of its two
plates) into flat arrays, so markers are tested against every object in one
pass instead of one containment call per object and sample along the step.
:meth:`PolygonObjects.absorb` runs the kernel: a marker is absorbed when it
ends inside an object or when its straight step from ``start`` crosses one of
the object's edges, exactly (no sampling), so a plate thinner than the step is
not jumped over. The object met first along the step takes the marker.

Example::

    from simgeometry.objects import Aperture2D
    from simgeometry.polygon_objects import polygon_objects

    packed = polygon_objects([Aperture2D.slit(...)])
    hit = packed.absorb(new_positions, alive, start=old_positions)
    # hit[i] is the slot of the object marker i hit (-1: none), alive updated
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np

from simgeometry.objects.aperture2d import Aperture2D
from simgeometry.objects.polygon import Polygon

Array = Any


def _c_ordered(points: Array) -> Array:
    """``points`` as a float ``(2, N)`` array with C strides, copied only if needed.

    A ``(2, 1)`` column picked by fancy indexing keeps Fortran strides that NumPy
    still flags as C-contiguous; Pyccel checks the strides.
    """
    points = xp.ascontiguousarray(xp.asarray(points, dtype=float))
    itemsize = points.dtype.itemsize
    if tuple(points.strides) != (points.shape[1] * itemsize, itemsize):
        points = points.copy(order="C")
    return points


@dataclass(frozen=True)
class PolygonObjects:
    """Polygon objects ``objects[index[k]]``, slot ``k``, as kernel arrays.

    Slot ``k`` is polygons ``first[k] <= p < last[k]``, polygon ``p`` the rows
    ``offsets[p] <= v < offsets[p + 1]`` of ``vertices`` (V, 2). ``boxed[k]``
    says whether a marker outside the box ``lower[k]``..``upper[k]`` cannot touch
    the object (false for an inverted one, solid outside its box). The ``_host``
    copies are for building per-call arrays without device transfers.
    """

    index: tuple[int, ...]
    vertices: Array
    offsets: Array
    first: Array
    last: Array
    inverted: Array
    lower: Array
    upper: Array
    boxed: Array
    first_host: np.ndarray
    last_host: np.ndarray
    inverted_host: np.ndarray

    def slot(self, object_index: int) -> int | None:
        """The slot of ``objects[object_index]``, or None if it is not here."""
        try:
            return self.index.index(object_index)
        except ValueError:
            return None

    def kernel_arguments(self) -> tuple[Array, ...]:
        """The object arguments of ``absorb_polygons_2d``, after the sweep length."""
        return (
            self.vertices,
            self.offsets,
            self.first,
            self.last,
            self.inverted,
            self.lower,
            self.upper,
            self.boxed,
        )

    def absorb(
        self,
        positions: Array,
        alive: Array,
        *,
        start: Array | None = None,
        rows: Array | None = None,
        sweep_length: float = np.inf,
    ) -> Array:
        """Absorb the alive markers that end in, or step through, an object.

        Args:
            positions: The marker positions after the step, ``(2, N)``.
            alive: ``(N,)`` booleans; absorbed markers are set to False.
            start: The positions before the step, ``(2, N)``; without it only
                the end points count.
            rows: The markers to test (default: all ``N``).
            sweep_length: Steps longer than this count by their end only (e.g.
                steps wrapped by a periodic side).

        Returns:
            ``hit``, one entry per row: the slot of the object it hit (the
            object ``objects[index[hit]]``), or -1.
        """
        from simgeometry.kernels.absorb_polygons_2d import kernel

        positions = _c_ordered(positions)
        if rows is None:
            rows = xp.arange(positions.shape[1], dtype=xp.int64)
        rows = xp.asarray(rows, dtype=xp.int64)
        use_start = start is not None
        if use_start:
            start = _c_ordered(start)
        hit = xp.empty(rows.shape[0], dtype=xp.int64)
        if rows.shape[0] == 0:
            return hit
        kernel(
            positions,
            start if use_start else positions,
            rows,
            int(use_start),
            float(sweep_length) if use_start else 0.0,
            *self.kernel_arguments(),
            # the CUDA version takes the mask as char (int8), the same bytes
            alive.view(xp.int8) if xp.is_gpu(alive) and alive.dtype == bool else alive,
            hit,
        )
        return hit


def _host_vertices(vertices: Array) -> np.ndarray:
    if xp.is_gpu(vertices):
        vertices = xp.to_numpy(vertices)
    return np.asarray(vertices, dtype=float)


def object_polygons(obj: object) -> list[np.ndarray] | None:
    """The (n, 2) vertex arrays whose union is ``obj``, or None if it is not polygons."""
    if isinstance(obj, Aperture2D):
        plates = (obj._lower_polygon, obj._upper_polygon)
        return [_host_vertices(p.vertices) for p in plates]
    if isinstance(obj, Polygon):
        return [_host_vertices(obj.vertices)]
    return None


#: The packed objects of recent calls; the key holds the objects themselves
#: (alive while cached, so their ids are not reused).
_CACHE: OrderedDict[tuple, tuple[tuple, PolygonObjects | None]] = OrderedDict()
_CACHE_SIZE = 16


def _pack(objects: Sequence[object], skip: frozenset[int]) -> PolygonObjects | None:
    index, vertices, offsets, first, last = [], [], [0], [], []
    inverted, lower, upper, boxed = [], [], [], []
    for object_index, obj in enumerate(objects):
        polygons = None if object_index in skip else object_polygons(obj)
        if not polygons:
            continue
        index.append(object_index)
        first.append(len(offsets) - 1)
        for polygon in polygons:
            vertices.append(polygon)
            offsets.append(offsets[-1] + polygon.shape[0])
        last.append(len(offsets) - 1)
        is_inverted = bool(getattr(obj, "inverted", False))
        inverted.append(int(is_inverted))
        lower.append(np.asarray(obj.lower_bound, dtype=float).reshape(2))
        upper.append(np.asarray(obj.upper_bound, dtype=float).reshape(2))
        boxed.append(int(not is_inverted))
    if not index:
        return None

    def integers(values: list[int]) -> Array:
        return xp.asarray(np.asarray(values, dtype=np.int64))

    return PolygonObjects(
        index=tuple(index),
        vertices=xp.asarray(np.concatenate(vertices)),
        offsets=integers(offsets),
        first=integers(first),
        last=integers(last),
        inverted=integers(inverted),
        lower=xp.asarray(np.array(lower)),
        upper=xp.asarray(np.array(upper)),
        boxed=integers(boxed),
        first_host=np.asarray(first, dtype=np.int64),
        last_host=np.asarray(last, dtype=np.int64),
        inverted_host=np.asarray(inverted, dtype=np.int64),
    )


def polygon_objects(
    objects: Sequence[object] | None,
    skip: frozenset[int] = frozenset(),
) -> PolygonObjects | None:
    """Pack the polygon objects among ``objects`` (not those indexed in ``skip``).

    Objects that are not polygons (e.g. a ``Circle``) are left out; None if none
    is left. The objects are expected to stay fixed, so the packing is cached
    by their identity (and the array backend).
    """
    if not objects:
        return None
    key = (tuple(id(obj) for obj in objects), skip, xp.get_backend())
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached[1]
    packed = _pack(objects, skip)
    _CACHE[key] = (tuple(objects), packed)
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return packed
