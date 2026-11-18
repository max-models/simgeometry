"""Absorb the markers that end inside, or stepped through, 2D polygon objects.

The Pyccel version (CPU) of the ``absorb_polygons_2d`` kernel;
``absorb_polygons_2d_cuda.cu`` next to this file is the CUDA version (GPU), with
the same arguments.

An object is the union of polygons ``object_first[k] <= p < object_last[k]``,
polygon ``p`` the vertices ``polygon_offsets[p] <= v < polygon_offsets[p + 1]``
of ``vertices`` (V, 2) with the closing edge implied; ``object_inverted[k] == 1``
makes it solid outside them. A marker is inside a polygon by the even-odd ray
casting of ``contains_polygon_2d``. With a step from ``start`` it also hits an
object when the step crosses one of its edges, exactly (no sampling), and the
object it meets first along the step takes it. The test is ``object_hit_time`` of
``polygon_hit_pyccel.py``, shared with kernels of other packages that fuse it into their
own loop (picnic's ``push_electrostatic_2d``); ``polygon_hit.cuh`` and
``polygon_hits.py`` are its CUDA and NumPy forms.
"""

import numpy as np

from simgeometry.kernels.polygon_hit_pyccel import object_hit_time, polygon_bounds


def absorb_polygons_2d(
    positions: "float[:,:]",
    start: "float[:,:]",
    rows: "int[:]",
    use_start: int,
    sweep_length: float,
    vertices: "float[:,:]",
    polygon_offsets: "int[:]",
    object_first: "int[:]",
    object_last: "int[:]",
    object_inverted: "int[:]",
    object_lower: "float[:,:]",
    object_upper: "float[:,:]",
    object_boxed: "int[:]",
    alive: "bool[:]",
    hit: "int[:]",
) -> None:
    """Absorb the alive markers ``rows`` that hit an object; ``hit[i]`` is its index.

    ``positions`` and ``start`` are ``(2, N)``; ``hit`` has one entry per row
    (-1: no hit). With ``use_start == 1`` a step from ``start`` no longer than
    ``sweep_length`` is also tested for crossings (a longer one was wrapped by
    a periodic side or reinjected, and only its end counts). An object with
    ``object_boxed[k] == 1`` is skipped by steps outside its box
    ``object_lower[k]``..``object_upper[k]``.
    """
    num_objects = object_first.shape[0]
    # the bounding box of every polygon, for the hit test
    num_polygons = polygon_offsets.shape[0] - 1
    polygon_lower = np.empty((max(num_polygons, 1), 2))
    polygon_upper = np.empty((max(num_polygons, 1), 2))
    polygon_bounds(vertices, polygon_offsets, polygon_lower, polygon_upper)
    # fmt: off
    #$ omp parallel for private(marker, x0, y0, x1, y1, sweep, best, obj, time) schedule(dynamic, 256)
    # fmt: on
    for index in range(rows.shape[0]):
        marker = rows[index]
        hit[index] = -1
        if not alive[marker]:
            continue
        x1 = positions[0, marker]
        y1 = positions[1, marker]
        x0 = x1
        y0 = y1
        sweep = 0
        if use_start == 1:
            x0 = start[0, marker]
            y0 = start[1, marker]
            if (
                sweep_length > 0.0
                and (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)
                <= sweep_length * sweep_length
            ):
                sweep = 1
        best = 2.0
        for obj in range(num_objects):
            if object_boxed[obj] == 1 and (
                max(x0, x1) < object_lower[obj, 0]
                or min(x0, x1) > object_upper[obj, 0]
                or max(y0, y1) < object_lower[obj, 1]
                or min(y0, y1) > object_upper[obj, 1]
            ):
                continue
            time = object_hit_time(
                vertices,
                polygon_offsets,
                object_first[obj],
                object_last[obj],
                object_inverted[obj],
                x0,
                y0,
                x1,
                y1,
                sweep,
                polygon_lower,
                polygon_upper,
            )
            if time < best:
                best = time
                hit[index] = obj
        if hit[index] >= 0:
            alive[marker] = False
