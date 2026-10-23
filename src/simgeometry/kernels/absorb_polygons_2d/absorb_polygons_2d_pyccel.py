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
object it meets first along the step takes it. Kernels that fuse the test into
their own loop (picnic's ``push_electrostatic_2d``) copy ``_object_hit_time``;
``polygon_hit.cuh`` and ``polygon_hits.py`` are its CUDA and NumPy forms.
"""


def _object_hit_time(
    vertices: "float[:,:]",
    polygon_offsets: "int[:]",
    first: int,
    last: int,
    inverted: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sweep: int,
) -> float:
    """Where along the step from (x0, y0) to (x1, y1) the marker enters the object.

    The parameter in (0, 1] of the first edge crossing (with ``sweep == 1``),
    else 0 if (x1, y1) lies in the solid, else 2 (no hit).
    """
    inside = False
    crossing = 2.0
    dx = x1 - x0
    dy = y1 - y0
    for polygon in range(first, last):
        begin = polygon_offsets[polygon]
        end = polygon_offsets[polygon + 1]
        in_polygon = False
        xj = vertices[end - 1, 0]
        yj = vertices[end - 1, 1]
        for vertex in range(begin, end):
            xi = vertices[vertex, 0]
            yi = vertices[vertex, 1]
            if (yi > y1) != (yj > y1) and x1 < (xj - xi) * (y1 - yi) / (
                (yj - yi) + 1e-16
            ) + xi:
                in_polygon = not in_polygon
            if sweep == 1:
                # the step x0 + t d meets the edge xj + u e
                ex = xi - xj
                ey = yi - yj
                denominator = dx * ey - dy * ex
                if denominator != 0.0:
                    wx = xj - x0
                    wy = yj - y0
                    t = (wx * ey - wy * ex) / denominator
                    u = (wx * dy - wy * dx) / denominator
                    if 0.0 < t <= 1.0 and 0.0 <= u <= 1.0 and t < crossing:
                        crossing = t
            xj = xi
            yj = yi
        if in_polygon:
            inside = True
    if inverted == 1:
        inside = not inside
    if crossing <= 1.0:
        return crossing
    if inside:
        return 0.0
    return 2.0


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
            time = _object_hit_time(
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
            )
            if time < best:
                best = time
                hit[index] = obj
        if hit[index] >= 0:
            alive[marker] = False
