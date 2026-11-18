"""The polygon-object hit test of the Pyccel kernels, shared with other packages.

The Pyccel version of ``polygon_hit.cuh`` (CUDA) and ``polygon_hits.py`` (NumPy), with the
same arithmetic. ``absorb_polygons_2d`` uses it, and so do Pyccel kernels of other packages
that fuse the test into their own loop (picnic's ``push_electrostatic_2d``)::

    from simgeometry.kernels.polygon_hit_pyccel import object_hit_time, polygon_bounds

cunumpy's kernel cache (``PyccelCache``) builds this module with the kernel. The functions
have loops, so they are plain functions, not ``@inline``: a call keeps its locals on the
stack, so they are safe under ``#$ omp parallel for``.
"""


def polygon_bounds(
    vertices: "float[:,:]",
    polygon_offsets: "int[:]",
    polygon_lower: "float[:,:]",
    polygon_upper: "float[:,:]",
) -> None:
    """Write the bounding box of every polygon into ``polygon_lower`` and ``polygon_upper``.

    Polygon ``p`` is the vertices ``polygon_offsets[p] <= v < polygon_offsets[p + 1]``
    of ``vertices`` (V, 2); the boxes are ``(polygons, 2)``.
    """
    num_polygons = polygon_offsets.shape[0] - 1
    for polygon in range(num_polygons):
        for axis in range(2):
            polygon_lower[polygon, axis] = 1.0e300
            polygon_upper[polygon, axis] = -1.0e300
            for vertex in range(polygon_offsets[polygon], polygon_offsets[polygon + 1]):
                if vertices[vertex, axis] < polygon_lower[polygon, axis]:
                    polygon_lower[polygon, axis] = vertices[vertex, axis]
                if vertices[vertex, axis] > polygon_upper[polygon, axis]:
                    polygon_upper[polygon, axis] = vertices[vertex, axis]


def object_hit_time(
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
    polygon_lower: "float[:,:]",
    polygon_upper: "float[:,:]",
) -> float:
    """Return where along the step from (x0, y0) to (x1, y1) the marker enters the object.

    The object is the union of polygons ``first <= p < last`` (solid outside them with
    ``inverted == 1``). The result is the parameter in (0, 1] of the first edge crossing
    (with ``sweep == 1``), else 0 if (x1, y1) lies in the solid, else 2 (no hit).
    ``polygon_lower`` and ``polygon_upper`` are the polygons' bounding boxes
    (`polygon_bounds`): a polygon the step stays beside along an axis is skipped, which
    changes no result.
    """
    inside = False
    crossing = 2.0
    dx = x1 - x0
    dy = y1 - y0
    for polygon in range(first, last):
        # the step and the polygon apart along an axis: neither inside nor crossed
        if (
            (x0 < polygon_lower[polygon, 0] and x1 < polygon_lower[polygon, 0])
            or (x0 > polygon_upper[polygon, 0] and x1 > polygon_upper[polygon, 0])
            or (y0 < polygon_lower[polygon, 1] and y1 < polygon_lower[polygon, 1])
            or (y0 > polygon_upper[polygon, 1] and y1 > polygon_upper[polygon, 1])
        ):
            continue
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
