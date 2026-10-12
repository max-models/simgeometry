"""Mask of the markers inside a 2D polygon, by even-odd ray casting.

The Pyccel version (CPU) of the ``contains_polygon_2d`` kernel;
``contains_polygon_2d_cuda.cu`` next to this file is the CUDA version (GPU), with
the same arguments.  ``positions`` is ``(2, N)`` (component-major), ``vertices`` is (n, 2), the closing edge implied.  The same
crossing rule as ``Markers._polygon_contains_points``.  With ``inverted == 1``
the mask is negated.
"""


def contains_polygon_2d(
    positions: "float[:,:]",
    mask: "bool[:]",
    vertices: "float[:,:]",
    inverted: int,
) -> None:
    """Set ``mask[marker]`` to whether the marker lies inside the polygon."""
    num_markers = positions.shape[1]
    num_vertices = vertices.shape[0]
    # fmt: off
    #$ omp parallel for private(x, y, inside, xj, yj, i, xi, yi) schedule(dynamic, 2048)
    # fmt: on
    for marker in range(num_markers):
        x = positions[0, marker]
        y = positions[1, marker]
        inside = False
        xj = vertices[num_vertices - 1, 0]
        yj = vertices[num_vertices - 1, 1]
        for i in range(num_vertices):
            xi = vertices[i, 0]
            yi = vertices[i, 1]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (
                (yj - yi) + 1e-16
            ) + xi:
                inside = not inside
            xj = xi
            yj = yi
        if inverted == 1:
            inside = not inside
        mask[marker] = inside
