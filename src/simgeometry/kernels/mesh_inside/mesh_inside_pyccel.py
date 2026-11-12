"""Mask of the points inside a closed triangle mesh, from the node states of its grid.

The Pyccel version (CPU) of the ``mesh_inside`` kernel; there is no CUDA version yet
(CuPy arrays are copied to the host). ``mesh_inside_numpy.py`` is the NumPy version,
with the same arguments and result.

``points`` is ``(3, N)`` (component-major). A point outside the grid's box is outside.
Otherwise the node of its cell nearest to it has a known state (``node_inside``), and
the segment from the point to that node, which stays in the cell, crosses the surface
an odd number of times exactly when the two states differ; only the triangles listed
for the cell can cross it.
"""


def _crosses(
    vertices: "float[:,:]",
    faces: "int[:,:]",
    face: int,
    ax: float,
    ay: float,
    az: float,
    dx: float,
    dy: float,
    dz: float,
) -> bool:
    """Whether ``a + t d``, ``0 < t <= 1``, meets the triangle ``face`` (Möller–Trumbore)."""
    v0 = faces[face, 0]
    v1 = faces[face, 1]
    v2 = faces[face, 2]
    e1x = vertices[v1, 0] - vertices[v0, 0]
    e1y = vertices[v1, 1] - vertices[v0, 1]
    e1z = vertices[v1, 2] - vertices[v0, 2]
    e2x = vertices[v2, 0] - vertices[v0, 0]
    e2y = vertices[v2, 1] - vertices[v0, 1]
    e2z = vertices[v2, 2] - vertices[v0, 2]
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if det == 0.0:
        return False
    inv = 1.0 / det
    sx = ax - vertices[v0, 0]
    sy = ay - vertices[v0, 1]
    sz = az - vertices[v0, 2]
    u = (sx * px + sy * py + sz * pz) * inv
    if u < 0.0 or u > 1.0:
        return False
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return False
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    return t > 0.0 and t <= 1.0


def mesh_inside(
    points: "float[:,:]",
    vertices: "float[:,:]",
    faces: "int[:,:]",
    lower: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    cell_start: "int[:]",
    cell_faces: "int[:]",
    node_inside: "bool[:]",
    inside: "bool[:]",
) -> None:
    """Set ``inside[i]`` to whether point ``i`` lies inside the closed mesh."""
    nx = cells[0]
    ny = cells[1]
    nz = cells[2]
    # fmt: off
    #$ omp parallel for private(gx, gy, gz, ix, iy, iz, cx, cy, cz, qx, qy, qz, state, cell, entry) schedule(dynamic, 1024)
    # fmt: on
    for i in range(points.shape[1]):
        inside[i] = False
        gx = (points[0, i] - lower[0]) / spacing[0]
        gy = (points[1, i] - lower[1]) / spacing[1]
        gz = (points[2, i] - lower[2]) / spacing[2]
        if gx < 0.0 or gy < 0.0 or gz < 0.0 or gx > nx or gy > ny or gz > nz:
            continue
        ix = min(int(gx), nx - 1)
        iy = min(int(gy), ny - 1)
        iz = min(int(gz), nz - 1)
        cx = ix
        if gx - ix >= 0.5:
            cx = ix + 1
        cy = iy
        if gy - iy >= 0.5:
            cy = iy + 1
        cz = iz
        if gz - iz >= 0.5:
            cz = iz + 1
        qx = lower[0] + cx * spacing[0]
        qy = lower[1] + cy * spacing[1]
        qz = lower[2] + cz * spacing[2]
        state = node_inside[(cx * (ny + 1) + cy) * (nz + 1) + cz]
        cell = (ix * ny + iy) * nz + iz
        for entry in range(cell_start[cell], cell_start[cell + 1]):
            if _crosses(
                vertices,
                faces,
                cell_faces[entry],
                points[0, i],
                points[1, i],
                points[2, i],
                qx - points[0, i],
                qy - points[1, i],
                qz - points[2, i],
            ):
                state = not state
        inside[i] = state
