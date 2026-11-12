"""The first triangle each segment meets, walking the cells of a uniform grid.

The Pyccel version (CPU) of the ``mesh_first_hits`` kernel; there is no CUDA version
yet (CuPy arrays are copied to the host). ``mesh_first_hits_numpy.py`` is the NumPy
version, with the same arguments and results.

Segment ``i`` runs from ``starts[:, i]`` to ``ends[:, i]`` (component-major, ``(3, N)``).
The grid (`simgeometry._triangle_grid.TriangleGrid`) lists for every cell the
faces whose box overlaps it. The segment is clipped to the grid's box and its cells
are visited in order (Amanatides and Woo); in every cell its triangles are tested by
the Möller–Trumbore algorithm, and the walk stops as soon as the nearest hit so far
lies before the cell's exit. A hit at a fraction ``t`` with ``t_min < t <= 1`` counts;
points on a triangle's edges count for both triangles sharing them.
"""


def _triangle_fraction(
    vertices: "float[:,:]",
    faces: "int[:,:]",
    face: int,
    ax: float,
    ay: float,
    az: float,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    """Fraction ``t`` where ``a + t d`` meets the triangle ``face``, or 2 for none.

    The barycentric coordinates of the hit are not returned: the caller computes
    them again for the face it keeps.
    """
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
        return 2.0
    inv = 1.0 / det
    sx = ax - vertices[v0, 0]
    sy = ay - vertices[v0, 1]
    sz = az - vertices[v0, 2]
    u = (sx * px + sy * py + sz * pz) * inv
    if u < 0.0 or u > 1.0:
        return 2.0
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return 2.0
    return (e2x * qx + e2y * qy + e2z * qz) * inv


def mesh_first_hits(
    starts: "float[:,:]",
    ends: "float[:,:]",
    vertices: "float[:,:]",
    faces: "int[:,:]",
    lower: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    cell_start: "int[:]",
    cell_faces: "int[:]",
    t_min: float,
    fraction: "float[:]",
    face: "int[:]",
) -> None:
    """Set ``fraction[i]`` and ``face[i]`` of the first hit of segment ``i``.

    Without a hit ``fraction[i]`` is 2 and ``face[i]`` is -1.
    """
    nx = cells[0]
    ny = cells[1]
    nz = cells[2]
    big = 1e300
    # fmt: off
    #$ omp parallel for private(ax, ay, az, dx, dy, dz, gx, gy, gz, gdx, gdy, gdz, t0, t1, ta, tb, missed, tc, ix, iy, iz, sx, sy, sz, mx, my, mz, ddx, ddy, ddz, best, best_face, cell, entry, candidate, t, exit_t) schedule(dynamic, 64)
    # fmt: on
    for i in range(starts.shape[1]):
        fraction[i] = 2.0
        face[i] = -1
        ax = starts[0, i]
        ay = starts[1, i]
        az = starts[2, i]
        dx = ends[0, i] - ax
        dy = ends[1, i] - ay
        dz = ends[2, i] - az
        # the segment in grid units, clipped to the grid's box
        gx = (ax - lower[0]) / spacing[0]
        gy = (ay - lower[1]) / spacing[1]
        gz = (az - lower[2]) / spacing[2]
        gdx = dx / spacing[0]
        gdy = dy / spacing[1]
        gdz = dz / spacing[2]
        t0 = 0.0
        t1 = 1.0
        missed = False
        if gdx != 0.0:
            ta = -gx / gdx
            tb = (nx - gx) / gdx
            t0 = max(t0, min(ta, tb))
            t1 = min(t1, max(ta, tb))
        elif gx < 0.0 or gx > nx:
            missed = True
        if gdy != 0.0:
            ta = -gy / gdy
            tb = (ny - gy) / gdy
            t0 = max(t0, min(ta, tb))
            t1 = min(t1, max(ta, tb))
        elif gy < 0.0 or gy > ny:
            missed = True
        if gdz != 0.0:
            ta = -gz / gdz
            tb = (nz - gz) / gdz
            t0 = max(t0, min(ta, tb))
            t1 = min(t1, max(ta, tb))
        elif gz < 0.0 or gz > nz:
            missed = True
        if missed or t0 > t1:
            continue
        tc = t0
        ix = min(max(int(gx + tc * gdx), 0), nx - 1)
        iy = min(max(int(gy + tc * gdy), 0), ny - 1)
        iz = min(max(int(gz + tc * gdz), 0), nz - 1)
        # the walk: the fraction where the segment leaves the cell along each axis
        sx = 0
        mx = big
        ddx = big
        if gdx > 0.0:
            sx = 1
            mx = (ix + 1 - gx) / gdx
            ddx = 1.0 / gdx
        elif gdx < 0.0:
            sx = -1
            mx = (ix - gx) / gdx
            ddx = -1.0 / gdx
        sy = 0
        my = big
        ddy = big
        if gdy > 0.0:
            sy = 1
            my = (iy + 1 - gy) / gdy
            ddy = 1.0 / gdy
        elif gdy < 0.0:
            sy = -1
            my = (iy - gy) / gdy
            ddy = -1.0 / gdy
        sz = 0
        mz = big
        ddz = big
        if gdz > 0.0:
            sz = 1
            mz = (iz + 1 - gz) / gdz
            ddz = 1.0 / gdz
        elif gdz < 0.0:
            sz = -1
            mz = (iz - gz) / gdz
            ddz = -1.0 / gdz
        best = 2.0
        best_face = -1
        while True:
            cell = (ix * ny + iy) * nz + iz
            for entry in range(cell_start[cell], cell_start[cell + 1]):
                candidate = cell_faces[entry]
                t = _triangle_fraction(
                    vertices, faces, candidate, ax, ay, az, dx, dy, dz
                )
                if t > t_min and t <= 1.0 and t < best:
                    best = t
                    best_face = candidate
            exit_t = min(mx, min(my, mz))
            if best <= exit_t or exit_t > t1:
                break
            if mx <= my and mx <= mz:
                ix += sx
                mx += ddx
                if ix < 0 or ix >= nx:
                    break
            elif my <= mz:
                iy += sy
                my += ddy
                if iy < 0 or iy >= ny:
                    break
            else:
                iz += sz
                mz += ddz
                if iz < 0 or iz >= nz:
                    break
        fraction[i] = best
        face[i] = best_face
