"""Triangle meshes: the grid, the inside and first-hit kernels, reading and writing."""

from __future__ import annotations

import importlib.util
import sys
import types

import cunumpy as xp
import numpy as np
import pytest

from simgeometry import GeometryCollection, TriangleMesh
from simgeometry.kernels.mesh_first_hits import kernel as mesh_first_hits
from simgeometry.kernels.mesh_inside import kernel as mesh_inside
from simgeometry.objects.triangle_mesh import uv_sphere

IMPLEMENTATIONS = [
    "numpy",
    "python",  # the uncompiled source of the Pyccel version
    pytest.param(
        "pyccel",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("pyccel") is None, reason="needs pyccel"
        ),
    ),
]

CUBE_VERTICES = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
)
# two triangles per face of the unit cube, outward
CUBE_FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2],  # x = 0
        [4, 6, 7], [4, 7, 5],  # x = 1
        [0, 4, 5], [0, 5, 1],  # y = 0
        [2, 3, 7], [2, 7, 6],  # y = 1
        [0, 2, 6], [0, 6, 4],  # z = 0
        [1, 5, 7], [1, 7, 3],  # z = 1
    ]
)  # fmt: skip


def _host(array):
    return np.asarray(xp.to_numpy(array))


def _sphere_crossing(a, b, radius=1.0):
    d = b - a
    aa = np.sum(d * d, axis=0)
    bb = np.sum(a * d, axis=0)
    cc = np.sum(a * a, axis=0) - radius**2
    root = np.sqrt(np.maximum(bb * bb - aa * cc, 0.0))
    t0, t1 = (-bb - root) / aa, (-bb + root) / aa
    return np.where(t0 > 0, t0, t1)


@pytest.fixture(scope="module")
def sphere():
    return uv_sphere(1.0, 12, 24)


def test_cube_is_closed_and_exact():
    cube = TriangleMesh(CUBE_VERTICES, CUBE_FACES)
    assert cube.closed and cube.n_faces == 12
    assert np.isclose(cube.volume(), 1.0) and np.isclose(cube.surface_area(), 6.0)
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 1.5, (3, 4000))
    expected = np.all((points > 0) & (points < 1), axis=0)
    assert np.array_equal(_host(cube.contains_points(points)), expected)
    assert cube.contains((0.5, 0.5, 0.5)) and not cube.contains((2.0, 0.5, 0.5))
    start = rng.uniform(0.2, 0.8, (3, 500))
    end = start + np.array([[3.0], [0.1], [-0.05]])
    found = cube.crossings(start, end)
    assert found.exact and np.all(_host(found.hit))
    expected_t = (1.0 - start[0]) / 3.0
    assert np.allclose(_host(found.fraction), expected_t, atol=1e-14)
    assert np.allclose(_host(found.normal), [[1.0], [0.0], [0.0]])


def test_reversed_faces_are_reoriented():
    cube = TriangleMesh(CUBE_VERTICES, CUBE_FACES[:, ::-1])
    assert cube.volume() > 0
    assert np.allclose(cube.face_normals[0], [-1.0, 0.0, 0.0])


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_inside_kernel_matches_the_sphere(sphere, implementation):
    rng = np.random.default_rng(1)
    points = rng.uniform(-1.3, 1.3, (3, 400 if implementation == "python" else 20000))
    with xp.kernels.use_host_kernel_implementation(implementation):
        inside = _host(sphere.contains_points(points))
    radius = np.linalg.norm(points, axis=0)
    # the mesh lies inside the sphere, within its sagitta of it
    sagitta = 1.0 - np.cos(np.pi / 12)
    assert np.all(inside[radius < 1.0 - sagitta])
    assert not np.any(inside[radius > 1.0])
    # in the band, compare with the faces themselves: inside iff below the plane hit
    band = (radius >= 1.0 - sagitta) & (radius <= 1.0)
    outward = points[:, band] * (2.0 / radius[band])
    hits = sphere.first_hit(points[:, band].T, outward.T)
    assert np.array_equal(inside[band], hits.hit)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_first_hit_kernel_matches_the_sphere(sphere, implementation):
    rng = np.random.default_rng(2)
    n = 300 if implementation == "python" else 5000
    start = rng.normal(size=(3, n))
    start *= 2.0 / np.linalg.norm(start, axis=0)
    end = rng.uniform(-0.3, 0.3, (3, n))
    with xp.kernels.use_host_kernel_implementation(implementation):
        found = sphere.crossings(start, end)
    fraction = _host(found.fraction)
    exact = _sphere_crossing(start, end)
    step = np.linalg.norm(end - start, axis=0)
    sagitta = 1.0 - np.cos(np.pi / 12)
    assert np.all(_host(found.hit))
    assert np.max((fraction - exact) * step) <= sagitta + 1e-12
    point = _host(found.point)
    normal = _host(found.normal)
    assert np.all(np.sum(normal * point, axis=0) > 0.9)  # outward


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS[1:])
def test_kernels_agree_with_numpy(sphere, implementation):
    rng = np.random.default_rng(3)
    start = rng.uniform(-1.5, 1.5, (3, 300))
    end = rng.uniform(-1.5, 1.5, (3, 300))
    end[:, :20] = start[:, :20]  # zero steps
    end[0, 20:40] = start[0, 20:40]  # steps parallel to the grid planes
    end[1, 20:40] = start[1, 20:40]
    start[:, 40:50] += 10.0  # far outside the grid
    results = {}
    for name in (implementation, "numpy"):
        with xp.kernels.use_host_kernel_implementation(name):
            results[name] = (
                sphere._hits(start, end, 1e-12),
                _host(sphere.contains_points(start)),
            )
    (t_a, f_a), in_a = results[implementation]
    (t_b, f_b), in_b = results["numpy"]
    assert np.array_equal(f_a >= 0, f_b >= 0)
    assert np.allclose(t_a, t_b, rtol=0, atol=1e-12)
    assert np.array_equal(in_a, in_b)
    assert np.all(f_a[:20] == -1)


def test_grid_cells_and_direct_kernel_call(sphere):
    coarse = uv_sphere(1.0, 12, 24)
    assert coarse.grid.faces_per_cell > 0
    single = TriangleMesh(coarse.vertices, coarse.faces, cells=1)
    assert tuple(single.grid.cells) == (1, 1, 1)
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]).T.copy()
    assert list(_host(single.contains_points(points))) == [True, False]
    grid = sphere.grid
    inside = np.zeros(2, dtype=bool)
    mesh_inside(
        points,
        sphere.vertices,
        sphere.faces,
        grid.lower,
        grid.spacing,
        grid.cells,
        grid.cell_start,
        grid.cell_faces,
        grid.node_inside,
        inside,
    )
    assert list(inside) == [True, False]
    fraction, face = np.zeros(1), np.zeros(1, dtype=np.int64)
    mesh_first_hits(
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [3.0]]),
        sphere.vertices,
        sphere.faces,
        grid.lower,
        grid.spacing,
        grid.cells,
        grid.cell_start,
        grid.cell_faces,
        1e-12,
        fraction,
        face,
    )
    assert face[0] >= 0 and 0.3 < fraction[0] < 0.34
    with pytest.raises(ValueError, match="positive"):
        TriangleMesh(coarse.vertices, coarse.faces, cells=0)


def test_open_mesh_is_a_surface_without_inside():
    square = TriangleMesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2], [0, 2, 3]],
    )
    assert not square.closed
    assert not np.any(_host(square.contains_points(np.full((3, 5), 0.5))))
    with pytest.raises(ValueError, match="no volume"):
        square.volume()
    assert np.isclose(square.surface_area(), 1.0)
    start = np.array([[0.5, 0.5, 1.0], [0.5, 0.5, -1.0], [2.0, 2.0, 1.0]])
    end = start * np.array([1.0, 1.0, -1.0])
    hits = square.first_hit(start, end)
    assert list(hits.hit) == [True, True, False]
    assert np.allclose(hits.normal[0], [0, 0, 1]) and np.allclose(
        hits.normal[1], [0, 0, -1]
    )
    assert np.allclose(hits.distance[:2], 1.0) and np.all(hits.face[:2] >= 0)
    assert hits.face[2] == -1 and np.all(np.isnan(hits.point[2]))
    # a segment starting on the surface leaves it
    leaving = square.first_hit([[0.5, 0.5, 0.0]], [[0.5, 0.5, 1.0]])
    assert not leaving.hit[0]
    inverted = TriangleMesh(square.vertices, square.faces, inverted=True)
    assert np.all(_host(inverted.contains_points(np.full((3, 2), 0.5))))


def test_closed_requests_are_checked():
    with pytest.raises(ValueError, match="not closed"):
        TriangleMesh(CUBE_VERTICES, CUBE_FACES[:-1], closed=True)
    flipped = CUBE_FACES.copy()
    flipped[0] = flipped[0, ::-1]
    with pytest.raises(ValueError, match="consistently"):
        TriangleMesh(CUBE_VERTICES, flipped, closed=True)
    assert not TriangleMesh(CUBE_VERTICES, flipped).closed
    assert not TriangleMesh(CUBE_VERTICES, CUBE_FACES, closed=False).closed


@pytest.mark.parametrize(
    ("vertices", "faces", "match"),
    [
        (np.zeros((2, 3)), [[0, 1, 1]], "vertices"),
        (CUBE_VERTICES, np.zeros((0, 3), dtype=int), "faces"),
        (np.full((3, 3), np.nan), [[0, 1, 2]], "finite"),
        (CUBE_VERTICES, [[0, 1, 99]], "do not exist"),
    ],
)
def test_invalid_meshes(vertices, faces, match):
    with pytest.raises(ValueError, match=match):
        TriangleMesh(vertices, faces)


def test_inverted_mesh_and_point_major_queries():
    cube = TriangleMesh(CUBE_VERTICES, CUBE_FACES, inverted=True)
    points = np.array([[[0.5, 0.5, 0.5], [2.0, 0.5, 0.5]]])
    assert _host(cube.inside_at(points)).tolist() == [[False, True]]
    found = cube.crossings(
        np.array([[0.5], [0.5], [0.5]]), np.array([[2.0], [0.5], [0.5]])
    )
    assert np.allclose(_host(found.normal)[:, 0], [-1.0, 0.0, 0.0])
    hits = cube.first_hit(points, points + [3.0, 0.0, 0.0])
    assert hits.hit.shape == (1, 2) and hits.point.shape == (1, 2, 3)
    assert np.allclose(hits.normal[0, 0], [-1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="same shape"):
        cube.first_hit(points, points[:, :1])
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        cube.inside_at(np.zeros((4, 2)))
    copy = cube.copy()
    assert copy.inverted and copy.n_faces == 12


def test_collections_and_transforms_use_the_kernel():
    cube = TriangleMesh(CUBE_VERTICES, CUBE_FACES)
    moved = cube.translated((3.0, 0.0, 0.0))
    walls = GeometryCollection({"near": cube, "far": moved})
    start = np.array([[-1.0, 1.5], [0.5, 0.5], [0.5, 0.5]])
    end = np.array([[5.0, 5.0], [0.5, 0.5], [0.5, 0.5]])
    found = walls.crossings(start, end)
    assert found.exact
    assert _host(found.surface).tolist() == [0, 1]
    assert np.allclose(_host(found.fraction), [1.0 / 6.0, 1.5 / 3.5])
    assert found.per_surface() == {"near": 1.0, "far": 1.0}


def test_read_and_write(tmp_path, monkeypatch):
    cube = TriangleMesh(CUBE_VERTICES, CUBE_FACES)
    for suffix in (".stl", ".obj", ".npz"):
        path = cube.save(tmp_path / f"cube{suffix}")
        again = TriangleMesh.read(path)
        assert again.closed and np.isclose(again.volume(), 1.0), suffix
    # a binary STL, and an OBJ with quads and texture indices
    record = np.dtype(
        [("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")]
    )
    facets = np.zeros(12, dtype=record)
    facets["corners"] = CUBE_VERTICES[CUBE_FACES]
    header = b"binary".ljust(80) + np.uint32(12).tobytes()
    (tmp_path / "binary.stl").write_bytes(header + facets.tobytes())
    assert np.isclose(TriangleMesh.read(tmp_path / "binary.stl").volume(), 1.0)
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\nf 1/1 2/2 3/3 4/4\n"
    (tmp_path / "quad.obj").write_text(obj)
    quad = TriangleMesh.read(tmp_path / "quad.obj")
    assert quad.n_faces == 2 and not quad.closed
    soup = TriangleMesh.read(tmp_path / "binary.stl", merge_tolerance=0)
    assert not soup.closed and soup.vertices.shape[0] == 36
    # other formats through meshio
    monkeypatch.setitem(sys.modules, "meshio", None)
    with pytest.raises(ImportError, match="meshio"):
        TriangleMesh.read(tmp_path / "cube.vtk")
    fake = types.SimpleNamespace(
        read=lambda path: types.SimpleNamespace(
            points=CUBE_VERTICES,
            cells=[
                types.SimpleNamespace(type="line", data=np.zeros((1, 2), int)),
                types.SimpleNamespace(type="triangle", data=CUBE_FACES),
            ],
        )
    )
    monkeypatch.setitem(sys.modules, "meshio", fake)
    assert np.isclose(TriangleMesh.read(tmp_path / "cube.vtk").volume(), 1.0)


def test_to_mesh_and_edge_length():
    cube = TriangleMesh(CUBE_VERTICES, CUBE_FACES)
    mesh = cube.to_mesh(resolution=8)
    assert mesh.cells.shape == (12, 3) and np.allclose(mesh.vertices, CUBE_VERTICES)
    assert 1.0 < cube.edge_length < np.sqrt(2.0)
    assert np.allclose(cube.face_areas, 0.5)


def test_node_states_count_every_line_evenly(sphere):
    from simgeometry._triangle_grid import _node_states_along

    grid = sphere.grid
    for axis in range(3):
        _, odd = _node_states_along(
            sphere.vertices, sphere.faces, grid.lower, grid.spacing, grid.cells, axis
        )
        assert not odd.any()
