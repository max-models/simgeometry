"""SlantedTube: membership, volume, bounds, level set, crossings, conductors and views."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry import GeometryCollection, Grid, SlantedTube, Tube
from simgeometry.crossings import crossings

RNG = np.random.default_rng(11)
C45 = np.cos(np.pi / 4)

SHAPES = {
    "nozzle z": SlantedTube((0, 0, 0), (0, 0, 1.0), 0.2, 0.3, (0, -C45, C45)),
    "rod oblique": SlantedTube(
        (0.1, -0.2, 0.0), (0.7, 0.2, 0.4), 0.0, 0.25, (0.6, 0.0, 0.8)
    ),
    "tube steep": SlantedTube(
        (0, 0, 0), (1.0, 0, 0), 0.1, 0.35, (np.cos(1.1), np.sin(1.1), 0.0)
    ),
}


def _box_points(shape, n, pad=0.0):
    lower = np.asarray(shape.lower_bound) - pad
    upper = np.asarray(shape.upper_bound) + pad
    return RNG.uniform(lower[:, None], upper[:, None], (3, n)), np.prod(upper - lower)


def _frame(shape):
    axis = np.asarray(shape.axis)
    u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    u /= np.linalg.norm(u)
    return axis, u, np.cross(axis, u)


def _surface(shape, spacing):
    """Dense points on the four faces: both walls, the flat end and the cut."""
    axis, u, v = _frame(shape)
    start, normal = np.asarray(shape.start), np.asarray(shape.cut_normal)
    end = np.asarray(shape.end)
    ri, ro = shape.inner_radius, shape.outer_radius
    phi = np.linspace(0, 2 * np.pi, int(2 * np.pi * ro / spacing) + 1, endpoint=False)
    ring = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v  # (m, 3)
    # axial position of the cut at each azimuth, for a point at radius r
    cut_s = lambda r: shape.length - r * (ring @ normal) / (normal @ axis)  # noqa: E731
    points = []
    for r in [ro] + ([ri] if ri > 0 else []):
        top = cut_s(r)
        for k in range(len(phi)):
            s = np.linspace(0, top[k], int(top[k] / spacing) + 2)
            points.append(start + s[:, None] * axis + r * ring[k])
    radii = np.linspace(ri, ro, max(2, int((ro - ri) / spacing) + 1))
    for r in radii:
        points.append(start + r * ring)  # flat end
        points.append(start + cut_s(r)[:, None] * axis + r * ring)  # cut
    surface = np.concatenate(points)
    # every point on the cut lies on its plane
    assert np.allclose((surface - end) @ normal <= 1e-12, True)
    return surface


@pytest.mark.parametrize("name", SHAPES)
def test_contains_and_level_set_signs_agree(name):
    shape = SHAPES[name]
    points, _ = _box_points(shape, 20000, pad=0.2)
    values = shape.level_set(points)
    inside = shape.contains_points(points)
    clear = np.abs(values) > 1e-12
    np.testing.assert_array_equal((values < 0)[clear], inside[clear])
    assert inside.any() and not inside.all()
    assert shape.has_level_set and not shape.level_set_exact
    for point in points[:, :20].T:
        assert shape.contains(point) == bool(shape.contains_points(point[:, None])[0])


@pytest.mark.parametrize("name", SHAPES)
def test_monte_carlo_volume(name):
    shape = SHAPES[name]
    n = 400_000
    points, box_volume = _box_points(shape, n)
    fraction = float(np.mean(shape.contains_points(points)))
    sigma = box_volume * np.sqrt(fraction * (1 - fraction) / n)
    assert fraction * box_volume == pytest.approx(shape.volume(), abs=4 * sigma)


@pytest.mark.parametrize("name", SHAPES)
def test_bounding_box_is_tight(name):
    shape = SHAPES[name]
    surface = _surface(shape, spacing=0.005)
    np.testing.assert_allclose(surface.min(axis=0), shape.lower_bound, atol=2e-3)
    np.testing.assert_allclose(surface.max(axis=0), shape.upper_bound, atol=2e-3)


@pytest.mark.parametrize("name", SHAPES)
def test_level_set_is_zero_on_the_surface_and_bounds_the_distance(name):
    spatial = pytest.importorskip("scipy.spatial")
    shape = SHAPES[name]
    spacing = 0.005
    surface = _surface(shape, spacing)
    np.testing.assert_allclose(shape.level_set(surface.T), 0.0, atol=1e-12)
    points, _ = _box_points(shape, 2000, pad=0.3)
    distance, _ = spatial.cKDTree(surface).query(points.T)
    values = np.abs(shape.level_set(points))
    # a bound: never more than the distance (up to the sampling of the surface)
    assert np.all(values <= distance + 1e-12)
    # and the distance itself for most points (away from the rim of the cut)
    close = np.abs(distance - values) <= spacing
    assert close.mean() > 0.6


def test_square_cut_is_the_tube():
    """With the cut normal along the axis the shape is a :class:`Tube`."""
    slanted = SlantedTube((0, 0, -0.5), (0, 0, 0.5), 0.3, 0.6, (0, 0, 2.0))
    tube = Tube((0, 0, -0.5), (0, 0, 0.5), 0.3, 0.6)
    assert slanted.cut_angle == pytest.approx(0.0)
    points, _ = _box_points(tube, 20000, pad=0.3)
    np.testing.assert_array_equal(
        slanted.contains_points(points), tube.contains_points(points)
    )
    inside = tube.contains_points(points)
    np.testing.assert_allclose(
        slanted.level_set(points)[inside], tube.level_set(points)[inside], atol=1e-12
    )
    np.testing.assert_allclose(slanted.lower_bound, tube.lower_bound, atol=1e-12)
    np.testing.assert_allclose(slanted.upper_bound, tube.upper_bound, atol=1e-12)
    assert slanted.volume() == pytest.approx(tube.volume())
    assert slanted.surface_area() == pytest.approx(tube.surface_area())


def test_surface_areas():
    pytest.importorskip("skimage")
    for name, shape in SHAPES.items():
        mesh = shape.to_mesh(resolution=128)
        faces = mesh.vertices[mesh.cells]
        area = 0.5 * np.linalg.norm(
            np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
        )
        assert area.sum() == pytest.approx(shape.surface_area(), rel=0.03), name


def _brute_force_first(shape, a, b, samples=40001):
    """First membership change along each segment, from dense samples."""
    t = np.linspace(0, 1, samples)
    first = np.full(a.shape[1], np.inf)
    for i in range(a.shape[1]):
        points = a[:, i, None] + t[None, :] * (b - a)[:, i, None]
        inside = shape.contains_points(points)
        change = np.flatnonzero(inside[1:] != inside[:-1])
        if change.size:
            first[i] = t[change[0] + 1]
    return first


@pytest.mark.parametrize("name", SHAPES)
@pytest.mark.parametrize("inverted", [False, True])
def test_analytic_crossings_match_dense_sampling(name, inverted):
    shape = SHAPES[name].copy()
    shape._inverted = inverted
    a, _ = _box_points(shape, 300, pad=0.2)
    b, _ = _box_points(shape, 300, pad=0.2)
    result = crossings(shape, a, b)
    assert result.exact
    reference = _brute_force_first(shape, a, b)
    np.testing.assert_array_equal(result.hit, np.isfinite(reference))
    assert result.hit.sum() > 100
    hit = result.hit
    step = np.linalg.norm(b - a, axis=0)
    np.testing.assert_array_less(
        np.abs(result.fraction[hit] - reference[hit]) * step[hit], 1e-4
    )
    # on the boundary, the normal out of the contains-region
    points = result.point[:, hit]
    np.testing.assert_allclose(shape.level_set(points), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(result.normal[:, hit], axis=0), 1.0)
    probe = points + 1e-6 * result.normal[:, hit]
    assert not shape.contains_points(probe).any()
    assert shape.contains_points(points - 1e-6 * result.normal[:, hit]).all()


def test_crossings_of_a_bore_and_a_rotated_copy():
    """A step along the bore leaves through the cut's opening; rotated stays analytic."""
    nozzle = SHAPES["nozzle z"]
    # along the axis inside the bore: no crossing, the bore is open at both ends
    a = np.array([[0.0], [0.0], [-0.5]])
    b = np.array([[0.0], [0.0], [2.0]])
    assert not crossings(nozzle, a, b).hit[0]
    # from the bore radially out through the wall, below the cut's lowest point
    a = np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.6]])
    b = np.array([[0.0, 0.0], [1.0, -1.0], [0.3, 0.6]])
    result = crossings(nozzle, a, b)
    np.testing.assert_allclose(result.fraction, [0.2, 0.2])
    np.testing.assert_allclose(result.normal[:, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(result.normal[:, 1], [0, 1, 0], atol=1e-12)
    # into the cut face from outside, along -normal
    normal = np.asarray(nozzle.cut_normal)
    rim_point = np.asarray(nozzle.end) + 0.25 * np.array([1.0, 0, 0])
    a = (rim_point + 0.1 * normal)[:, None]
    b = (rim_point - 0.1 * normal)[:, None]
    result = crossings(nozzle, a, b)
    assert result.hit[0] and result.fraction[0] == pytest.approx(0.5)
    np.testing.assert_allclose(result.normal[:, 0], normal, atol=1e-12)
    rotated = nozzle.rotated(30, axis=(1, 0, 0))
    ends_a, _ = _box_points(rotated, 200, pad=0.2)
    ends_b, _ = _box_points(rotated, 200, pad=0.2)
    result = crossings(rotated, ends_a, ends_b)
    assert result.exact
    np.testing.assert_array_equal(
        result.hit, np.isfinite(_brute_force_first(rotated, ends_a, ends_b, 8001))
    )


def test_constructor_checks_and_properties():
    nozzle = SlantedTube((0, 0, 0), (0, 0, 2), 0.1, 0.2, (0, -1, 1))
    assert nozzle.length == 2 and nozzle.axis == (0, 0, 1)
    assert nozzle.cut_angle == pytest.approx(45.0)
    np.testing.assert_allclose(nozzle.cut_normal, (0, -C45, C45))
    assert nozzle.end == (0, 0, 2) and nozzle.start == (0, 0, 0)
    with pytest.raises(ValueError):
        SlantedTube((0, 0, 0), (0, 0, 0), 0.1, 0.2, (0, 0, 1))
    with pytest.raises(ValueError):
        SlantedTube((0, 0, 0), (0, 0, 1), 0.3, 0.2, (0, 0, 1))
    with pytest.raises(ValueError):
        SlantedTube((0, 0, 0), (0, 0, 1), 0.1, 0.2, (0, 1, 0))  # cut along the axis
    with pytest.raises(ValueError):
        SlantedTube((0, 0, 0), (0, 0, 1), 0.1, 0.2, (0, 0, -1))
    with pytest.raises(ValueError):
        SlantedTube((0, 0, 0), (0, 0, 0.1), 0.1, 0.2, (0, -1, 1))  # cut too long
    with pytest.raises(ValueError):
        SlantedTube((0, 0, 0), (0, 0, 1), 0.1, 0.2, (0, 0, 0))


def test_copy_translate_mirror_and_inversion():
    nozzle = SlantedTube((0, 0, 0), (0, 0, 1), 0.1, 0.2, (0, -1, 1), bc=2.0)
    # below the axis the duct is shorter than above it
    assert nozzle.contains((0, 0.15, 1.1)) and not nozzle.contains((0, -0.15, 0.9))
    moved = nozzle.copy()
    moved.translate((1, 0, 0))
    assert moved.contains((1, 0.15, 1.1)) and not moved.contains((0, 0.15, 1.1))
    assert nozzle.contains((0, 0.15, 1.1)) and moved._bc == 2.0
    np.testing.assert_allclose(moved.lower_bound, np.add(nozzle.lower_bound, (1, 0, 0)))
    flipped = nozzle.copy()
    flipped.mirror(1, 0.0)
    assert flipped.contains((0, -0.15, 1.1)) and not flipped.contains((0, 0.15, 0.9))
    np.testing.assert_allclose(flipped.cut_normal, (0, C45, C45))
    np.testing.assert_allclose(flipped.upper_bound[1], -nozzle.lower_bound[1])
    hole = SlantedTube((0, 0, 0), (0, 0, 1), 0.1, 0.2, (0, -1, 1), inverted=True)
    assert hole.contains((0, 0, 0.5)) and not hole.contains((0, 0.15, 0.5))
    assert hole.level_set(np.array([[0.0], [0.15], [0.5]]))[0] == pytest.approx(0.05)


def test_grids_conductors_and_views():
    nozzle = SlantedTube((0, 0, -0.5), (0, 0, 0.5), 0.2, 0.35, (0, -1, 1))
    grid = Grid(
        ndim=3,
        lower_bound=(-0.6, -0.6, -0.8),
        upper_bound=(0.6, 0.6, 1.0),
        num_gridpoints=(13, 13, 19),
    )
    np.testing.assert_array_equal(
        nozzle.node_mask(grid).ravel(), nozzle.contains_points(grid.node_points())
    )
    oven = GeometryCollection(
        {"nozzle": nozzle, "wall": Tube((0, 0, -0.8), (0, 0, -0.6), 0.1, 0.6)},
        voltages={"nozzle": 5.0, "wall": 0.0},
    )
    assert oven.conductors == ("nozzle", "wall")
    values = oven.dirichlet_values(grid.node_points())
    inside = nozzle.contains_points(grid.node_points())
    assert inside.any()
    np.testing.assert_allclose(values.voltages[inside], 5.0)
    samples = nozzle.sample_interior(200, rng=1)
    assert nozzle.contains_points(samples).all()


def test_show_3d_and_mesh():
    pytest.importorskip("matplotlib")
    pytest.importorskip("skimage")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nozzle = SHAPES["nozzle z"]
    nozzle.show_3d(resolution=32, display=False)
    nozzle.slice(x=0).show(resolution=64, display=False)
    surface = nozzle.sample_surface(100, rng=2, resolution=64)
    assert np.max(np.abs(nozzle.level_set(surface))) < 0.02
    plt.close("all")
