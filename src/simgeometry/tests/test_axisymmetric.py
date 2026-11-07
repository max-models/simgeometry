"""Cylinder, Tube, Cone and Torus: membership, volumes, bounds and exact level sets."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry import Cone, Cylinder, Grid, Torus, Tube

RNG = np.random.default_rng(7)
OBLIQUE = np.array([1.0, -2.0, 2.0]) / 3.0

SHAPES = {
    "cylinder z": Cylinder((0, 0, -0.5), (0, 0, 0.7), 0.4),
    "cylinder oblique": Cylinder(
        (0.1, 0.2, 0.3), np.add((0.1, 0.2, 0.3), OBLIQUE), 0.3
    ),
    "cylinder fillet": Cylinder((0, 0, -0.5), (0, 0.5, 0.5), 0.4, fillet=0.15),
    "tube": Tube((0, 0, -0.4), (0, 0, 0.4), 0.3, 0.6),
    "tube oblique fillet": Tube((0.2, 0, 0), (0.2, 0.6, 0.8), 0.2, 0.5, fillet=0.1),
    "frustum": Cone((0, 0, 0), (0, 0, 1), 0.6, 0.2),
    "cone oblique": Cone((0, 0, 0), tuple(1.2 * OBLIQUE), 0.0, 0.5),
    "torus": Torus((0, 0, 0), 0.8, 0.25),
    "torus oblique": Torus((0.1, 0.0, -0.2), 0.6, 0.3, axis=OBLIQUE),
}


def _box_points(shape, n, pad=0.0):
    lower = np.asarray(shape.lower_bound) - pad
    upper = np.asarray(shape.upper_bound) + pad
    return RNG.uniform(lower[:, None], upper[:, None], (3, n)), np.prod(upper - lower)


def _profile(shape, spacing):
    """Dense ``(rho, s)`` points on the profile of the solid, in its own frame."""

    def segment(a, b):
        n = max(2, int(np.hypot(*np.subtract(b, a)) / spacing) + 1)
        t = np.linspace(0, 1, n)
        return np.outer(1 - t, a) + np.outer(t, b)

    def arc(centre, radius, start, stop):
        n = max(2, int(radius * abs(stop - start) / spacing) + 1)
        angle = np.linspace(start, stop, n)
        return np.column_stack(
            (centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle))
        )

    if isinstance(shape, Torus):
        return arc((shape.major_radius, 0.0), shape.minor_radius, 0, 2 * np.pi)
    h = shape.length / 2
    if isinstance(shape, Cone):
        r0, r1 = shape.start_radius, shape.end_radius
        return np.concatenate(
            [
                segment((0, -h), (r0, -h)),
                segment((r0, -h), (r1, h)),
                segment((r1, h), (0, h)),
            ]
        )
    f = shape.fillet
    inner = shape.inner_radius if isinstance(shape, Tube) else 0.0
    outer = shape.outer_radius if isinstance(shape, Tube) else shape.radius
    pieces = [
        segment((outer, -h + f), (outer, h - f)),
        segment((outer - f, h), (inner + (f if inner else 0), h)),
        segment((inner + (f if inner else 0), -h), (outer - f, -h)),
    ]
    if f:
        pieces += [
            arc((outer - f, h - f), f, 0, np.pi / 2),
            arc((outer - f, -h + f), f, -np.pi / 2, 0),
        ]
    if inner:
        pieces.append(segment((inner, h - f), (inner, -h + f)))
        if f:
            pieces += [
                arc((inner + f, h - f), f, np.pi / 2, np.pi),
                arc((inner + f, -h + f), f, np.pi, 1.5 * np.pi),
            ]
    return np.concatenate(pieces)


def _surface(shape, spacing):
    """Points on the surface: the profile revolved about the axis."""
    profile = _profile(shape, spacing)
    axis = np.asarray(shape.axis)
    u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    rho, s = profile[:, 0], profile[:, 1]
    count = int(2 * np.pi * max(rho.max(), spacing) / spacing) + 1
    theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
    ring = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v  # (count, 3)
    blocks = (
        np.asarray(shape.midpoint)
        + s[:, None, None] * axis
        + rho[:, None, None] * ring[None, :, :]
    )
    return blocks.reshape(-1, 3)


@pytest.mark.parametrize("name", SHAPES)
def test_contains_and_level_set_signs_agree(name):
    shape = SHAPES[name]
    points, _ = _box_points(shape, 20000, pad=0.2)
    values = shape.level_set(points)
    inside = shape.contains_points(points)
    clear = np.abs(values) > 1e-12
    np.testing.assert_array_equal((values < 0)[clear], inside[clear])
    assert inside.any() and not inside.all()
    assert shape.has_level_set and shape.level_set_exact
    # contains agrees with contains_points
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
    surface = _surface(shape, spacing=0.01)
    np.testing.assert_allclose(surface.min(axis=0), shape.lower_bound, atol=2e-3)
    np.testing.assert_allclose(surface.max(axis=0), shape.upper_bound, atol=2e-3)


@pytest.mark.parametrize("name", SHAPES)
def test_level_set_is_the_exact_distance(name):
    spatial = pytest.importorskip("scipy.spatial")
    shape = SHAPES[name]
    spacing = 0.006
    surface = _surface(shape, spacing)
    np.testing.assert_allclose(shape.level_set(surface.T), 0.0, atol=1e-12)
    points, _ = _box_points(shape, 400, pad=0.3)
    distance, _ = spatial.cKDTree(surface).query(points.T)
    values = np.abs(shape.level_set(points))
    # the sampled surface overestimates the distance by at most half a sample gap
    assert np.all(values <= distance + 1e-12)
    assert np.all(distance - values <= spacing)


def test_surface_areas():
    pytest.importorskip("skimage")
    for name, shape in SHAPES.items():
        mesh = shape.to_mesh(resolution=96)
        faces = mesh.vertices[mesh.cells]
        area = 0.5 * np.linalg.norm(
            np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
        )
        assert area.sum() == pytest.approx(shape.surface_area(), rel=0.03), name


def test_constructor_checks_and_properties():
    cylinder = Cylinder((0, 0, 0), (0, 0, 2), 0.5)
    assert cylinder.length == 2 and cylinder.midpoint == (0, 0, 1)
    assert cylinder.axis == (0, 0, 1) and cylinder.start == (0, 0, 0)
    assert cylinder.volume() == pytest.approx(np.pi * 0.25 * 2)
    with pytest.raises(ValueError):
        Cylinder((0, 0, 0), (0, 0, 0), 1)
    with pytest.raises(ValueError):
        Cylinder((0, 0, 0), (0, 0, 1), -1)
    with pytest.raises(ValueError):
        Cylinder((0, 0, 0), (0, 0, 1), 1, fillet=0.6)
    with pytest.raises(ValueError):
        Tube((0, 0, 0), (0, 0, 1), 0.5, 0.4)
    with pytest.raises(ValueError):
        Tube((0, 0, 0), (0, 0, 1), 0.4, 0.5, fillet=0.06)
    with pytest.raises(ValueError):
        Cone((0, 0, 0), (0, 0, 1), 0, 0)
    with pytest.raises(ValueError):
        Torus((0, 0, 0), 1, 1.5)
    with pytest.raises(ValueError):
        Torus((0, 0, 0), 1, 0.5, axis=(0, 0, 0))


def test_copy_translate_mirror_and_inversion():
    cone = Cone((0, 0, 0), (0, 0, 1), 0.5, 0.1, bc=3.0)
    moved = cone.copy()
    moved.translate((1, 0, 0))
    assert moved.contains((1, 0, 0.1)) and not moved.contains((0, 0, 0.1))
    assert cone.contains((0, 0, 0.1)) and moved._bc == 3.0
    np.testing.assert_allclose(moved.lower_bound, np.add(cone.lower_bound, (1, 0, 0)))
    mirrored = cone.copy()
    mirrored.mirror(2, 0.0)
    assert mirrored.contains((0, 0, -0.1)) and mirrored.contains((0.45, 0, -0.05))
    assert not mirrored.contains((0.45, 0, -0.95))
    np.testing.assert_allclose(mirrored.end, (0, 0, -1))
    hole = Torus((0, 0, 0), 1, 0.2, inverted=True)
    assert not hole.contains((1, 0, 0)) and hole.contains((0, 0, 0))
    assert hole.level_set(np.array([[1.0], [0.0], [0.0]]))[0] == pytest.approx(0.2)


def test_booleans_transforms_grids_and_sampling():
    tube = Tube((0, 0, -0.3), (0, 0, 0.3), 0.3, 0.5)
    rod = Cylinder((-1, 0, 0), (1, 0, 0), 0.1)
    cut = tube - rod
    assert cut.has_level_set and not cut.level_set_exact
    assert not cut.contains((0.4, 0, 0)) and cut.contains((0, 0.4, 0))
    rotated = tube.rotated(90, axis=(1, 0, 0))
    assert rotated.level_set_exact
    reference = Tube((0, 0.3, 0), (0, -0.3, 0), 0.3, 0.5)
    points, _ = _box_points(reference, 500, pad=0.2)
    np.testing.assert_allclose(
        rotated.level_set(points), reference.level_set(points), atol=1e-12
    )
    np.testing.assert_allclose(rotated.lower_bound, reference.lower_bound, atol=1e-12)
    grid = Grid(
        ndim=3,
        lower_bound=(-0.6, -0.6, -0.6),
        upper_bound=(0.6, 0.6, 0.6),
        num_gridpoints=(13, 13, 13),
    )
    nodes = grid.node_points()
    np.testing.assert_array_equal(
        tube.node_mask(grid).ravel(), tube.contains_points(nodes)
    )
    samples = tube.sample_interior(200, rng=1)
    assert tube.contains_points(samples).all()


def test_plotting_and_meshes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("skimage")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tube = Tube((0, 0, -0.3), (0, 0, 0.3), 0.3, 0.5, fillet=0.05)
    tube.show_3d(resolution=24, display=False)
    tube.slice(x=0).show(resolution=64, display=False)
    surface = tube.sample_surface(100, rng=2, resolution=48)
    assert np.max(np.abs(tube.level_set(surface))) < 0.01
    plt.close("all")
