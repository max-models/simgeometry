"""Level sets: sign convention, exactness, propagation and their use in queries."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry import (
    Aperture2D,
    Circle,
    Cuboid,
    Cylinder,
    FunctionObject2D,
    GeometryCollection,
    Grid,
    Line,
    Polygon,
    Rectangle,
    Sphere,
    Torus,
    Triangle,
    Tube,
)
from simgeometry.cut_edges import bracketed_edge_geometry
from simgeometry.queries import intersection_fractions
from simgeometry.shapes import Shape

RNG = np.random.default_rng(1)


class MembershipOnly(Shape):
    """A shape that hides the level set of its source, for the fallback paths."""

    def __init__(self, source):
        self.source = source
        self.evaluated = 0
        super().__init__(source.ndim, source.lower_bound, source.upper_bound)

    def contains_points(self, points):
        self.evaluated += np.shape(points)[1]
        return self.source.contains_points(points)

    def contains(self, coords):
        return self.source.contains(coords)


def _points(shape, n=2000, pad=0.3):
    lower = np.asarray(shape.lower_bound, float) - pad
    upper = np.asarray(shape.upper_bound, float) + pad
    return RNG.uniform(lower[:, None], upper[:, None], (shape.ndim, n))


PRIMITIVES = [
    Line((0.5,), (1.0,)),
    Circle((0.2, -0.1), 0.7),
    Rectangle((0.0, 0.5), (1.0, 0.4)),
    Polygon(((0, 0), (2, 0), (2, 1), (1, 0.4), (0, 1))),
    Triangle(((0, 0), (1, 0), (0.2, 0.8))),
    Aperture2D.slit(0.0, 0.2, 0.1, 0.5),
    Sphere((0.1, 0.2, 0.3), 0.8),
    Cuboid((0, 0, 0), (1.0, 0.5, 2.0)),
]


@pytest.mark.parametrize("shape", PRIMITIVES, ids=lambda s: type(s).__name__)
@pytest.mark.parametrize("inverted", [False, True])
def test_level_set_sign_agrees_with_contains(shape, inverted):
    shape = shape.copy()
    shape._inverted = inverted
    points = _points(shape)
    values = shape.level_set(points)
    inside = shape.contains_points(points)
    clear = np.abs(values) > 1e-12
    np.testing.assert_array_equal((values < 0)[clear], inside[clear])
    assert shape.has_level_set and shape.level_set_exact


def test_level_set_matches_signed_distance_convention():
    circle = Circle((0, 0), 1)
    points = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(circle.level_set(points), [-1, 1, -0.5])
    np.testing.assert_allclose(circle.signed_distance(points), [-1, 1, -0.5])
    inverted = Circle((0, 0), 1, inverted=True)
    np.testing.assert_allclose(inverted.level_set(points), [1, -1, 0.5])
    np.testing.assert_allclose(inverted.signed_distance(points), [1, -1, 0.5])


def test_box_and_polygon_level_sets_are_the_exact_distance():
    box = Rectangle((0, 0), (2, 1))
    points = np.array([[0, 2, 2, 0.5], [0, 0, 1.5, 0.25]], float)
    np.testing.assert_allclose(box.level_set(points), [-0.5, 1, np.hypot(1, 1), -0.25])
    square = Polygon(((-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)))
    np.testing.assert_allclose(square.level_set(points), box.level_set(points))
    cube = Cuboid((0, 0, 0), (2, 2, 2))
    np.testing.assert_allclose(
        cube.level_set(np.array([[2.0], [2.0], [2.0]])), [3**0.5]
    )
    np.testing.assert_allclose(
        Line((0,), (2,)).level_set(np.array([[0.5, 3.0]])), [-0.5, 2]
    )


def test_shapes_without_a_level_set_say_so():
    ring = FunctionObject2D(lambda p: 1 <= p[0] ** 2 + p[1] ** 2 <= 4, (-2, -2), (2, 2))
    assert not ring.has_level_set
    with pytest.raises(NotImplementedError):
        ring.level_set(np.zeros((2, 1)))
    combined = ring | Circle((0, 0), 1)
    assert not combined.has_level_set
    with pytest.raises(NotImplementedError):
        combined.level_set(np.zeros((2, 1)))


def test_boolean_level_sets_are_min_max_bounds():
    a, b = Circle((0, 0), 1), Circle((1, 0), 1)
    points = _points(a | b)
    for shape, combine in (
        (a | b, np.minimum),
        (a & b, np.maximum),
        (a - b, lambda u, v: np.maximum(u, -v)),
    ):
        assert shape.has_level_set and not shape.level_set_exact
        values = shape.level_set(points)
        np.testing.assert_allclose(
            values, combine(a.level_set(points), b.level_set(points))
        )
        clear = np.abs(values) > 1e-12
        np.testing.assert_array_equal(
            (values < 0)[clear], shape.contains_points(points)[clear]
        )
    # outside a union the minimum is still the exact distance
    np.testing.assert_allclose(
        (a | b).level_set(np.array([[0.5], [3.0]])), [np.hypot(0.5, 3) - 1]
    )


def test_transformed_level_sets():
    circle = Circle((0, 0), 1)
    points = _points(circle, pad=2)
    moved = circle.rotated(30).translated((1, 2))
    assert moved.level_set_exact
    np.testing.assert_allclose(
        moved.level_set(points), Circle((1, 2), 1).level_set(points), atol=1e-14
    )
    scaled = circle.scaled(3.0)
    assert scaled.level_set_exact
    np.testing.assert_allclose(
        scaled.level_set(points), Circle((0, 0), 3).level_set(points)
    )
    stretched = circle.scaled((2.0, 0.5))
    assert stretched.has_level_set and not stretched.level_set_exact
    # a bound on the distance to the ellipse, with the sign of membership
    theta = np.linspace(0, 2 * np.pi, 20001)
    ellipse = np.stack((2 * np.cos(theta), 0.5 * np.sin(theta)))
    distance = np.min(
        np.hypot(points[0][:, None] - ellipse[0], points[1][:, None] - ellipse[1]),
        axis=1,
    )
    values = stretched.level_set(points)
    assert np.all(np.abs(values) <= distance + 1e-6)
    clear = np.abs(values) > 1e-12
    np.testing.assert_array_equal(
        (values < 0)[clear], stretched.contains_points(points)[clear]
    )


def test_sections_and_collections_have_bounded_level_sets():
    sphere = Sphere((0, 0, 0), 1)
    section = sphere.slice(z=0.6)
    assert section.has_level_set and not section.level_set_exact
    np.testing.assert_allclose(section.level_set(np.array([[0.0], [0.0]])), [-0.4])
    collection = GeometryCollection({"a": Circle((0, 0), 1), "b": Circle((3, 0), 1)})
    assert collection.has_level_set
    np.testing.assert_allclose(collection.level_set(np.array([[1.5], [0.0]])), [0.5])
    mixed = GeometryCollection(
        {"a": Circle((0, 0), 1), "f": FunctionObject2D(lambda p: False, (0, 0), (1, 1))}
    )
    assert not mixed.has_level_set


def test_circle_cut_positions_match_the_analytic_crossings():
    centre, radius = np.array([0.5, 0.5]), 0.3
    circle = Circle(centre, radius)
    grid = Grid(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=(64, 64))
    geometry = bracketed_edge_geometry(grid, [circle])
    h = grid.grid_spacing
    checked = 0
    for (dim, side), fraction in geometry.fractions.items():
        cut = np.isfinite(fraction)
        nodes = geometry.nodes[:, cut]
        other = 1 - dim
        half_chord = np.sqrt(radius**2 - (nodes[other] - centre[other]) ** 2)
        crossing = centre[dim] + (half_chord if side == 0 else -half_chord)
        expected = np.abs(crossing - nodes[dim]) / h[dim]
        np.testing.assert_allclose(fraction[cut], expected, rtol=0, atol=1e-13)
        checked += int(cut.sum())
    assert checked > 100


def test_level_set_fractions_agree_with_bisection_with_fewer_evaluations():
    sphere = Sphere((0.1, 0.2, 0.3), 0.8)
    theta = RNG.uniform(0, np.pi, 5000)
    phi = RNG.uniform(0, 2 * np.pi, 5000)
    direction = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    centre = np.asarray(sphere.midpoint)[:, None]
    outside = centre + 0.81 * direction
    inside = centre + 0.79 * direction + RNG.uniform(-0.005, 0.005, direction.shape)
    fallback = MembershipOnly(sphere)
    bisected = intersection_fractions(fallback, outside, inside)
    refined = intersection_fractions(sphere, outside, inside)
    for fractions in (bisected, refined):
        crossing = outside + fractions * (inside - outside)
        np.testing.assert_allclose(sphere.level_set(crossing), 0.0, atol=1e-15)
    np.testing.assert_allclose(refined, bisected, atol=1e-12)
    assert fallback.evaluated == 50 * 5000

    class Counted(MembershipOnly):
        has_level_set = True

        def level_set(self, points):
            self.evaluated += np.shape(points)[1]
            return self.source.level_set(points)

    counted = Counted(sphere)
    intersection_fractions(counted, outside, inside)
    assert counted.evaluated < 10 * 5000


def test_level_set_fractions_fall_back_to_bisection_without_a_sign_change():
    # the half-open rectangle: the node on its upper edge is outside with a zero level
    box = Rectangle((0, 0), (2, 2))
    outside = np.array([[1.0, 0.5], [0.5, 1.5]])
    inside = np.array([[0.5, 0.5], [0.5, 0.5]])
    fractions = intersection_fractions(box, outside, inside)
    np.testing.assert_allclose(fractions, [0.0, 0.5], atol=1e-14)


def test_periodic_wrap_with_a_level_set():
    grid = Grid(
        ndim=2,
        lower_bound=(0, 0),
        upper_bound=(1, 1),
        num_gridpoints=(16, 16),
        periodic=(True, True),
    )
    wall = Rectangle((0.02, 0.5), (0.1, 2.0))
    geometry = bracketed_edge_geometry(grid, [wall])
    fallback = bracketed_edge_geometry(grid, [MembershipOnly(wall)])
    for key, fraction in geometry.fractions.items():
        np.testing.assert_allclose(fraction, fallback.fractions[key], atol=1e-12)


def test_signed_distance_returns_exact_level_sets():
    sphere = Sphere((0, 0, 0), 1)
    points = _points(sphere)
    np.testing.assert_allclose(
        sphere.signed_distance(points), np.linalg.norm(points, axis=0) - 1, atol=1e-15
    )
    cylinder = Cylinder((0, 0, -1), (0, 0, 1), 0.5)
    assert cylinder.signed_distance([0, 0, 2]) == pytest.approx(1.0)
    assert cylinder.signed_distance([1, 0, 0]) == pytest.approx(0.5)
    assert cylinder.signed_distance([0, 0, 0]) == pytest.approx(-0.5)
    # a non-exact level set keeps the sampled-surface path
    shell = Sphere((0, 0, 0), 1) - Sphere((0, 0, 0), 0.4)
    assert shell.signed_distance([0, 0, 0], resolution=48) == pytest.approx(
        0.4, abs=0.05
    )


def test_cut_edges_and_crossings_use_the_level_set():
    tube = Tube((0, 0, -0.5), (0, 0, 0.5), 0.3, 0.6)
    grid = Grid(
        ndim=3,
        lower_bound=(-1, -1, -1),
        upper_bound=(1, 1, 1),
        num_gridpoints=(9, 9, 9),
    )
    hits = tube.cut_edges(grid)
    assert len(hits.fractions) > 50
    np.testing.assert_allclose(tube.level_set(hits.points), 0.0, atol=1e-14)
    start = np.array([[0.0, 0.45], [0.0, 0.0], [0.0, 0.0]])
    end = np.array([[1.0, 0.45], [0.0, 0.0], [0.0, 1.0]])
    result = tube.crossings(start, end)
    np.testing.assert_allclose(result.fraction, [0.3, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.normal[:, 0], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result.normal[:, 1], [0, 0, 1], atol=1e-6)


def test_normals_and_meshes_from_level_sets():
    torus = Torus((0, 0, 0), 1.0, 0.25)
    points = np.array([[1.25, 0.0, 0.75], [0.0, 1.0, 0.0], [0.0, 0.25, 0.0]])
    normals = torus.surface_normal(points)
    np.testing.assert_allclose(normals[:, 0], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(normals[:, 1], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(normals[:, 2], [-1, 0, 0], atol=1e-6)
    pytest.importorskip("skimage")
    mesh = torus.to_mesh(resolution=40)
    assert np.max(np.abs(torus.level_set(mesh.vertices.T))) < 0.01
