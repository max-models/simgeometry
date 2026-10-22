"""Rectangles and cuboids with rounded corners and edges (``fillet``)."""

import cunumpy as xp
import numpy as np
import pytest

from simgeometry import Circle, Cuboid, Rectangle
from simgeometry._level_set import segment_distance
from simgeometry.shapes import _signed_area

RNG = np.random.default_rng(7)
PLATE = Rectangle((0.3, -0.2), (2.0, 1.0), fillet=0.3)
BLOCK = Cuboid((0.1, 0.0, -0.2), (1.2, 0.8, 0.6), fillet=0.2)


def host(array):
    return np.asarray(xp.to_numpy(array))


def _points(shape, n=500, pad=0.4):
    lower = np.asarray(shape.lower_bound, float) - pad
    upper = np.asarray(shape.upper_bound, float) + pad
    return RNG.uniform(lower[:, None], upper[:, None], (shape.ndim, n))


def _capsule_distance(points, a, b, radius):
    """Signed distance to the segment ``a``-``b`` widened by ``radius``."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[:, None]
    t = np.clip(np.sum((points - a) * (b - a), axis=0) / np.sum((b - a) ** 2), 0, 1)
    return np.linalg.norm(points - (a + t * (b - a)), axis=0) - radius


def test_inside_and_outside():
    # the plate spans [-0.7, 1.3] x [-0.7, 0.3]; corners rounded by 0.3
    inside = np.array([[0.3, -0.69, 1.29, 0.3, -0.35], [-0.2, -0.2, -0.2, 0.29, 0.0]])
    outside = np.array([[-0.69, 1.29, 1.29, -0.69, 1.4], [-0.69, -0.69, 0.29, 0.29, 0]])
    assert host(PLATE.contains_points(inside)).all()
    assert not host(PLATE.contains_points(outside)).any()
    assert PLATE.contains((0.3, -0.2)) and not PLATE.contains((-0.69, 0.29))
    # half sides (0.6, 0.4, 0.3): near a corner, near an edge, past an edge
    offsets = np.array([[0.58, 0.5, 0.55], [0.38, 0.3, 0.35], [0.28, 0.0, 0.2]])
    centre = np.asarray(BLOCK.midpoint, float)[:, None]
    np.testing.assert_array_equal(
        host(BLOCK.contains_points(centre + offsets)), [False, True, False]
    )
    assert BLOCK.contains(tuple(centre[:, 0]))


@pytest.mark.parametrize("inverted", [False, True])
def test_inversion_negates_membership_and_level_set(inverted):
    shape = Rectangle((0.3, -0.2), (2.0, 1.0), inverted=inverted, fillet=0.3)
    points = _points(shape)
    values = host(shape.level_set(points))
    np.testing.assert_allclose(values, host(PLATE.level_set(points)) * (-1) ** inverted)
    np.testing.assert_array_equal(host(shape.contains_points(points)), values <= 0)


def test_level_set_is_the_distance_to_a_fine_outline():
    points = _points(PLATE, 400)
    (loop,) = PLATE.outlines(max_spacing=1e-3)
    distance = host(
        segment_distance(
            xp.asarray(points),
            xp.asarray(loop),
            xp.asarray(np.roll(loop, -1, axis=1)),
        )
    )
    inside = host(PLATE.contains_points(points))
    reference = np.where(inside, -distance, distance)
    assert PLATE.has_level_set and PLATE.level_set_exact
    # the chords of the arcs are within r (1 - cos(h / 2r)) of the arc
    np.testing.assert_allclose(host(PLATE.level_set(points)), reference, atol=1e-6)


def test_cuboid_level_set_is_the_distance_to_its_closest_points():
    points = _points(BLOCK)
    closest = BLOCK.closest_point(points)
    assert np.abs(host(BLOCK.level_set(closest))).max() < 1e-12
    distance = np.linalg.norm(points - closest, axis=0)
    np.testing.assert_allclose(np.abs(host(BLOCK.level_set(points))), distance)
    # in the middle plane: the rounded rectangle, or the distance 0.3 to the faces
    rectangle = Rectangle((0.1, 0.0), (1.2, 0.8), fillet=0.2)
    flat = points[:2]
    middle = np.vstack((flat, np.full(flat.shape[1], -0.2)))
    np.testing.assert_allclose(
        host(BLOCK.level_set(middle)),
        np.maximum(host(rectangle.level_set(flat)), -0.3),
    )


@pytest.mark.parametrize("radius", [0.05, 0.3, 0.5])
def test_area_of_the_outline(radius):
    w, h = 2.0, 1.0
    shape = Rectangle((0.0, 0.0), (w, h), fillet=radius)
    (loop,) = shape.outlines(max_spacing=1e-3)
    assert _signed_area(loop) == pytest.approx(w * h - (4 - np.pi) * radius**2, 1e-6)


def test_volume_and_surface_area():
    a, b, c = BLOCK._inner_sides()
    r = BLOCK.fillet
    assert BLOCK.volume() == pytest.approx(
        a * b * c
        + 2 * r * (a * b + b * c + c * a)
        + np.pi * r**2 * (a + b + c)
        + 4 / 3 * np.pi * r**3
    )
    ball = Cuboid((0, 0, 0), (1, 1, 1), fillet=0.5)
    assert ball.volume() == pytest.approx(np.pi / 6)
    assert ball.surface_area() == pytest.approx(np.pi)
    assert Cuboid((0, 0, 0), (1, 2, 3), fillet=0).volume() == pytest.approx(6)
    # the volume by the midpoint rule on 60 x 40 x 30 cells
    lower, upper = np.asarray(BLOCK.lower_bound), np.asarray(BLOCK.upper_bound)
    cells = np.array([60, 40, 30])
    width = (upper - lower) / cells
    axes = [lo + (np.arange(n) + 0.5) * h for lo, n, h in zip(lower, cells, width)]
    centres = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])
    inside = host(BLOCK.contains_points(centres)).sum()
    assert inside * np.prod(width) == pytest.approx(BLOCK.volume(), rel=0.01)


def test_zero_radius_is_the_rectangle():
    points = _points(PLATE)
    sharp = Rectangle((0.3, -0.2), (2.0, 1.0))
    nearly = Rectangle((0.3, -0.2), (2.0, 1.0), fillet=1e-12)
    np.testing.assert_allclose(
        host(nearly.level_set(points)), host(sharp.level_set(points)), atol=1e-11
    )
    np.testing.assert_array_equal(
        host(nearly.contains_points(points)), host(sharp.contains_points(points))
    )
    (loop,) = Rectangle((0.0, 0.0), (2.0, 1.0), fillet=1e-12).outlines()
    assert _signed_area(loop) == pytest.approx(2.0)
    assert sharp.exact_outline and not PLATE.exact_outline
    cube = Cuboid((0, 0, 0), (1.2, 0.8, 0.6))
    rounded = Cuboid((0, 0, 0), (1.2, 0.8, 0.6), fillet=1e-12)
    points = _points(cube)
    np.testing.assert_allclose(
        host(rounded.level_set(points)), host(cube.level_set(points)), atol=1e-11
    )


def test_half_the_shorter_side_is_the_stadium():
    stadium = Rectangle((0.3, -0.2), (2.0, 1.0), fillet=0.5)
    points = _points(stadium)
    np.testing.assert_allclose(
        host(stadium.level_set(points)),
        _capsule_distance(points, (-0.2, -0.2), (0.8, -0.2), 0.5),
        atol=1e-12,
    )
    disc = Rectangle((0.0, 0.0), (1.0, 1.0), fillet=0.5)
    circle = Circle((0.0, 0.0), 0.5)
    np.testing.assert_allclose(
        host(disc.level_set(points)), host(circle.level_set(points)), atol=1e-12
    )
    # no zero-length edges where the flat sides vanish
    for shape in (stadium, disc):
        (loop,) = shape.outlines()
        closed = np.concatenate((loop, loop[:, :1]), axis=1)
        assert np.hypot(*np.diff(closed, axis=1)).min() > 1e-3


@pytest.mark.parametrize("fillet", [0.3, 0.5])
def test_outlines_are_closed_counter_clockwise_and_on_the_boundary(fillet):
    shape = Rectangle((0.3, -0.2), (2.0, 1.0), fillet=fillet)
    for max_spacing in (None, 0.05, lambda p: 0.02 + 0.05 * np.abs(p[0])):
        (loop,) = shape.outlines(max_spacing)
        assert loop.shape[0] == 2 and _signed_area(loop) > 0
        assert not np.allclose(loop[:, 0], loop[:, -1])  # not repeated
        assert np.abs(host(shape.level_set(loop))).max() < 1e-12
        if max_spacing == 0.05:
            closed = np.concatenate((loop, loop[:, :1]), axis=1)
            assert np.hypot(*np.diff(closed, axis=1)).max() <= 0.05 + 1e-12
    # the straight edges stay single segments without max_spacing
    (loop,) = shape.outlines()
    bottom = np.isclose(loop[1], shape.lower_bound[1])
    assert bottom.sum() == 2


def test_crossings_against_brute_force():
    shapes = [PLATE, PLATE.rotated(25).translated((0.1, 0.2)), BLOCK]
    for shape in shapes:
        start, end = _points(shape, 300), _points(shape, 300)
        result = shape.crossings(start, end)
        assert result.exact
        samples = np.linspace(0.0, 1.0, 4001)
        along = start[:, :, None] + (end - start)[:, :, None] * samples
        flat = along.reshape(shape.ndim, -1)
        inside = host(shape.contains_points(flat)).reshape(start.shape[1], -1)
        change = inside[:, 1:] != inside[:, :-1]
        hit = host(result.hit)
        # a segment that only grazes the boundary between samples may differ
        assert np.mean(hit == change.any(axis=1)) > 0.99
        both = hit & change.any(axis=1)
        first = samples[np.argmax(change, axis=1)][both]
        fraction = host(result.fraction)[both]
        assert np.all((fraction >= first - 1e-12) & (fraction <= first + 2.5e-4))
        point, normal = host(result.point)[:, hit], host(result.normal)[:, hit]
        assert np.abs(host(shape.level_set(point))).max() < 1e-12
        # the normals point out: a little along them is outside
        assert not host(shape.contains_points(point + 1e-6 * normal)).any()


def test_segment_queries_and_normals():
    hits = PLATE.intersect_segment((-2.0, -0.2), (2.0, -0.2))
    assert hits.exact
    np.testing.assert_allclose(hits.points[0], [-0.7, 1.3])
    corner = np.array([0.3 + 0.7 + 0.3 * np.cos(0.4), -0.2 + 0.2 + 0.3 * np.sin(0.4)])
    np.testing.assert_allclose(
        PLATE.surface_normal(corner), [np.cos(0.4), np.sin(0.4)], atol=1e-12
    )
    np.testing.assert_allclose(
        PLATE.signed_distance(np.array([1.3 + 0.5, 0.3 + 0.5])),
        np.hypot(0.5 + 0.3, 0.5 + 0.3) - 0.3,
    )


def test_transforms_copies_and_validation():
    moved = PLATE.copy()
    moved.translate((1.0, 0.0))
    moved.mirror(0, 0.0)
    assert moved.fillet == PLATE.fillet
    points = _points(PLATE)
    np.testing.assert_allclose(
        host(moved.level_set(points * [[-1], [1]] - [[1.0], [0.0]])),
        host(PLATE.level_set(points)),
        atol=1e-12,
    )
    block = BLOCK.copy()
    block.translate((0.0, 0.0, 1.0))
    assert block.fillet == BLOCK.fillet
    with pytest.raises(ValueError, match="fillet"):
        Rectangle((0, 0), (2.0, 1.0), fillet=0.6)
    with pytest.raises(ValueError, match="fillet"):
        Cuboid((0, 0, 0), (2.0, 1.0, 0.4), fillet=0.3)
    with pytest.raises(ValueError, match="fillet"):
        Rectangle((0, 0), (2.0, 1.0), fillet=-0.1)


def test_booleans_panels_meshes_and_slices():
    holed = PLATE - Circle((0.3, -0.2), 0.2)
    areas = sorted(_signed_area(loop) for loop in holed.outlines(max_spacing=0.05))
    assert areas[0] == pytest.approx(-np.pi * 0.2**2, rel=1e-2)
    assert areas[1] == pytest.approx(2.0 - (4 - np.pi) * 0.3**2, rel=1e-3)
    panels = PLATE.boundary_panels(max_spacing=0.05)
    perimeter = 2 * (2.0 + 1.0) - (8 - 2 * np.pi) * 0.3
    assert panels.length.sum() == pytest.approx(perimeter, rel=1e-3)
    mesh = BLOCK.to_mesh(resolution=24)
    # linear interpolation between float32 samples 0.03 to 0.06 apart
    assert np.abs(host(BLOCK.level_set(mesh.vertices.T))).max() < 1e-2
    # 0.05 above the flat part of the sides
    vertices, _ = BLOCK.get_vertices(slice_axis=2, slice_position=-0.05)
    radius = np.sqrt(0.2**2 - 0.05**2)
    section = Rectangle((0.1, 0.0), (0.8 + 2 * radius, 0.4 + 2 * radius), fillet=radius)
    assert np.abs(host(section.level_set(host(vertices).T))).max() < 1e-12
