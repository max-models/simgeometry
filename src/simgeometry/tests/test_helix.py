"""Helix: the level set against dense sampling, volume, bounds, frames, crossings, Yee fractions."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry import Grid, Helix
from simgeometry.crossings import crossings

RNG = np.random.default_rng(5)

SHAPES = {
    "coil": Helix((0, 0, 0.02), 0.14, 0.02, 6, 0.003),
    "oblique left": Helix(
        (0.1, -0.2, 0.3), 0.05, 0.03, 2.5, 0.008, axis=(1, 1, 0.5), left_handed=True
    ),
    "steep": Helix((0, 0, 0), 0.02, 0.2, 1.3, 0.005, start_direction=(0, 1, 0)),
}


def _box_points(shape, n, pad=0.0):
    lower = np.asarray(shape.lower_bound) - pad
    upper = np.asarray(shape.upper_bound) + pad
    return RNG.uniform(lower[:, None], upper[:, None], (3, n)), np.prod(upper - lower)


def _sampled_distance(shape, points, samples=100_000):
    t = np.linspace(0, 2 * np.pi * shape.turns, samples)
    curve = np.asarray(shape.centreline(t))
    return np.array(
        [np.min(np.linalg.norm(curve - p[:, None], axis=0)) for p in points.T]
    )


@pytest.mark.parametrize("name", SHAPES)
def test_level_set_is_the_distance_to_the_centreline(name):
    shape = SHAPES[name]
    box, _ = _box_points(shape, 300, pad=0.03)
    along = np.asarray(shape.centreline(RNG.uniform(0, 2 * np.pi * shape.turns, 300)))
    points = np.concatenate(
        [box, along + RNG.normal(0, shape.wire_radius, along.shape)], axis=1
    )
    expected = _sampled_distance(shape, points) - shape.wire_radius
    values = np.asarray(shape.level_set(points))
    # exact within a wire radius of the wire (the dense sampling's own error is
    # below 1e-6 here), a lower bound farther away
    near = expected < shape.wire_radius
    assert near.sum() > 100
    np.testing.assert_allclose(values[near], expected[near], atol=2e-6)
    assert np.all(values <= expected + 2e-6)
    inside = np.asarray(shape.contains_points(points))
    np.testing.assert_array_equal(inside, values <= 0)
    assert shape.has_level_set and not shape.level_set_exact
    for point in points[:, :5].T:
        assert shape.contains(point) == bool(inside[np.argmax(points[0] == point[0])])


@pytest.mark.parametrize("name", SHAPES)
def test_volume_and_bounds(name):
    shape = SHAPES[name]
    points, box = _box_points(shape, 300_000)
    inside = np.asarray(shape.contains_points(points))
    assert np.mean(inside) * box == pytest.approx(shape.volume(), rel=0.03)
    outside, _ = _box_points(shape, 20_000, pad=0.02)
    beyond = np.any(
        (outside < np.asarray(shape.lower_bound)[:, None])
        | (outside > np.asarray(shape.upper_bound)[:, None]),
        axis=0,
    )
    assert not np.asarray(shape.contains_points(outside[:, beyond])).any()


def test_frame_handedness_and_ends():
    coil = SHAPES["coil"]
    first, last = np.asarray(coil.centreline([0.0, 2 * np.pi * coil.turns])).T
    np.testing.assert_allclose(first, (0.14, 0, 0.02), atol=1e-15)
    np.testing.assert_allclose(last, (0.14, 0, 0.14), atol=1e-12)
    # right-handed: a quarter turn goes from +x to +y
    np.testing.assert_allclose(
        np.asarray(coil.centreline([np.pi / 2]))[:, 0], (0, 0.14, 0.025), atol=1e-12
    )
    assert not coil.left_handed and SHAPES["oblique left"].left_handed
    assert coil.length == pytest.approx(0.12)
    assert coil.wire_length == pytest.approx(6 * np.hypot(2 * np.pi * 0.14, 0.02))
    assert coil.surface_area() == pytest.approx(
        2 * np.pi * 0.003 * coil.wire_length + 4 * np.pi * 0.003**2
    )
    # hemispherical caps: just beyond the start, along the wire's tangent
    assert coil.contains((0.14, -0.0029, 0.02))
    assert not coil.contains((0.14, -0.0031, 0.02))


def test_copy_translate_and_mirror():
    shape = SHAPES["oblique left"]
    points, _ = _box_points(shape, 2000, pad=0.02)
    copied = shape.copy()
    np.testing.assert_allclose(copied.level_set(points), shape.level_set(points))
    assert copied.left_handed and copied.start_direction == shape.start_direction
    moved = shape.copy()
    moved.translate((0.1, 0.0, -0.2))
    np.testing.assert_allclose(
        moved.level_set(points + np.array([[0.1], [0.0], [-0.2]])),
        shape.level_set(points),
        atol=1e-12,
    )
    mirrored = shape.copy()
    mirrored.mirror(1, 0.05)
    image = points.copy()
    image[1] = 0.1 - image[1]
    np.testing.assert_allclose(
        mirrored.level_set(image), shape.level_set(points), atol=1e-12
    )
    assert not mirrored.left_handed
    with pytest.raises(ValueError, match="Axis"):
        mirrored.mirror(3, 0.0)


def test_crossings_of_the_wire():
    coil = SHAPES["coil"]
    # radial segments through the first turn's wire at z = 0.02 + 0.02 * angle / 2 pi
    angles = np.linspace(0.1, 2 * np.pi - 0.1, 50)
    z = 0.02 + 0.02 * angles / (2 * np.pi)
    start = np.stack([0.1 * np.cos(angles), 0.1 * np.sin(angles), z])
    end = np.stack([0.2 * np.cos(angles), 0.2 * np.sin(angles), z])
    result = crossings(coil, start, end)
    assert np.all(np.asarray(result.hit))
    radius = 0.1 + 0.1 * np.asarray(result.fraction)
    # the wire's inner side, where the plane z = const cuts it (to 1e-3 of the pitch)
    np.testing.assert_allclose(radius, 0.14 - 0.003, atol=2e-5)


def test_yee_fractions_of_the_wire():
    coil = Helix((0, 0, -0.01), 0.03, 0.02, 1, 0.005)
    grid = Grid(3, (-0.04, -0.04, -0.02), (0.04, 0.04, 0.02), (33, 33, 17))
    fractions = coil.yee_fractions(grid, samples=16)
    h = 0.08 / 32
    # each family of edges, its inside lengths times h^2, is the wire's volume
    for family in fractions.edges:
        assert np.sum(family) * h**3 == pytest.approx(coil.volume(), rel=0.05)
    # the projected traces times h over all three families: twice the surface's area
    area = sum(np.sum(t) * h for t in fractions.projected_traces)
    assert area == pytest.approx(2 * coil.surface_area(), rel=0.05)


def test_invalid_input():
    with pytest.raises(ValueError, match="wire_radius"):
        Helix((0, 0, 0), 0.1, 0.02, 3, 0.2)
    with pytest.raises(ValueError, match="pitch"):
        Helix((0, 0, 0), 0.1, 0.0, 3, 0.01)
    with pytest.raises(ValueError, match="nonzero"):
        Helix((0, 0, 0), 0.1, 0.02, 3, 0.01, axis=(0, 0, 0))
    with pytest.raises(ValueError, match="start_direction"):
        Helix((0, 0, 0), 0.1, 0.02, 3, 0.01, start_direction=(0, 0, 2))
