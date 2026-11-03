"""PolarRepeat: the copies of a shape about an axis, against their explicit union."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry import Circle, Cuboid, Grid, PolarRepeat, Sphere, Tube

RNG = np.random.default_rng(3)


def _union(shape, count, axis=(0, 0, 1), origin=None):
    union = shape
    for k in range(1, count):
        union = union | shape.rotated(360.0 * k / count, axis=axis, origin=origin)
    return union


@pytest.mark.parametrize(
    ("source", "count", "axis", "origin"),
    [
        (Cuboid((0.12, 0, 0.08), (0.02, 0.002, 0.15)), 24, (0, 0, 1), None),
        (Sphere((0.2, 0.0, 0.1), 0.05), 5, (0, 0, 1), (0.0, 0.0, 0.1)),
        (Sphere((0.0, 0.3, 0.0), 0.1), 3, (1, 0, 0), (0.1, 0.0, 0.0)),
    ],
)
def test_membership_and_level_set_are_those_of_the_union(source, count, axis, origin):
    repeat = PolarRepeat(source, count, axis=axis, origin=origin)
    union = _union(source, count, axis, origin)
    lower = np.asarray(repeat.lower_bound) - 0.02
    upper = np.asarray(repeat.upper_bound) + 0.02
    points = RNG.uniform(lower[:, None], upper[:, None], (3, 50_000))
    np.testing.assert_array_equal(
        np.asarray(repeat.contains_points(points)),
        np.asarray(union.contains_points(points)),
    )
    np.testing.assert_allclose(
        repeat.level_set(points), union.level_set(points), atol=1e-12
    )
    assert repeat.has_level_set and not repeat.level_set_exact
    inside = np.asarray(union.contains_points(points))
    assert np.all((points[:, inside] >= lower[:, None] + 0.02 - 1e-12).all(axis=0))
    assert np.all((points[:, inside] <= upper[:, None] - 0.02 + 1e-12).all(axis=0))
    assert repeat.contains(points[:, np.argmax(inside)])
    assert repeat.volume() == pytest.approx(count * source.volume())


def test_slotted_tube_fractions_match_the_union():
    tube = Tube((0, 0, 0), (0, 0, 0.1), 0.03, 0.036)
    slit = Cuboid((0.033, 0, 0.05), (0.02, 0.003, 0.08))
    grid = Grid(3, (-0.04, -0.04, 0.0), (0.04, 0.04, 0.1), (17, 17, 11))
    fast = (tube - PolarRepeat(slit, 8)).yee_fractions(grid, samples=16)
    slow = (tube - _union(slit, 8)).yee_fractions(grid, samples=16)
    for a in range(3):
        np.testing.assert_allclose(fast.edges[a], slow.edges[a], atol=1e-9)
        np.testing.assert_allclose(fast.faces[a], slow.faces[a], atol=1e-9)
        np.testing.assert_allclose(fast.traces[a], slow.traces[a], atol=1e-9)


def test_invalid_input():
    with pytest.raises(ValueError, match="3D"):
        PolarRepeat(Circle((1, 0), 0.1), 3)
    with pytest.raises(ValueError, match="count"):
        PolarRepeat(Sphere((1, 0, 0), 0.1), 0)
    with pytest.raises(ValueError, match="axis"):
        PolarRepeat(Sphere((1, 0, 0), 0.1), 3, axis=(0, 0, 0))
