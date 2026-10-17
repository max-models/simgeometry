"""Tests for the batched first-crossing query :meth:`Shape.crossings`."""

import cunumpy as xp
import numpy as np
import pytest

from simgeometry import (
    Aperture2D,
    Circle,
    Cuboid,
    Polygon,
    Rectangle,
    Sphere,
    Triangle,
)
from simgeometry.crossings import START_TOLERANCE, Crossings
from simgeometry.objects import Line


def host(array):
    return np.asarray(xp.to_numpy(array))


def random_segments(ndim, n, seed=0, low=-2.0, high=2.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, (ndim, n)), rng.uniform(low, high, (ndim, n))


def first_reference(shape, start, end):
    """First crossing from the scalar segment query's analytic candidates.

    Uses ``queries._candidates`` and ``surface_normal`` (what ``intersect_segment``
    builds on) rather than ``intersect_segment`` itself, which calls
    ``contains_points`` with no points when a segment misses, and the compiled
    polygon kernel rejects an empty array. Returns (fraction, normal) or None.
    """
    from simgeometry.queries import _candidates, surface_normal

    candidates, exact = _candidates(shape, start, end, 256, 1e-10)
    assert exact
    ts = sorted(t for t in candidates if START_TOLERANCE < t <= 1 + 1e-10)
    if not ts:
        return None
    t = min(ts[0], 1.0)
    return t, surface_normal(shape, start + t * (end - start))


ANALYTIC = {
    "circle": Circle(midpoint=(0.2, -0.1), radius=0.9),
    "sphere": Sphere(midpoint=(0.1, 0.0, -0.2), radius=0.8),
    "rectangle": Rectangle(midpoint=(0.1, 0.2), side_lengths=(1.4, 0.8)),
    "cuboid": Cuboid(midpoint=(0.0, 0.1, 0.0), side_lengths=(1.2, 0.6, 1.0)),
    "triangle": Triangle(vertices=((-1.0, -1.0), (1.2, -0.6), (0.1, 1.3))),
    "polygon": Polygon(
        vertices=np.array([(-1, -1), (1, -1), (1, 0), (0, 0), (0, 1), (-1, 1)], float)
    ),
    "line": Line(midpoint=(0.1,), side_lengths=(1.0,)),
}


@pytest.mark.parametrize("name", ANALYTIC)
def test_first_crossing_matches_the_scalar_query(name):
    shape = ANALYTIC[name]
    start, end = random_segments(shape.ndim, 200, seed=list(ANALYTIC).index(name))
    result = shape.crossings(start, end)
    assert isinstance(result, Crossings) and result.exact and len(result) == 200
    hit, fraction = host(result.hit), host(result.fraction)
    point, normal = host(result.point), host(result.normal)
    for i in range(200):
        reference = first_reference(shape, start[:, i], end[:, i])
        if reference is None:
            assert not hit[i]
            assert np.isnan(fraction[i]) and np.all(np.isnan(point[:, i]))
            continue
        assert hit[i]
        assert fraction[i] == pytest.approx(reference[0], abs=1e-9)
        np.testing.assert_allclose(
            point[:, i], start[:, i] + fraction[i] * (end[:, i] - start[:, i])
        )
        np.testing.assert_allclose(np.linalg.norm(normal[:, i]), 1.0)
        if not np.any(np.isnan(reference[1])):
            np.testing.assert_allclose(normal[:, i], reference[1], atol=1e-9)
    assert hit.any() and not hit.all()


def test_a_step_through_a_thin_plate_is_caught_at_its_first_face():
    plate = Rectangle(midpoint=(0.0, 0.0), side_lengths=(0.01, 2.0))
    start = np.array([[-1.0, 1.0], [0.0, 0.5]])
    end = np.array([[1.0, -1.0], [0.0, 0.5]])
    # both end points are outside the plate: an end-point test misses it
    assert not host(plate.contains_points(end)).any()
    result = plate.crossings(start, end)
    np.testing.assert_array_equal(host(result.hit), [True, True])
    np.testing.assert_allclose(host(result.fraction), [0.4975, 0.4975])
    np.testing.assert_allclose(host(result.normal), [[-1.0, 1.0], [0.0, 0.0]])


def test_aperture_plates_are_both_tested():
    slit = Aperture2D(
        axis=0,
        position=0.0,
        vertices=((-0.05, 0.3), (0.05, 0.3), (0.05, 2.0), (-0.05, 2.0)),
    )
    start = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, 0.0]])
    end = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
    result = slit.crossings(start, end)
    np.testing.assert_array_equal(host(result.hit), [True, True, False])
    np.testing.assert_allclose(host(result.fraction)[:2], [0.475, 0.475])


def test_inverted_shapes_flip_the_normal_but_not_the_crossing():
    start, end = np.array([[0.0], [0.0]]), np.array([[2.0], [0.0]])
    plain = Circle(midpoint=(0.0, 0.0), radius=1.0).crossings(start, end)
    inverted = Circle(midpoint=(0.0, 0.0), radius=1.0, inverted=True).crossings(
        start, end
    )
    assert host(plain.fraction)[0] == host(inverted.fraction)[0] == 0.5
    np.testing.assert_allclose(host(plain.normal)[:, 0], [1.0, 0.0])
    np.testing.assert_allclose(host(inverted.normal)[:, 0], [-1.0, 0.0])


def test_transformed_shapes_map_points_and_normals():
    bar = Rectangle(midpoint=(0.0, 0.0), side_lengths=(2.0, 0.4))
    tilted = bar.rotated(30).scaled((1.0, 2.0)).translated((0.5, -0.2))
    start, end = random_segments(2, 200, seed=3)
    result = tilted.crossings(start, end)
    assert result.exact
    hit = host(result.hit)
    for i in np.flatnonzero(hit):
        reference = first_reference(tilted, start[:, i], end[:, i])
        assert reference is not None
        assert host(result.fraction)[i] == pytest.approx(reference[0], abs=1e-9)
        np.testing.assert_allclose(host(result.normal)[:, i], reference[1], atol=1e-6)
    assert hit.any()
    # the normals are outward: a little along the normal is outside
    point, normal = host(result.point)[:, hit], host(result.normal)[:, hit]
    assert not host(tilted.contains_points(point + 1e-6 * normal)).any()


def test_boolean_shapes_are_sampled_and_agree_with_the_analytic_parts():
    plate = Rectangle(midpoint=(0.0, 0.0), side_lengths=(2.0, 2.0))
    hole = Circle(midpoint=(0.0, 0.0), radius=0.5)
    holed = plate - hole
    start = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    end = np.array([[2.0, 2.0, 0.0], [0.0, 0.0, 0.3]])
    result = holed.crossings(start, end)
    assert not result.exact
    np.testing.assert_array_equal(host(result.hit), [True, True, False])
    # from outside: the plate's left face; from the hole's centre: the hole's rim
    np.testing.assert_allclose(host(result.fraction)[:2], [0.25, 0.25], atol=1e-9)
    np.testing.assert_allclose(
        host(result.normal)[:, :2], [[-1.0, -1.0], [0.0, 0.0]], atol=0.1
    )


def test_a_crossing_at_the_start_is_not_reported_but_one_at_the_end_is():
    circle = Circle(midpoint=(0.0, 0.0), radius=1.0)
    # leaving from exactly on the boundary; arriving exactly on it
    start = np.array([[1.0, 0.0], [0.0, 0.0]])
    end = np.array([[2.0, 1.0], [0.0, 0.0]])
    result = circle.crossings(start, end)
    np.testing.assert_array_equal(host(result.hit), [False, True])
    assert host(result.fraction)[1] == 1.0


def test_grazing_and_degenerate_segments_do_not_cross():
    circle = Circle(midpoint=(0.0, 0.0), radius=1.0)
    start = np.array([[-1.0, 0.3], [1.0, 0.3]])
    end = np.array([[1.0, 0.3], [1.0, 0.3]])  # tangent; zero length
    assert not host(circle.crossings(start, end).hit).any()


def test_no_segments():
    result = Circle(midpoint=(0.0, 0.0), radius=1.0).crossings(
        np.zeros((2, 0)), np.zeros((2, 0))
    )
    assert len(result) == 0 and host(result.point).shape == (2, 0)


def test_large_batches_are_chunked(monkeypatch):
    import simgeometry.crossings as module

    monkeypatch.setattr(module, "MAX_PAIRS", 50)
    polygon = ANALYTIC["polygon"]
    start, end = random_segments(2, 300, seed=5)
    chunked = polygon.crossings(start, end)
    monkeypatch.setattr(module, "MAX_PAIRS", 4_000_000)
    whole = polygon.crossings(start, end)
    np.testing.assert_array_equal(host(chunked.fraction), host(whole.fraction))


def test_mismatched_inputs_are_rejected():
    circle = Circle(midpoint=(0.0, 0.0), radius=1.0)
    with pytest.raises(ValueError, match="same shape"):
        circle.crossings(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        circle.crossings(np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="samples"):
        (circle - Circle(midpoint=(0.0, 0.0), radius=0.5)).crossings(
            np.zeros((2, 1)), np.ones((2, 1)), samples=1
        )
