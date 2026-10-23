"""The swept polygon absorption: absorb_polygons_2d and the packed polygon objects."""

from __future__ import annotations

import importlib.util

import cunumpy as xp
import numpy as np
import pytest

from simgeometry.kernels.absorb_polygons_2d import kernel as absorb_polygons_2d
from simgeometry.kernels.contains_polygon_2d import kernel as contains_polygon_2d
from simgeometry.objects import Aperture2D, Circle, Polygon
from simgeometry.polygon_objects import object_polygons, polygon_objects

# two plates of an aperture (one object), a triangle, and an inverted square
PLATE_LOW = [(4.0, 0.0), (4.0, 3.0), (5.0, 3.0), (5.0, 0.0)]
PLATE_HIGH = [(4.0, 7.0), (4.0, 10.0), (5.0, 10.0), (5.0, 7.0)]
TRIANGLE = [(7.0, 4.0), (8.0, 6.0), (9.0, 4.0)]
SQUARE = [(1.0, 1.0), (1.0, 9.0), (9.5, 9.0), (9.5, 1.0)]

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


def _objects(polygon_sets, inverted):
    vertices, offsets, first, last = [], [0], [], []
    for polygons in polygon_sets:
        first.append(len(offsets) - 1)
        for polygon in polygons:
            vertices.extend(polygon)
            offsets.append(offsets[-1] + len(polygon))
        last.append(len(offsets) - 1)
    lower = np.array([np.min(np.concatenate(p), axis=0) for p in polygon_sets])
    upper = np.array([np.max(np.concatenate(p), axis=0) for p in polygon_sets])
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(first, dtype=np.int64),
        np.asarray(last, dtype=np.int64),
        np.asarray(inverted, dtype=np.int64),
        lower,
        upper,
        np.asarray([1 - flag for flag in inverted], dtype=np.int64),
    )


OBJECTS = _objects([[PLATE_LOW, PLATE_HIGH], [TRIANGLE], [SQUARE]], [0, 0, 1])


def _absorb(implementation, end, start, rows, use_start, sweep_length, objects, alive):
    alive = alive.copy()
    hit = np.full(rows.shape[0], 99, dtype=np.int64)
    with xp.kernels.use_host_kernel_implementation(implementation):
        absorb_polygons_2d(
            end, start, rows, use_start, sweep_length, *objects, alive, hit
        )
    return alive, hit


def _random_steps(n, seed, step=1.5):
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.0, 10.0, (2, n))
    end = start + rng.normal(scale=step, size=(2, n))
    return start, end


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS[1:])
@pytest.mark.parametrize("use_start", [0, 1])
def test_pyccel_version_matches_numpy(implementation, use_start):
    n = 1500
    start, end = _random_steps(n, seed=1)
    alive = np.random.default_rng(2).random(n) > 0.1
    rows = np.flatnonzero(np.random.default_rng(3).random(n) > 0.3)
    results = [
        _absorb(name, end, start, rows, use_start, 3.0, OBJECTS, alive)
        for name in (implementation, "numpy")
    ]
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    hit = results[0][1]
    assert set(np.unique(hit)) == {-1, 0, 1, 2}
    # dead rows are never hit, and every hit row dies
    np.testing.assert_array_equal(hit[~alive[rows]], -1)
    assert not results[0][0][rows[hit >= 0]].any()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_steps_through_thin_plates_and_corners_are_caught_exactly(implementation):
    objects = _objects([[PLATE_LOW, PLATE_HIGH], [TRIANGLE]], [0, 0])
    steps = {
        # (start, end): the object it hits (-1: none)
        ((3.0, 1.0), (6.0, 1.0)): 0,  # through a plate, ends beyond it
        ((3.9, 3.05), (4.1, 2.95)): 0,  # clips the plate's corner
        ((3.0, 5.0), (6.0, 5.0)): -1,  # through the aperture
        ((3.0, 2.0), (6.5, 5.0)): 0,  # the plate before the triangle
        ((6.5, 5.0), (9.5, 5.0)): 1,  # through the triangle
        ((6.0, 6.5), (10.0, 6.5)): -1,  # above the triangle
        ((4.5, 5.0), (4.5, 1.5)): 0,  # ends inside, from the gap
    }
    start = np.array([s for s, _ in steps]).T.copy()
    end = np.array([e for _, e in steps]).T.copy()
    rows = np.arange(len(steps), dtype=np.int64)
    alive = np.ones(len(steps), dtype=bool)
    absorbed, hit = _absorb(implementation, end, start, rows, 1, 10.0, objects, alive)
    np.testing.assert_array_equal(hit, list(steps.values()))
    np.testing.assert_array_equal(absorbed, hit < 0)
    # without the step only the end counts
    _, hit = _absorb(implementation, end, start, rows, 0, 10.0, objects, alive)
    np.testing.assert_array_equal(hit, [-1, 0, -1, -1, -1, -1, 0])
    # a step longer than the sweep length (wrapped) counts by its end only
    _, hit = _absorb(implementation, end, start, rows, 1, 0.5, objects, alive)
    np.testing.assert_array_equal(hit, [-1, 0, -1, -1, -1, -1, 0])


def test_ends_match_contains_polygon():
    """Without a step the test is the even-odd rule of contains_polygon_2d."""
    rng = np.random.default_rng(4)
    points = rng.uniform(-1.0, 11.0, (2, 3000))
    rows = np.arange(3000, dtype=np.int64)
    alive, hit = _absorb(
        "numpy", points, points, rows, 0, 0.0, OBJECTS, np.ones(3000, dtype=bool)
    )
    inside = np.zeros((3, 3000), dtype=bool)
    for k, (polygons, inverted) in enumerate(
        [([PLATE_LOW, PLATE_HIGH], 0), ([TRIANGLE], 0), ([SQUARE], 1)]
    ):
        for polygon in polygons:
            mask = np.empty(3000, dtype=bool)
            contains_polygon_2d(points, mask, np.asarray(polygon, dtype=float), 0)
            inside[k] |= mask
        if inverted:
            inside[k] = ~inside[k]
    first = np.where(inside.any(axis=0), np.argmax(inside, axis=0), -1)
    np.testing.assert_array_equal(hit, first)
    np.testing.assert_array_equal(alive, first < 0)


@pytest.mark.parametrize("use_start", [0, 1])
def test_cuda_version_matches_numpy_in_emulation(use_start):
    """The CUDA source, compiled as C++ and run serially, against the NumPy version."""
    from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

    if emulation_compiler() is None:
        pytest.skip("no C++ compiler")
    n = 400
    start, end = _random_steps(n, seed=13)
    rows = np.flatnonzero(np.random.default_rng(14).random(n) > 0.2).astype(np.int64)
    alive = np.random.default_rng(15).random(n) > 0.1
    expected_alive, expected_hit = _absorb(
        "numpy", end, start, rows, use_start, 3.0, OBJECTS, alive
    )
    emulated_alive = alive.astype(np.int8)
    hit = np.full(rows.shape[0], 99, dtype=np.int64)
    emulate_cuda_kernel(
        absorb_polygons_2d.cuda_kernel,
        end,
        start,
        rows,
        use_start,
        3.0,
        *OBJECTS,
        emulated_alive,
        hit,
        n_threads=rows.shape[0],
    )
    np.testing.assert_array_equal(emulated_alive.astype(bool), expected_alive)
    np.testing.assert_array_equal(hit, expected_hit)
    assert set(np.unique(expected_hit)) == {-1, 0, 1, 2}


def test_polygon_objects_pack_polygons_and_apertures_only():
    aperture = Aperture2D.slit(4.5, 2.0, 1.0, 3.0, axis=0, position=5.0)
    triangle = Polygon(TRIANGLE)
    circle = Circle(midpoint=(2.0, 2.0), radius=0.5)
    square = Polygon(SQUARE, inverted=True)
    packed = polygon_objects([aperture, circle, triangle, square])
    assert packed.index == (0, 2, 3)
    assert packed.slot(1) is None and packed.slot(2) == 1
    np.testing.assert_array_equal(packed.first_host, [0, 2, 3])
    np.testing.assert_array_equal(packed.last_host, [2, 3, 4])
    np.testing.assert_array_equal(packed.inverted_host, [0, 0, 1])
    np.testing.assert_array_equal(xp.to_numpy(packed.boxed), [1, 1, 0])
    assert len(object_polygons(aperture)) == 2 and object_polygons(circle) is None
    # cached by identity, and None without polygons
    assert polygon_objects([aperture, circle, triangle, square]) is packed
    assert polygon_objects([circle]) is None and polygon_objects([]) is None
    assert polygon_objects([aperture, triangle], skip=frozenset({0})).index == (1,)


def test_absorb_takes_a_step_longer_than_the_plate_thickness():
    """A slit: one marker passes the aperture, one jumps over a plate and is taken."""
    aperture = Aperture2D.slit(4.5, 2.0, 0.1, 5.0, axis=0, position=5.0)
    circle = Circle(midpoint=(8.0, 5.0), radius=0.5)  # not a polygon: ignored
    triangle = Polygon(TRIANGLE)
    packed = polygon_objects([circle, aperture, triangle])
    start = np.array([[3.0, 3.0, 3.0, 6.5], [5.0, 1.0, 9.0, 5.2]])
    end = np.array([[6.0, 6.0, 6.0, 9.5], [5.0, 1.0, 9.0, 5.2]])
    alive = np.ones(4, dtype=bool)
    hit = packed.absorb(end, alive, start=start)
    np.testing.assert_array_equal(hit, [-1, 0, 0, 1])
    np.testing.assert_array_equal(alive, [True, False, False, False])
    assert [packed.index[k] for k in hit[hit >= 0]] == [1, 1, 2]
    # by the end points only, nothing ends in a plate
    alive = np.ones(4, dtype=bool)
    np.testing.assert_array_equal(packed.absorb(end, alive), [-1, -1, -1, -1])
    # a subset of rows, and a sweep length shorter than the steps
    alive = np.ones(4, dtype=bool)
    hit = packed.absorb(end, alive, start=start, rows=np.array([1, 3]))
    np.testing.assert_array_equal(hit, [0, 1])
    alive = np.ones(4, dtype=bool)
    hit = packed.absorb(end, alive, start=start, sweep_length=1.0)
    np.testing.assert_array_equal(hit, [-1, -1, -1, -1])
