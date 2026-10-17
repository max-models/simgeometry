"""Tests for the boundary loops of 2D Boolean shapes."""

import numpy as np
import pytest

from simgeometry import Circle, Rectangle
from simgeometry.shapes import _signed_area

pytest.importorskip("shapely")

PLATE = Rectangle(midpoint=(0.0, 0.0), side_lengths=(4.0, 2.0))
HOLE = Circle(midpoint=(0.5, 0.0), radius=0.5)


def _edges(loop):
    closed = np.concatenate((loop, loop[:, :1]), axis=1)
    return np.hypot(*np.diff(closed, axis=1))


def _assert_on_boundary(shape, loops, step=1e-6):
    """Vertices nudged to the left of the loop land inside, to the right outside.

    Vertices, not edge midpoints: a curved operand enters as its inscribed polygon,
    whose chords cut inside the curve, but its vertices lie on it.
    """
    for loop in loops:
        incoming = loop - np.roll(loop, 1, axis=1)
        outgoing = np.roll(loop, -1, axis=1) - loop
        normals = sum(
            np.stack((-d[1], d[0])) / np.hypot(*d) for d in (incoming, outgoing)
        )
        normals /= np.hypot(*normals)
        inside = np.asarray(shape.contains_points(loop + step * normals), dtype=bool)
        outside = np.asarray(shape.contains_points(loop - step * normals), dtype=bool)
        assert inside.all() and not outside.any()


def test_plate_minus_disc_has_a_counter_clockwise_outline_and_a_clockwise_hole():
    shape = PLATE - HOLE
    outer, hole = shape.outlines()
    assert _signed_area(outer) == pytest.approx(8.0)
    assert _signed_area(hole) < 0
    (disc,) = HOLE.outlines()
    assert _signed_area(outer) + _signed_area(hole) == pytest.approx(
        8.0 - _signed_area(disc)
    )
    assert 8.0 - np.pi * 0.25 < _signed_area(outer) + _signed_area(hole) < 8.0
    _assert_on_boundary(shape, shape.outlines())


def test_union_of_overlapping_rectangles_is_one_exact_loop():
    shape = Rectangle((0.0, 0.0), (2.0, 1.0)) | Rectangle((1.0, 0.5), (2.0, 1.0))
    (loop,) = shape.outlines()
    assert _signed_area(loop) == pytest.approx(2.0 + 2.0 - 0.5)
    assert loop.shape[1] == 8
    _assert_on_boundary(shape, [loop])


def test_intersection_and_nested_difference():
    a = Rectangle((0.0, 0.0), (2.0, 1.0))
    b = Rectangle((1.0, 0.5), (2.0, 1.0))
    (loop,) = (a & b).outlines()
    assert _signed_area(loop) == pytest.approx(0.5)

    nested = (a | b) - Rectangle((1.0, 0.5), (0.4, 0.4))
    loops = nested.outlines()
    assert [_signed_area(loop) > 0 for loop in loops] == [True, False]
    assert sum(_signed_area(loop) for loop in loops) == pytest.approx(3.5 - 0.16)
    _assert_on_boundary(nested, loops)


def test_disjoint_union_gives_two_counter_clockwise_loops_in_order():
    shape = Rectangle((3.0, 0.0), (1.0, 1.0)) | Circle((0.0, 0.0), 0.5)
    loops = shape.outlines()
    assert len(loops) == 2
    assert all(_signed_area(loop) > 0 for loop in loops)
    assert loops[0][0].max() < loops[1][0].min()  # sorted left to right


def test_max_spacing_bounds_every_edge():
    shape = PLATE - HOLE
    loops = shape.outlines(max_spacing=0.05)
    for loop in loops:
        assert _edges(loop).max() <= 0.05 + 1e-12
    _assert_on_boundary(shape, loops)
    (hole,) = [loop for loop in loops if _signed_area(loop) < 0]
    np.testing.assert_allclose(np.hypot(hole[0] - 0.5, hole[1]), 0.5)


def test_callable_spacing_grades_the_loops():
    shape = PLATE - HOLE

    def spacing(points):
        return np.where(points[0] > 1.5, 0.02, 0.2)

    outer = shape.outlines(max_spacing=spacing)[0]
    right = outer[0] > 1.6
    left = outer[0] < -1.6
    assert np.sum(right) > 5 * np.sum(left)


def test_transformed_operand():
    shape = PLATE - Rectangle((0.0, 0.0), (1.0, 1.0)).rotated(45)
    outer, hole = shape.outlines()
    assert _signed_area(hole) == pytest.approx(-1.0)
    _assert_on_boundary(shape, [outer, hole])


def test_inverted_and_unsupported_operands_raise():
    inverted = Circle((0.0, 0.0), 0.5, inverted=True)
    with pytest.raises(NotImplementedError, match="inverted"):
        (PLATE - inverted).outlines()
    from simgeometry import FunctionObject2D

    blob = FunctionObject2D(lambda p: p[0] ** 2 + p[1] ** 2 < 1, (-1, -1), (1, 1))
    with pytest.raises(NotImplementedError, match="outlines"):
        (PLATE | blob).outlines()
