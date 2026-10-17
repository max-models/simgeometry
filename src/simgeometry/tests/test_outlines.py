"""Tests for the boundary loops of 2D shapes."""

import numpy as np
import pytest

from simgeometry import Aperture2D, Circle, Polygon, Rectangle, Triangle
from simgeometry.shapes import _signed_area

SHAPES = {
    "rectangle": Rectangle(midpoint=(0.5, 0.25), side_lengths=(1.0, 0.5)),
    "triangle": Triangle(vertices=((0.0, 0.0), (0.0, 1.0), (1.0, 0.0))),
    "polygon": Polygon(
        vertices=np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0)])
    ),
    "circle": Circle(midpoint=(1.0, -1.0), radius=0.5),
    "aperture2d": Aperture2D(
        axis=0,
        position=0.0,
        vertices=((-0.1, 0.5), (0.1, 0.5), (0.1, 3.0), (-0.1, 3.0)),
    ),
}


@pytest.mark.parametrize("name", SHAPES)
def test_loops_are_counter_clockwise_and_component_major(name):
    for loop in SHAPES[name].outlines():
        assert isinstance(loop, np.ndarray)
        assert loop.shape[0] == 2 and loop.shape[1] >= 3
        assert _signed_area(loop) > 0


@pytest.mark.parametrize("name", SHAPES)
def test_max_spacing_bounds_every_edge_and_keeps_the_area(name):
    shape = SHAPES[name]
    for coarse, fine in zip(shape.outlines(), shape.outlines(max_spacing=0.05)):
        closed = np.concatenate((fine, fine[:, :1]), axis=1)
        assert np.max(np.hypot(*np.diff(closed, axis=1))) <= 0.05 + 1e-12
        if name != "circle":  # refining a polygon only adds collinear vertices
            assert _signed_area(fine) == pytest.approx(_signed_area(coarse))


def test_rectangle_outline_is_its_corners():
    (loop,) = SHAPES["rectangle"].outlines()
    np.testing.assert_allclose(loop, [[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.5, 0.5]])


def test_circle_vertices_lie_on_the_circle_with_the_requested_count():
    circle = SHAPES["circle"]
    n = 37
    (loop,) = circle.outlines(max_spacing=2 * np.pi * circle.radius / n)
    assert loop.shape == (2, n)
    np.testing.assert_allclose(np.hypot(loop[0] - 1.0, loop[1] + 1.0), 0.5)


def test_aperture_has_one_loop_per_plate():
    lower, upper = SHAPES["aperture2d"].outlines()
    assert lower[1].max() == pytest.approx(-0.5)
    assert upper[1].min() == pytest.approx(0.5)


def test_inversion_does_not_change_the_outline():
    plain = Circle(midpoint=(0.0, 0.0), radius=1.0).outlines()
    inverted = Circle(midpoint=(0.0, 0.0), radius=1.0, inverted=True).outlines()
    np.testing.assert_array_equal(plain[0], inverted[0])


def test_transformed_outline_matches_the_transformed_shape():
    rectangle = SHAPES["rectangle"]
    mirrored = rectangle.transformed([[-2.0, 0.0], [0.0, 1.0]], offset=(1.0, 0.0))
    (loop,) = mirrored.outlines(max_spacing=0.1)
    assert _signed_area(loop) > 0
    assert np.max(np.hypot(*np.diff(loop, axis=1))) <= 0.1 + 1e-12
    np.testing.assert_allclose(loop.min(axis=1), [-1.0, 0.0])
    np.testing.assert_allclose(loop.max(axis=1), [1.0, 0.5])


def test_max_spacing_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        SHAPES["circle"].outlines(max_spacing=0.0)
    with pytest.raises(ValueError, match="positive"):
        SHAPES["circle"].outlines(max_spacing=lambda p: np.zeros(p.shape[1]))


@pytest.mark.parametrize("name", SHAPES)
def test_constant_callable_spacing_matches_the_constant(name):
    shape = SHAPES[name]
    for plain, graded in zip(
        shape.outlines(max_spacing=0.05),
        shape.outlines(max_spacing=lambda p: np.full(p.shape[1], 0.05)),
    ):
        assert plain.shape == graded.shape
        np.testing.assert_allclose(plain, graded, atol=1e-12)


def test_graded_spacing_follows_the_function():
    """Edges of a plate get dense near its tip and sparse far from it."""
    (_, upper) = SHAPES["aperture2d"].outlines(
        max_spacing=lambda p: 0.02 + 0.2 * (np.abs(p[1]) - 0.5)
    )
    closed = np.concatenate((upper, upper[:, :1]), axis=1)
    edges = np.hypot(*np.diff(closed, axis=1))
    mid_y = np.abs(0.5 * (closed[1, 1:] + closed[1, :-1]))
    target = 0.02 + 0.2 * (mid_y - 0.5)
    sides = mid_y < 2.9  # the 0.2-wide end cap is shorter than its target spacing
    np.testing.assert_allclose(edges[sides], target[sides], rtol=0.1, atol=2e-3)
    assert edges[mid_y < 0.6].max() < 0.05 < edges[mid_y > 2.5].min()


def test_graded_spacing_on_a_circle_and_a_transformed_shape():
    def spacing(p):
        return np.where(p[0] > 1.0, 0.01, 0.1)

    (loop,) = SHAPES["circle"].outlines(max_spacing=spacing)
    right = loop[0] > 1.05
    left = loop[0] < 0.95
    assert np.sum(right) > 5 * np.sum(left)
    np.testing.assert_allclose(np.hypot(loop[0] - 1.0, loop[1] + 1.0), 0.5)

    moved = SHAPES["rectangle"].translated((0.5, 0.0))
    (loop,) = moved.outlines(max_spacing=spacing)
    assert np.sum(loop[0] > 1.0) > 5 * np.sum(loop[0] < 1.0)
