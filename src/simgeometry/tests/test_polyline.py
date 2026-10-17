"""Tests for open polylines."""

import numpy as np
import pytest

from simgeometry import Circle, Polygon, Polyline, Rectangle


def test_polyline_is_an_open_curve_without_area():
    wire = Polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)])
    assert wire.closed is False and wire.exact_outline is True
    assert wire.length == pytest.approx(3.0)
    assert wire.lower_bound == (0.0, 0.0) and wire.upper_bound == (1.0, 2.0)
    assert not wire.contains((0.5, 0.0))
    mask = np.asarray(wire.contains_points(np.array([[0.5, 1.0], [0.0, 1.0]])))
    assert mask.dtype == bool and not mask.any()


def test_a_horizontal_polyline_has_a_flat_bounding_box():
    plate = Polyline([(0.0, 1.0), (2.0, 1.0)])
    assert plate.lower_bound == (0.0, 1.0) and plate.upper_bound == (2.0, 1.0)


def test_outline_is_open_and_keeps_both_ends():
    wire = Polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)])
    (line,) = wire.outlines()
    np.testing.assert_array_equal(line, [[0.0, 1.0, 1.0], [0.0, 0.0, 2.0]])
    (fine,) = wire.outlines(max_spacing=0.25)
    np.testing.assert_array_equal(fine[:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(fine[:, -1], [1.0, 2.0])
    assert fine.shape[1] == 4 + 8 + 1
    assert np.hypot(*np.diff(fine, axis=1)).max() <= 0.25 + 1e-12


def test_outline_keeps_the_given_direction_even_when_clockwise():
    hook = Polyline([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    (line,) = hook.outlines()
    np.testing.assert_array_equal(line[:, 0], [0.0, 1.0])


def test_transformed_polyline_stays_open_and_ordered_under_a_reflection():
    wire = Polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)])
    mirrored = wire.transformed([[-1.0, 0.0], [0.0, 1.0]])
    assert mirrored.closed is False and mirrored.exact_outline is True
    (line,) = mirrored.outlines()
    np.testing.assert_array_equal(line, [[0.0, -1.0, -1.0], [0.0, 0.0, 2.0]])


def test_invalid_polylines_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        Polyline([(0.0, 0.0)])
    with pytest.raises(ValueError, match="differ"):
        Polyline([(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)])


def test_crossings_find_a_thin_plate():
    plate = Polyline([(0.0, -1.0), (0.0, 1.0)])
    start = np.array([[-1.0, -1.0, 1.0], [0.0, 2.0, 0.5]])
    end = np.array([[1.0, 1.0, -1.0], [0.0, 2.0, 0.5]])
    hits = plate.crossings(start, end)
    np.testing.assert_array_equal(np.asarray(hits.hit), [True, False, True])
    np.testing.assert_allclose(np.asarray(hits.fraction)[[0, 2]], 0.5)


def test_open_curves_cannot_be_combined_in_boolean_shapes():
    pytest.importorskip("shapely")
    with pytest.raises(NotImplementedError, match="open curve"):
        (Rectangle((0, 0), (1, 1)) - Polyline([(0, 0), (1, 1)])).outlines()


def test_polyline_plots_as_a_line():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    ax = Polyline([(0.0, 0.0), (1.0, 1.0)]).plot()
    assert len(ax.lines) == 1 and not ax.patches
    assert Circle((0, 0), 1).plot().patches


def test_polygon_contains_no_points():
    """An empty batch must not reach the compiled kernel, which rejects size 0."""
    triangle = Polygon(vertices=np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
    assert np.asarray(triangle.contains_points(np.zeros((2, 0)))).shape == (0,)
