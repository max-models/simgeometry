"""Tests for boundary panels."""

import numpy as np
import pytest

from simgeometry import Aperture2D, Circle, Polyline, Rectangle
from simgeometry.panels import Panels


def test_rectangle_panels_have_outward_normals_and_the_perimeter():
    rectangle = Rectangle(midpoint=(0.0, 0.0), side_lengths=(2.0, 1.0))
    panels = rectangle.boundary_panels(max_spacing=0.25)
    assert isinstance(panels, Panels) and panels.closed
    assert len(panels) == 8 + 4 + 8 + 4
    assert panels.length.sum() == pytest.approx(6.0)
    assert panels.length.max() <= 0.25 + 1e-12
    # outward: a small step along the normal leaves the rectangle
    outside = panels.midpoint + 1e-6 * panels.normal
    inside = panels.midpoint - 1e-6 * panels.normal
    assert not np.asarray(rectangle.contains_points(outside)).any()
    assert np.asarray(rectangle.contains_points(inside)).all()


def test_panels_close_each_loop_and_record_their_piece():
    slit = Aperture2D.slit(0.0, half_width=0.5, thickness=0.2, length=1.0)
    panels = slit.boundary_panels(max_spacing=0.1)
    assert set(panels.piece.tolist()) == {0, 1}
    for piece in (0, 1):
        own = panels.piece == piece
        np.testing.assert_allclose(
            panels.end[:, own][:, -1], panels.start[:, own][:, 0]
        )
        assert panels.length[own].sum() == pytest.approx(2 * (0.2 + 1.0))


def test_circle_panel_normals_point_radially_outward():
    panels = Circle((1.0, 2.0), 0.5).boundary_panels(max_spacing=0.05)
    radial = panels.midpoint - np.array([[1.0], [2.0]])
    radial /= np.hypot(*radial)
    np.testing.assert_allclose(np.sum(radial * panels.normal, axis=0), 1.0, atol=1e-12)


def test_hole_normals_point_into_the_hole():
    pytest.importorskip("shapely")
    plate = Rectangle((0.0, 0.0), (4.0, 4.0)) - Circle((0.0, 0.0), 1.0)
    panels = plate.boundary_panels(max_spacing=0.1)
    hole = np.hypot(*panels.midpoint) < 1.5
    toward_centre = -panels.midpoint[:, hole] / np.hypot(*panels.midpoint[:, hole])
    assert np.all(np.sum(toward_centre * panels.normal[:, hole], axis=0) > 0.99)


def test_open_curve_has_one_panel_fewer_than_vertices_and_right_hand_normals():
    wire = Polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    panels = wire.boundary_panels()
    assert not panels.closed and len(panels) == 2
    np.testing.assert_allclose(panels.normal, [[0.0, 1.0], [-1.0, 0.0]], atol=1e-12)


def test_transformed_shape_panels():
    rotated = Rectangle((0.0, 0.0), (2.0, 1.0)).rotated(90)
    panels = rotated.boundary_panels(max_spacing=0.5)
    assert panels.length.sum() == pytest.approx(6.0)
    outside = panels.midpoint + 1e-6 * panels.normal
    assert not np.asarray(rotated.contains_points(outside)).any()
