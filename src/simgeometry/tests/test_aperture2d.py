import numpy as np

"""Tests for two-dimensional aperture geometry."""

import cunumpy as xp
import pytest

from simgeometry.objects.aperture2d import Aperture2D


class TestAperture2D:
    """Group tests for two-dimensional aperture geometry."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        axis = 0
        position = 10.0
        vertices = ((0, 0), (3, 5), (5, 0))
        aperture = Aperture2D(axis=axis, position=position, vertices=vertices)

        assert aperture.axis == axis
        assert aperture.position == position
        assert aperture.vertices == vertices
        # Check that polygons are created
        assert aperture._lower_polygon is not None
        assert aperture._upper_polygon is not None

    def test_init_invalid_axis(self) -> None:
        """Test initialization with invalid axis."""
        with pytest.raises(AssertionError, match="Axis must be one of"):
            Aperture2D(axis=2, position=10.0, vertices=((0, 0), (3, 5), (5, 0)))

    def test_axis_setter_invalid(self) -> None:
        """Test axis setter with invalid value."""
        aperture = Aperture2D(axis=0, position=10.0, vertices=((0, 0), (3, 5), (5, 0)))
        with pytest.raises(AssertionError, match="Axis must be one of"):
            aperture.axis = 2

    def test_vertices_setter_invalid_length(self) -> None:
        """Test vertices setter with invalid vertex length."""
        aperture = Aperture2D(axis=0, position=10.0, vertices=((0, 0), (3, 5), (5, 0)))
        with pytest.raises(
            AssertionError,
            match="vertices must be a tuple of length 2",
        ):
            aperture.vertices = ((0,), (3, 5), (5, 0))

    def test_rvec_property(self) -> None:
        """Test rvec property extraction."""
        vertices = ((0, 1), (3, 2), (5, 3))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)
        assert aperture.rvec == [1, 2, 3]

    def test_zvec_property(self) -> None:
        """Test zvec property extraction."""
        vertices = ((0, 1), (3, 2), (5, 3))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)
        assert aperture.zvec == [0, 3, 5]

    def test_outlines_mirror_the_profile_about_the_centre_line(self) -> None:
        """The lower plate is at ``position - radial``, the upper at ``position + radial``."""
        vertices = ((0, 0), (3, 5), (5, 0))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)
        lower, upper = aperture.outlines()
        assert sorted(map(tuple, lower.T.tolist())) == [(0, 10), (3, 5), (5, 10)]
        assert sorted(map(tuple, upper.T.tolist())) == [(0, 10), (3, 15), (5, 10)]

    def test_axis_one_runs_the_beam_along_y(self) -> None:
        """With ``axis=1`` the profile's axial coordinate is y and the plates sit at x."""
        vertices = ((0, 1), (2, 1), (2, 3), (0, 3))
        along_x = Aperture2D(axis=0, position=0.5, vertices=vertices)
        along_y = Aperture2D(axis=1, position=0.5, vertices=vertices)
        for loop_x, loop_y in zip(along_x.outlines(), along_y.outlines()):
            assert sorted(map(tuple, loop_y.T.tolist())) == sorted(
                map(tuple, loop_x[::-1].T.tolist())
            )
        assert along_y.contains((2.5, 1.0)) and not along_y.contains((1.0, 2.5))
        assert along_x.contains((1.0, 2.5)) and not along_x.contains((2.5, 1.0))
        assert along_y.lower_bound == (-2.5, 0.0)
        assert along_y.upper_bound == (3.5, 2.0)

    def test_slit_builds_two_rectangular_plates(self) -> None:
        """``slit`` leaves a gap of twice the half-width between plates of the given size."""
        slit = Aperture2D.slit(
            4.0, half_width=0.5, thickness=0.2, length=3.0, position=1.0
        )
        lower, upper = slit.outlines()
        np.testing.assert_allclose(upper.min(axis=1), [3.9, 1.5])
        np.testing.assert_allclose(upper.max(axis=1), [4.1, 4.5])
        np.testing.assert_allclose(lower.min(axis=1), [3.9, -2.5])
        np.testing.assert_allclose(lower.max(axis=1), [4.1, 0.5])
        assert slit.contains((4.0, 2.0)) and not slit.contains((4.0, 1.0))
        vertical = Aperture2D.slit(4.0, 0.5, 0.2, 3.0, axis=1)
        assert vertical.contains((2.0, 4.0)) and not vertical.contains((0.0, 4.0))
        with pytest.raises(ValueError, match="positive"):
            Aperture2D.slit(0.0, 0.5, 0.0, 1.0)

    def test_contains_in_lower(self) -> None:
        """Test contains method for points in lower polygon."""
        vertices = ((0, 0), (6, 3), (0, 6))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        # Point inside lower polygon
        assert aperture.contains(xp.array([2.0, 8.0]))

    def test_contains_in_upper(self) -> None:
        """Test contains method for points in upper polygon."""
        vertices = ((0, 0), (6, 3), (0, 6))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        # Point inside upper polygon
        assert aperture.contains(xp.array([2.0, 12.0]))

    def test_contains_outside(self) -> None:
        """Test contains method for points outside both polygons."""
        vertices = ((0, 0), (6, 3), (0, 6))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        # Point outside both polygons
        assert not aperture.contains(xp.array([10.0, 10.0]))

    def test_bounds_cover_both_aperture_halves(self) -> None:
        """The aperture bounds must span both lower and upper polygons."""
        vertices = ((0, 0), (6, 3), (0, 6))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        assert aperture.lower_bound == (0.0, 4.0)
        assert aperture.upper_bound == (6.0, 16.0)

    def test_get_indices_in_object(self) -> None:
        """Test getting grid indices inside the aperture."""
        vertices = ((0, 0), (4, 2), (0, 4))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        # Mock grid object
        class MockGrid:
            """Provide a lightweight grid mock for tests."""

            def node_coordinates(self, index):
                """Return node coordinates for a grid whose lower bound is the origin."""
                return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing

            def __init__(self) -> None:
                """Initialize the mock grid instance."""
                self.indices = [
                    (0, 8),
                    (1, 8),
                    (2, 8),
                    (3, 8),
                    (4, 8),
                    (0, 9),
                    (1, 9),
                    (2, 9),
                    (3, 9),
                    (4, 9),
                    (0, 10),
                    (1, 10),
                    (2, 10),
                    (3, 10),
                    (4, 10),
                    (0, 11),
                    (1, 11),
                    (2, 11),
                    (3, 11),
                    (4, 11),
                    (0, 12),
                    (1, 12),
                    (2, 12),
                    (3, 12),
                    (4, 12),
                ]
                self.grid_spacing = 1.0

        grid = MockGrid()
        indices = aperture.get_indices_in_object(grid)

        # Should include indices from both lower and upper polygons
        # Lower polygon: around y=8-10, Upper polygon: around y=10-12
        assert len(indices) > 0
        # Check that all returned indices are in the grid
        for idx in indices:
            assert idx in grid.indices


@pytest.mark.parametrize("inverted", [False, True])
def test_contains_points_matches_contains(inverted: bool) -> None:
    """The vectorised test agrees with the point-by-point one, inside and outside."""
    aperture = Aperture2D(
        axis=0,
        position=0.0,
        vertices=((0.0, 0.0004), (0.002, 0.0004), (0.0025, 0.0007), (0.004, 0.0007)),
        inverted=inverted,
    )
    rng = np.random.default_rng(0)
    lower, upper = (
        np.asarray(aperture.lower_bound)[:2],
        np.asarray(aperture.upper_bound)[:2],
    )
    margin = 0.2 * (upper - lower)
    points = rng.uniform(lower - margin, upper + margin, (4000, 2)).T
    expected = np.array([bool(aperture.contains(point)) for point in points.T])
    np.testing.assert_array_equal(
        xp.to_numpy(aperture.contains_points(points)), expected
    )
