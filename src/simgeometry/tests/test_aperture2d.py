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

    def test_get_vertices_lower(self) -> None:
        """Test getting lower polygon vertices."""
        vertices = ((0, 0), (3, 5), (5, 0))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        lower_vertices = aperture.get_vertices_lower()
        expected = xp.array(
            [
                (0, 10.0),  # z=0, perp=10-0=10
                (3, 5.0),  # z=3, perp=10-5=5
                (5, 10.0),  # z=5, perp=10-0=10
            ],
        )
        assert xp.allclose(lower_vertices, expected)

    def test_get_vertices_upper(self) -> None:
        """Test getting upper polygon vertices."""
        vertices = ((0, 0), (3, 5), (5, 0))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        upper_vertices = aperture.get_vertices_upper()
        expected = xp.array(
            [
                (0, 10.0),  # z=0, perp=10+0=10
                (3, 15.0),  # z=3, perp=10+5=15
                (5, 10.0),  # z=5, perp=10+0=10
            ],
        )
        assert xp.allclose(upper_vertices, expected)

    def test_get_vertices(self) -> None:
        """Test getting all vertices (both polygons)."""
        vertices = ((0, 0), (3, 5), (5, 0))
        aperture = Aperture2D(axis=0, position=10.0, vertices=vertices)

        all_vertices = aperture.get_vertices()
        assert len(all_vertices) == 2  # Two polygons

        # Each should be a PointList with the vertices
        lower_verts = all_vertices[0]
        upper_verts = all_vertices[1]

        # Check coordinates
        lower_coords = lower_verts.get_coordinates()
        upper_coords = upper_verts.get_coordinates()

        expected_lower = [(0.0, 10.0), (3.0, 5.0), (5.0, 10.0)]
        expected_upper = [(0.0, 10.0), (3.0, 15.0), (5.0, 10.0)]

        assert lower_coords == expected_lower
        assert upper_coords == expected_upper

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
