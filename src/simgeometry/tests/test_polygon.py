import numpy as np

"""Tests for polygon."""

import cunumpy as xp
import pytest

from simgeometry.objects.polygon import Polygon
from simgeometry.pointlist import PointList


class TestPolygon:
    """Group tests for polygon."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        polygon = Polygon(vertices)

        assert xp.array_equal(polygon.vertices, xp.array(vertices))
        assert polygon.lower_bound == (0.0, 0.0)
        assert polygon.upper_bound == (1.0, 1.0)

    def test_init_invalid_vertex_count(self) -> None:
        """Test initialization with insufficient vertices."""
        with pytest.raises(
            AssertionError,
            match="Polygon must have at least three vertices",
        ):
            Polygon([(0.0, 0.0), (1.0, 0.0)])  # Only 2 vertices

    def test_init_invalid_vertex_dimensions(self) -> None:
        """Test initialization with vertices having wrong dimensions."""
        with pytest.raises(
            AssertionError,
            match="Each vertex must have two coordinates",
        ):
            Polygon([(0.0, 0.0), (1.0,), (1.0, 1.0)])  # Second vertex has 1 coord

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        polygon = Polygon(vertices)
        polygon.mirror(axis=0, position=0.5)

        # Mirror over x=0.5, so y coords: 0->1, 0->1, 1->0, 1->0
        expected_vertices = xp.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        assert xp.array_equal(polygon.vertices, expected_vertices)
        assert polygon.lower_bound == (0.0, 0.0)
        assert polygon.upper_bound == (1.0, 1.0)

    def test_mirror_y_axis(self) -> None:
        """Test mirroring across y-axis."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        polygon = Polygon(vertices)
        polygon.mirror(axis=1, position=0.5)

        # Mirror over y=0.5, so x coords: 0->1, 1->0, 1->0, 0->1
        expected_vertices = xp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert xp.array_equal(polygon.vertices, expected_vertices)
        assert polygon.lower_bound == (0.0, 0.0)
        assert polygon.upper_bound == (1.0, 1.0)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        polygon = Polygon(vertices)
        with pytest.raises(AssertionError, match="Axis must be 0"):
            polygon.mirror(axis=2, position=0.0)

    def test_copy(self) -> None:
        """Test copying the polygon."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        original = Polygon(vertices, bc="test_bc")
        copy_polygon = original.copy()

        assert xp.array_equal(copy_polygon.vertices, original.vertices)
        assert copy_polygon.vertices is not original.vertices  # Different arrays
        assert copy_polygon._bc == original._bc
        assert copy_polygon is not original

    def test_translate(self) -> None:
        """Test translating the polygon."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        polygon = Polygon(vertices)
        polygon.translate((2.0, 3.0))

        expected_vertices = xp.array([[2.0, 3.0], [3.0, 3.0], [3.0, 4.0], [2.0, 4.0]])
        assert xp.array_equal(polygon.vertices, expected_vertices)
        assert polygon.lower_bound == (2.0, 3.0)
        assert polygon.upper_bound == (3.0, 4.0)

    def test_get_vertices(self) -> None:
        """Test getting vertices of the polygon."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        polygon = Polygon(vertices)
        verts = polygon.get_vertices()

        assert isinstance(verts, PointList)
        assert len(verts.points) == 4
        expected_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for actual, expected in zip(verts.points, expected_coords):
            assert actual.coordinates == expected

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the polygon."""
        vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        polygon = Polygon(vertices)

        assert polygon.contains((1.0, 1.0))  # Center
        assert polygon.contains((0.5, 0.5))  # Inside
        assert polygon.contains((1.5, 1.5))  # Inside

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the polygon."""
        vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        polygon = Polygon(vertices)

        assert not polygon.contains((3.0, 1.0))  # Outside
        assert not polygon.contains((-1.0, 1.0))  # Outside
        assert not polygon.contains((1.0, -1.0))  # Outside

    def test_contains_on_boundary(self) -> None:
        """Test contains method for points on the boundary."""
        vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        polygon = Polygon(vertices)

        # Ray casting typically excludes boundary, but check implementation
        # Assuming it includes or excludes based on code
        assert polygon.contains((1.0, 0.0))  # On bottom edge

    def test_get_indices_in_object(self) -> None:
        """Test getting indices of grid points inside the polygon."""

        # Mock grid object
        class MockGrid:
            """Provide a lightweight grid mock for tests."""

            def node_coordinates(self, index):
                """Return node coordinates for a grid whose lower bound is the origin."""
                return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing

            def __init__(self) -> None:
                """Initialize the mock grid instance."""
                self.indices = [
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                    (1, 1),
                    (2, 1),
                    (0, 2),
                    (1, 2),
                    (2, 2),
                ]
                self.grid_spacing = xp.array([1.0, 1.0])

        vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        polygon = Polygon(vertices)
        grid = MockGrid()
        indices = polygon.get_indices_in_object(grid)

        # Points inside the square (based on ray casting)
        expected_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert set(indices) == set(expected_indices)


if __name__ == "__main__":
    pytest.main([__file__])
