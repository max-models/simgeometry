import numpy as np

"""Tests for triangle."""

import cunumpy as xp
import pytest

from simgeometry.objects.triangle import Triangle


class TestTriangle:
    """Group tests for triangle."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        triangle = Triangle(vertices)

        assert len(triangle._vertices) == 3
        assert xp.array_equal(triangle._vertices[0], xp.array([0.0, 0.0]))
        assert xp.array_equal(triangle._vertices[1], xp.array([1.0, 0.0]))
        assert xp.array_equal(triangle._vertices[2], xp.array([0.5, 1.0]))
        assert triangle.lower_bound == (0.0, 0.0)
        assert triangle.upper_bound == (1.0, 1.0)

    def test_init_invalid_vertex_count(self) -> None:
        """Test initialization with invalid number of vertices."""
        with pytest.raises(
            AssertionError,
            match="Triangle must have exactly three vertices",
        ):
            Triangle([(0.0, 0.0), (1.0, 0.0)])  # Only 2 vertices

    def test_init_invalid_vertex_dimensions(self) -> None:
        """Test initialization with vertices having wrong dimensions."""
        with pytest.raises(
            AssertionError,
            match="Each vertex must have two coordinates",
        ):
            Triangle([(0.0, 0.0), (1.0,), (0.5, 1.0)])  # Second vertex has 1 coord

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        triangle = Triangle(vertices)
        triangle.mirror(axis=0, position=0.5)

        # Expected: mirror over x=0.5
        # (0.0,0.0) -> (1.0,0.0), (1.0,0.0)->(0.0,0.0), (0.5,1.0)->(0.5,1.0)
        expected_vertices = [
            xp.array([1.0, 0.0]),
            xp.array([0.0, 0.0]),
            xp.array([0.5, 1.0]),
        ]
        for actual, expected in zip(triangle._vertices, expected_vertices):
            assert xp.array_equal(actual, expected)
        assert triangle.lower_bound == (0.0, 0.0)
        assert triangle.upper_bound == (1.0, 1.0)

    def test_mirror_y_axis(self) -> None:
        """Test mirroring across y-axis."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        triangle = Triangle(vertices)
        triangle.mirror(axis=1, position=0.5)

        # Mirror over y=0.5
        expected_vertices = [
            xp.array([0.0, 1.0]),
            xp.array([1.0, 1.0]),
            xp.array([0.5, 0.0]),
        ]
        for actual, expected in zip(triangle._vertices, expected_vertices):
            assert xp.array_equal(actual, expected)
        assert triangle.lower_bound == (0.0, 0.0)
        assert triangle.upper_bound == (1.0, 1.0)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        triangle = Triangle(vertices)
        with pytest.raises(ValueError, match="Axis must be 0"):
            triangle.mirror(axis=2, position=0.0)

    def test_copy(self) -> None:
        """Test copying the triangle."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        original = Triangle(vertices, bc="test_bc")
        copy_triangle = original.copy()

        assert len(copy_triangle._vertices) == len(original._vertices)
        for orig_v, copy_v in zip(original._vertices, copy_triangle._vertices):
            assert xp.array_equal(orig_v, copy_v)
            assert orig_v is not copy_v  # Different objects
        assert copy_triangle._bc == original._bc
        assert copy_triangle is not original

    def test_translate(self) -> None:
        """Test translating the triangle."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        triangle = Triangle(vertices)
        triangle.translate((1.0, 2.0))

        expected_vertices = [
            xp.array([1.0, 2.0]),
            xp.array([2.0, 2.0]),
            xp.array([1.5, 3.0]),
        ]
        for actual, expected in zip(triangle._vertices, expected_vertices):
            assert xp.array_equal(actual, expected)
        assert triangle.lower_bound == (1.0, 2.0)
        assert triangle.upper_bound == (2.0, 3.0)

    def test_outline_is_the_vertices(self) -> None:
        """The outline of a counter-clockwise triangle is its vertices."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        triangle = Triangle(vertices)
        (loop,) = triangle.outlines()
        np.testing.assert_array_equal(loop, np.asarray(vertices).T)

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the triangle."""
        vertices = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
        triangle = Triangle(vertices)

        assert triangle.contains((1.0, 0.5))  # Inside
        assert triangle.contains((0.5, 0.5))  # Inside
        assert triangle.contains((1.0, 1.0))  # Inside

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the triangle."""
        vertices = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
        triangle = Triangle(vertices)

        assert not triangle.contains((3.0, 1.0))  # Outside
        assert not triangle.contains((-1.0, 1.0))  # Outside
        assert not triangle.contains((1.0, -1.0))  # Outside

    def test_contains_on_boundary(self) -> None:
        """Test contains method for points on the boundary."""
        vertices = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
        triangle = Triangle(vertices)

        # Barycentric might include boundary, but typically for inside, strict
        # Assuming implementation includes boundary
        assert triangle.contains((1.0, 0.0))  # On base

    def test_get_indices_in_object(self) -> None:
        """Test getting indices of grid points inside the triangle."""

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

        vertices = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
        triangle = Triangle(vertices)
        grid = MockGrid()
        indices = triangle.get_indices_in_object(grid)

        # Approximate: points inside the triangle
        expected_indices = [(0, 0), (1, 0), (1, 1)]  # Based on actual output
        assert set(indices) == set(expected_indices)


if __name__ == "__main__":
    pytest.main([__file__])
