import numpy as np

"""Tests for rectangle."""

import cunumpy as xp
import pytest

from simgeometry.objects.rectangle import Rectangle
from simgeometry.pointlist import PointList


class TestRectangle:
    """Group tests for rectangle."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        midpoint = (1.0, 2.0)
        side_lengths = (2.0, 4.0)
        rect = Rectangle(midpoint, side_lengths)

        assert rect.midpoint == [1.0, 2.0]  # Now a list
        assert rect.side_lengths == side_lengths
        assert rect.lower_bound == (0.0, 0.0)  # midpoint - half_lengths
        assert rect.upper_bound == (2.0, 4.0)  # midpoint + half_lengths

    def test_init_invalid_midpoint_length(self) -> None:
        """Test initialization with invalid midpoint length."""
        with pytest.raises(AssertionError, match="Length of midpoint must match ndim"):
            Rectangle((1.0,), (2.0, 4.0))

    def test_init_invalid_side_lengths_length(self) -> None:
        """Test initialization with invalid side_lengths length."""
        with pytest.raises(
            AssertionError,
            match="Length of side_lengths must match ndim",
        ):
            Rectangle((1.0, 2.0), (2.0,))

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))
        rect.mirror(axis=0, position=0.0)

        expected_midpoint = [-1.0, 2.0]  # 2*0 - 1 = -1
        assert rect.midpoint == expected_midpoint
        assert rect.lower_bound == (-2.0, 0.0)
        assert rect.upper_bound == (0.0, 4.0)

    def test_mirror_y_axis(self) -> None:
        """Test mirroring across y-axis."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))
        rect.mirror(axis=1, position=1.0)

        expected_midpoint = [1.0, 0.0]  # 2*1 - 2 = 0
        assert rect.midpoint == expected_midpoint
        assert rect.lower_bound == (0.0, -2.0)
        assert rect.upper_bound == (2.0, 2.0)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))
        with pytest.raises(ValueError, match="Axis must be 0"):
            rect.mirror(axis=2, position=0.0)

    def test_copy(self) -> None:
        """Test copying the rectangle."""
        original = Rectangle((1.0, 2.0), (2.0, 4.0), bc="test_bc")
        copy_rect = original.copy()

        assert copy_rect.midpoint == original.midpoint
        assert copy_rect.side_lengths == original.side_lengths
        assert copy_rect._bc == original._bc
        assert copy_rect is not original  # Different objects

    def test_translate(self) -> None:
        """Test translating the rectangle."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))
        rect.translate((1.0, -1.0))

        expected_midpoint = [2.0, 1.0]
        assert rect.midpoint == expected_midpoint
        assert rect.lower_bound == (1.0, -1.0)
        assert rect.upper_bound == (3.0, 3.0)

    def test_get_vertices(self) -> None:
        """Test getting vertices of the rectangle."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))
        vertices = rect.get_vertices()

        expected_coords = [
            (0.0, 0.0),  # lower_bound
            (2.0, 0.0),
            (2.0, 4.0),
            (0.0, 4.0),  # upper_bound
        ]
        assert isinstance(vertices, PointList)
        assert len(vertices.points) == 4
        for actual, expected in zip(vertices.points, expected_coords):
            assert actual.coordinates == expected

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the rectangle."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))

        assert rect.contains((1.0, 2.0))  # midpoint
        assert rect.contains((0.0, 0.0))  # lower bound (inclusive)
        assert rect.contains((1.9, 3.9))  # near upper bound

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the rectangle."""
        rect = Rectangle((1.0, 2.0), (2.0, 4.0))

        assert not rect.contains((2.0, 2.0))  # upper bound x (exclusive)
        assert not rect.contains((1.0, 4.0))  # upper bound y (exclusive)
        assert not rect.contains((-0.1, 2.0))  # below lower x
        assert not rect.contains((1.0, -0.1))  # below lower y

    def test_get_indices_in_object(self) -> None:
        """Test getting indices of grid points inside the rectangle."""

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

        rect = Rectangle((1.0, 1.0), (2.0, 2.0))  # Covers from (0,0) to (2,2)
        grid = MockGrid()
        indices = rect.get_indices_in_object(grid)

        expected_indices = [(0, 0), (1, 0), (0, 1), (1, 1)]  # Inside bounds
        assert set(indices) == set(expected_indices)


if __name__ == "__main__":
    pytest.main([__file__])
