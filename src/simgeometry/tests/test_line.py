import numpy as np

"""Tests for line."""

import cunumpy as xp
import pytest

from simgeometry.objects.line import Line


class TestLine:
    """Group tests for line."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        midpoint = (1.0,)
        side_lengths = (2.0,)
        line = Line(midpoint, side_lengths)

        assert xp.allclose(line.midpoint, [1.0])
        assert xp.allclose(line.side_lengths, [2.0])
        assert line.lower_bound == (0.0,)  # midpoint - half_lengths
        assert line.upper_bound == (2.0,)  # midpoint + half_lengths

    def test_init_invalid_midpoint_length(self) -> None:
        """Test initialization with invalid midpoint length."""
        with pytest.raises(AssertionError, match="Length of midpoint must match ndim"):
            Line((1.0, 2.0), (2.0,))

    def test_init_invalid_side_lengths_length(self) -> None:
        """Test initialization with invalid side_lengths length."""
        with pytest.raises(
            AssertionError,
            match="Length of side_lengths must match ndim",
        ):
            Line((1.0,), (2.0, 4.0))

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        line = Line((1.0,), (2.0,))
        line.mirror(axis=0, position=0.0)

        expected_midpoint = [-1.0]  # 2*0 - 1 = -1
        assert xp.allclose(line.midpoint, expected_midpoint)
        assert line.lower_bound == (-2.0,)
        assert line.upper_bound == (0.0,)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        line = Line((1.0,), (2.0,))
        with pytest.raises(ValueError, match="Axis must be 0"):
            line.mirror(axis=1, position=0.0)

    def test_copy(self) -> None:
        """Test copying the line."""
        original = Line((1.0,), (2.0,), bc="test_bc")
        copied = original.copy()

        assert xp.allclose(copied.midpoint, original.midpoint)
        assert xp.allclose(copied.side_lengths, original.side_lengths)
        assert copied._bc == original._bc
        assert copied is not original  # Different objects
        assert not xp.shares_memory(copied.midpoint, original.midpoint)

    def test_translate(self) -> None:
        """Test translating the line."""
        line = Line((1.0,), (2.0,))
        line.translate((3.0,))  # 1D offset for 1D shape

        expected_midpoint = [4.0]  # 1.0 + 3.0
        assert xp.allclose(line.midpoint, expected_midpoint)
        assert line.lower_bound == (3.0,)  # 4.0 - 1.0
        assert line.upper_bound == (5.0,)  # 4.0 + 1.0

    def test_get_vertices(self) -> None:
        """Test getting vertices of the line."""
        line = Line((1.0,), (2.0,))
        vertices = line.get_vertices()

        expected = [(0.0, 2.0)]  # lower_bound, upper_bound
        assert vertices == expected

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the line."""
        line = Line((1.0,), (2.0,))

        assert line.contains((0.5,))  # Within bounds
        assert line.contains((1.5,))
        assert line.contains((0.0,))  # Boundary
        assert line.contains((2.0,))

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the line."""
        line = Line((1.0,), (2.0,))

        assert not line.contains((-0.1,))
        assert not line.contains((2.1,))

    def test_get_indices_in_object(self) -> None:
        """Test getting grid indices inside the line."""
        line = Line((1.0,), (2.0,))

        # Mock grid object
        class MockGrid:
            """Provide a lightweight grid mock for tests."""

            def node_coordinates(self, index):
                """Return node coordinates for a grid whose lower bound is the origin."""
                return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing

            def __init__(self) -> None:
                """Initialize the mock grid instance."""
                self.indices = [(0,), (1,), (2,), (3,), (4,)]
                self.grid_spacing = 0.5

        grid = MockGrid()
        indices = line.get_indices_in_object(grid)

        # Line spans from 0.0 to 2.0, grid points at 0.0, 0.5, 1.0, 1.5, 2.0
        # All should be inside since bounds are inclusive
        expected_indices = [(0,), (1,), (2,), (3,), (4,)]
        assert indices == expected_indices
