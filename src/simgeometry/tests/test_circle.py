import numpy as np

"""Tests for circle."""

import cunumpy as xp
import pytest

from simgeometry.objects.circle import Circle


class TestCircle:
    """Group tests for circle."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        midpoint = (1.0, 2.0)
        radius = 3.0
        circle = Circle(midpoint, radius)

        assert circle.midpoint == [1.0, 2.0]
        assert circle.radius == 3.0
        assert circle.lower_bound == (-2.0, -1.0)  # midpoint - radius
        assert circle.upper_bound == (4.0, 5.0)  # midpoint + radius

    def test_init_invalid_midpoint_length(self) -> None:
        """Test initialization with invalid midpoint length."""
        with pytest.raises(AssertionError, match="Midpoint must have two coordinates"):
            Circle((1.0,), 3.0)

    def test_init_invalid_radius(self) -> None:
        """Test initialization with invalid radius."""
        with pytest.raises(AssertionError, match="Radius must be a positive number"):
            Circle((1.0, 2.0), -1.0)

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        circle = Circle((1.0, 2.0), 3.0)
        circle.mirror(axis=0, position=0.0)

        expected_midpoint = [-1.0, 2.0]  # 2*0 - 1 = -1
        assert circle.midpoint == expected_midpoint
        assert circle.lower_bound == (-4.0, -1.0)
        assert circle.upper_bound == (2.0, 5.0)

    def test_mirror_y_axis(self) -> None:
        """Test mirroring across y-axis."""
        circle = Circle((1.0, 2.0), 3.0)
        circle.mirror(axis=1, position=1.0)

        expected_midpoint = [1.0, 0.0]  # 2*1 - 2 = 0
        assert circle.midpoint == expected_midpoint
        assert circle.lower_bound == (-2.0, -3.0)
        assert circle.upper_bound == (4.0, 3.0)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        circle = Circle((1.0, 2.0), 3.0)
        with pytest.raises(ValueError, match="Axis must be 0"):
            circle.mirror(axis=2, position=0.0)

    def test_copy(self) -> None:
        """Test copying the circle."""
        original = Circle((1.0, 2.0), 3.0, bc="test_bc")
        copy_circle = original.copy()

        assert copy_circle.midpoint == original.midpoint
        assert copy_circle.radius == original.radius
        assert copy_circle._bc == original._bc
        assert copy_circle is not original  # Different objects

    def test_translate(self) -> None:
        """Test translating the circle."""
        circle = Circle((1.0, 2.0), 3.0)
        circle.translate((2.0, -1.0))

        expected_midpoint = [3.0, 1.0]
        assert circle.midpoint == expected_midpoint
        assert circle.lower_bound == (0.0, -2.0)
        assert circle.upper_bound == (6.0, 4.0)

    def test_outline_vertices_lie_on_the_circle(self) -> None:
        """The outline is a counter-clockwise loop on the circle."""
        circle = Circle((0.0, 0.0), 1.0)
        (loop,) = circle.outlines(max_spacing=2 * np.pi / 4)
        np.testing.assert_allclose(
            loop, [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]], atol=1e-12
        )
        assert circle.exact_outline is False

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the circle."""
        circle = Circle((0.0, 0.0), 2.0)

        assert circle.contains((0.0, 0.0))  # center
        assert circle.contains((1.0, 1.0))  # inside, distance sqrt(2) ≈ 1.414 < 2
        assert circle.contains((1.4, 1.4))  # inside, distance sqrt(3.92) ≈ 1.98 < 2

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the circle."""
        circle = Circle((0.0, 0.0), 2.0)

        assert not circle.contains((3.0, 0.0))  # outside
        assert not circle.contains((0.0, 3.0))  # outside
        assert not circle.contains((2.1, 0.0))  # just outside

    def test_contains_on_boundary(self) -> None:
        """Test contains method for points on the boundary."""
        circle = Circle((0.0, 0.0), 2.0)

        assert circle.contains((2.0, 0.0))  # on boundary
        assert circle.contains((0.0, 2.0))  # on boundary

    def test_get_indices_in_object(self) -> None:
        """Test getting indices of grid points inside the circle."""

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

        circle = Circle((1.0, 1.0), 1.5)  # Center at (1,1), radius 1.5
        grid = MockGrid()
        indices = circle.get_indices_in_object(grid)

        # Points within distance 1.5 from (1,1)
        expected_indices = [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ]  # All within
        assert set(indices) == set(expected_indices)


if __name__ == "__main__":
    pytest.main([__file__])
