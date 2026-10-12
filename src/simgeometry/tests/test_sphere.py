import numpy as np

"""Tests for sphere."""

import cunumpy as xp
import pytest

from simgeometry.objects.sphere import Sphere


class TestSphere:
    """Group tests for sphere."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        midpoint = (1.0, 2.0, 3.0)
        radius = 4.0
        sphere = Sphere(midpoint, radius)

        assert sphere.midpoint == [1.0, 2.0, 3.0]
        assert sphere.radius == 4.0
        assert sphere.lower_bound == (-3.0, -2.0, -1.0)  # midpoint - radius
        assert sphere.upper_bound == (5.0, 6.0, 7.0)  # midpoint + radius

    def test_init_invalid_midpoint_length(self) -> None:
        """Test initialization with invalid midpoint length."""
        with pytest.raises(AssertionError, match="Length of midpoint must be 3"):
            Sphere((1.0, 2.0), 4.0)

    def test_init_invalid_radius(self) -> None:
        """Test initialization with invalid radius."""
        with pytest.raises(AssertionError, match="Radius must be a positive number"):
            Sphere((1.0, 2.0, 3.0), -1.0)

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        sphere = Sphere((1.0, 2.0, 3.0), 4.0)
        sphere.mirror(axis=0, position=0.0)

        expected_midpoint = [-1.0, 2.0, 3.0]  # 2*0 - 1 = -1
        assert sphere.midpoint == expected_midpoint
        assert sphere.lower_bound == (-5.0, -2.0, -1.0)
        assert sphere.upper_bound == (3.0, 6.0, 7.0)

    def test_mirror_y_axis(self) -> None:
        """Test mirroring across y-axis."""
        sphere = Sphere((1.0, 2.0, 3.0), 4.0)
        sphere.mirror(axis=1, position=1.0)

        expected_midpoint = [1.0, 0.0, 3.0]  # 2*1 - 2 = 0
        assert sphere.midpoint == expected_midpoint
        assert sphere.lower_bound == (-3.0, -4.0, -1.0)
        assert sphere.upper_bound == (5.0, 4.0, 7.0)

    def test_mirror_z_axis(self) -> None:
        """Test mirroring across z-axis."""
        sphere = Sphere((1.0, 2.0, 3.0), 4.0)
        sphere.mirror(axis=2, position=2.0)

        expected_midpoint = [1.0, 2.0, 1.0]  # 2*2 - 3 = 1
        assert sphere.midpoint == expected_midpoint
        assert sphere.lower_bound == (-3.0, -2.0, -3.0)
        assert sphere.upper_bound == (5.0, 6.0, 5.0)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        sphere = Sphere((1.0, 2.0, 3.0), 4.0)
        with pytest.raises(ValueError, match="Axis must be 0"):
            sphere.mirror(axis=3, position=0.0)

    def test_copy(self) -> None:
        """Test copying the sphere."""
        original = Sphere((1.0, 2.0, 3.0), 4.0, bc="test_bc")
        copy_sphere = original.copy()

        assert copy_sphere.midpoint == original.midpoint
        assert copy_sphere.radius == original.radius
        assert copy_sphere._bc == original._bc
        assert copy_sphere is not original  # Different objects

    def test_translate(self) -> None:
        """Test translating the sphere."""
        sphere = Sphere((1.0, 2.0, 3.0), 4.0)
        sphere.translate((2.0, -1.0, 3.0))

        expected_midpoint = [3.0, 1.0, 6.0]
        assert sphere.midpoint == expected_midpoint
        assert sphere.lower_bound == (-1.0, -3.0, 2.0)
        assert sphere.upper_bound == (7.0, 5.0, 10.0)

    def test_get_vertices_slice_intersect(self) -> None:
        """Test getting vertices for a slice that intersects the sphere."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)
        vertices, inverted = sphere.get_vertices(slice_axis=2, slice_position=0.0)

        # Should return a circle in xy plane
        assert not inverted
        assert vertices.shape == (100, 2)  # N=100 points, x,y coords

    def test_get_vertices_slice_touch(self) -> None:
        """Test getting vertices for a slice that touches the sphere."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)
        vertices, inverted = sphere.get_vertices(slice_axis=2, slice_position=2.0)

        # Should return a single point
        assert not inverted
        assert vertices.shape == (1, 2)  # One point with [x, y]

    def test_get_vertices_slice_no_intersect(self) -> None:
        """Test getting vertices for a slice that doesn't intersect."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)
        vertices, inverted = sphere.get_vertices(slice_axis=2, slice_position=3.0)

        assert not inverted
        assert vertices.size == 0  # Empty array

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the sphere."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)

        assert sphere.contains((0.0, 0.0, 0.0))  # center
        assert sphere.contains((1.0, 1.0, 1.0))  # inside, distance sqrt(3) ≈ 1.732 < 2

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the sphere."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)

        assert not sphere.contains((3.0, 0.0, 0.0))  # outside
        assert not sphere.contains((0.0, 3.0, 0.0))  # outside

    def test_contains_on_boundary(self) -> None:
        """Test contains method for points on the boundary."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)

        assert sphere.contains((2.0, 0.0, 0.0))  # on boundary

    def test_get_indices_in_object(self) -> None:
        """Test getting indices of grid points inside the sphere."""

        # Mock grid object
        class MockGrid:
            """Provide a lightweight grid mock for tests."""

            def node_coordinates(self, index):
                """Return node coordinates for a grid whose lower bound is the origin."""
                return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing

            def __init__(self) -> None:
                """Initialize the mock grid instance."""
                self.indices = [
                    (0, 0, 0),
                    (1, 0, 0),
                    (2, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                    (2, 1, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                    (2, 0, 1),
                ]
                self.grid_spacing = xp.array([1.0, 1.0, 1.0])

        sphere = Sphere((1.0, 1.0, 1.0), 1.5)  # Center at (1,1,1), radius 1.5
        grid = MockGrid()
        indices = sphere.get_indices_in_object(grid)

        # Points within distance 1.5 from (1,1,1)
        expected_indices = [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (2, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (2, 0, 1),
        ]
        assert set(indices) == set(expected_indices)

    def test_volume(self) -> None:
        """Test volume calculation."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)
        expected_volume = (4 / 3) * xp.pi * 8  # r^3 = 8
        assert abs(sphere.volume() - expected_volume) < 1e-10

    def test_surface_area(self) -> None:
        """Test surface area calculation."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0)
        expected_area = 4 * xp.pi * 4  # r^2 = 4
        assert abs(sphere.surface_area() - expected_area) < 1e-10

    def test_inverted_contains(self) -> None:
        """Test inverted sphere containment logic."""
        sphere = Sphere((0.0, 0.0, 0.0), 2.0, inverted=True)
        assert sphere.inverted
        assert not sphere.contains((0.0, 0.0, 0.0))
        assert sphere.contains((3.0, 0.0, 0.0))


if __name__ == "__main__":
    pytest.main([__file__])
