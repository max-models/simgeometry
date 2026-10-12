import numpy as np

"""Tests for cuboid."""

import cunumpy as xp
import pytest

from simgeometry.objects.cuboid import Cuboid


class TestCuboid:
    """Group tests for cuboid."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        midpoint = (1.0, 2.0, 3.0)
        side_lengths = (2.0, 4.0, 6.0)
        cuboid = Cuboid(midpoint, side_lengths)

        assert xp.allclose(cuboid.midpoint, [1.0, 2.0, 3.0])
        assert xp.allclose(cuboid.side_lengths, [2.0, 4.0, 6.0])
        assert cuboid.lower_bound == (0.0, 0.0, 0.0)  # midpoint - half_lengths
        assert cuboid.upper_bound == (2.0, 4.0, 6.0)  # midpoint + half_lengths

    def test_init_invalid_midpoint_length(self) -> None:
        """Test initialization with invalid midpoint length."""
        with pytest.raises(AssertionError, match="Length of midpoint must be 3"):
            Cuboid((1.0, 2.0), (2.0, 4.0, 6.0))

    def test_init_invalid_side_lengths_length(self) -> None:
        """Test initialization with invalid side_lengths length."""
        with pytest.raises(AssertionError, match="Length of side_lengths must be 3"):
            Cuboid((1.0, 2.0, 3.0), (2.0, 4.0))

    def test_mirror_x_axis(self) -> None:
        """Test mirroring across x-axis."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        cuboid.mirror(axis=0, position=0.0)

        expected_midpoint = [-1.0, 2.0, 3.0]  # 2*0 - 1 = -1
        assert xp.allclose(cuboid.midpoint, expected_midpoint)
        assert cuboid.lower_bound == (-2.0, 0.0, 0.0)
        assert cuboid.upper_bound == (0.0, 4.0, 6.0)

    def test_mirror_y_axis(self) -> None:
        """Test mirroring across y-axis."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        cuboid.mirror(axis=1, position=1.0)

        expected_midpoint = [1.0, 0.0, 3.0]  # 2*1 - 2 = 0
        assert xp.allclose(cuboid.midpoint, expected_midpoint)
        assert cuboid.lower_bound == (0.0, -2.0, 0.0)
        assert cuboid.upper_bound == (2.0, 2.0, 6.0)

    def test_mirror_z_axis(self) -> None:
        """Test mirroring across z-axis."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        cuboid.mirror(axis=2, position=2.0)

        expected_midpoint = [1.0, 2.0, 1.0]  # 2*2 - 3 = 1
        assert xp.allclose(cuboid.midpoint, expected_midpoint)
        assert cuboid.lower_bound == (0.0, 0.0, -2.0)
        assert cuboid.upper_bound == (2.0, 4.0, 4.0)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        with pytest.raises(ValueError, match="Axis must be 0"):
            cuboid.mirror(axis=3, position=0.0)

    def test_copy(self) -> None:
        """Test copying the cuboid."""
        original = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), bc="test_bc")
        copied = original.copy()

        assert xp.allclose(copied.midpoint, original.midpoint)
        assert xp.allclose(copied.side_lengths, original.side_lengths)
        assert copied._bc == original._bc
        assert copied is not original  # Different objects
        assert not xp.shares_memory(copied.midpoint, original.midpoint)

    def test_translate(self) -> None:
        """Test translating the cuboid."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        cuboid.translate((3.0, 1.0, 2.0))  # 3D offset

        expected_midpoint = [4.0, 3.0, 5.0]  # 1+3, 2+1, 3+2
        assert xp.allclose(cuboid.midpoint, expected_midpoint)
        assert cuboid.lower_bound == (3.0, 1.0, 2.0)  # 4-1, 3-2, 5-3
        assert cuboid.upper_bound == (5.0, 5.0, 8.0)  # 4+1, 3+2, 5+3

    def test_get_vertices_slice_intersect(self) -> None:
        """Test getting vertices for a slice that intersects the cuboid."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        vertices, inverted = cuboid.get_vertices(slice_axis=2, slice_position=3.0)

        # Should return the rectangle at z=3
        expected_vertices = xp.array(
            [
                [0.0, 0.0],  # lower x, lower y
                [2.0, 0.0],  # upper x, lower y
                [2.0, 4.0],  # upper x, upper y
                [0.0, 4.0],  # lower x, upper y
            ],
        )
        assert not inverted
        assert xp.allclose(vertices, expected_vertices)

    def test_get_vertices_slice_no_intersect(self) -> None:
        """Test getting vertices for a slice outside the cuboid."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        vertices, inverted = cuboid.get_vertices(slice_axis=2, slice_position=10.0)

        # Should return empty array
        assert not inverted
        assert len(vertices) == 0

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the cuboid."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))

        assert cuboid.contains((0.5, 1.0, 2.0))  # Inside
        assert cuboid.contains((1.5, 3.0, 5.0))
        assert cuboid.contains((0.0, 0.0, 0.0))  # Boundary
        assert cuboid.contains((2.0, 4.0, 6.0))

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the cuboid."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))

        assert not cuboid.contains((-0.1, 1.0, 2.0))
        assert not cuboid.contains((2.1, 1.0, 2.0))
        assert not cuboid.contains((1.0, -0.1, 2.0))
        assert not cuboid.contains((1.0, 4.1, 2.0))
        assert not cuboid.contains((1.0, 1.0, -0.1))
        assert not cuboid.contains((1.0, 1.0, 6.1))

    def test_get_indices_in_object(self) -> None:
        """Test getting grid indices inside the cuboid."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))

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
                self.grid_spacing = 1.0

        grid = MockGrid()
        indices = cuboid.get_indices_in_object(grid)

        # Cuboid spans from (0,0,0) to (2,4,6), grid points at integer coordinates
        # All points should be inside
        expected_indices = [
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
        assert set(indices) == set(expected_indices)

    def test_volume(self) -> None:
        """Test volume calculation."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        expected_volume = 2.0 * 4.0 * 6.0  # 48.0
        assert cuboid.volume() == expected_volume

    def test_surface_area(self) -> None:
        """Test surface area calculation."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
        # Surface area = 2*(l*w + w*h + h*l) = 2*(2*4 + 4*6 + 6*2) = 2*(8 + 24 + 12) = 2*44 = 88
        expected_area = 88.0
        assert cuboid.surface_area() == expected_area

    def test_inverted_contains(self) -> None:
        """Test inverted cuboid containment logic."""
        cuboid = Cuboid((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), inverted=True)
        assert cuboid.inverted
        assert not cuboid.contains((1.0, 2.0, 3.0))
        assert cuboid.contains((20.0, 20.0, 20.0))
