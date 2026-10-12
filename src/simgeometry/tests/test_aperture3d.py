import numpy as np

"""Tests for three-dimensional aperture geometry."""

import cunumpy as xp
import pytest

from simgeometry.objects.aperture3d import Aperture3D


class TestAperture3D:
    """Group tests for three-dimensional aperture geometry."""

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        axis = "x"
        center_coords = (1.0, 2.0)
        vertices = [(0, 0.5), (5, 1.0), (10, 0.5)]
        lower_bound = (0, 0, 0)
        upper_bound = (10, 3, 4)

        aperture = Aperture3D(
            axis=axis,
            center_coords=center_coords,
            vertices=vertices,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        assert aperture.axis == axis
        assert xp.allclose(aperture.center_coords, [1.0, 2.0])
        assert xp.allclose(aperture.vertices, xp.array(vertices))
        assert aperture.axis_index == 0
        assert aperture._perpendicular_indices == [1, 2]

    def test_init_invalid_axis(self) -> None:
        """Test initialization with invalid axis."""
        with pytest.raises(AssertionError, match="Axis must be one of"):
            Aperture3D(
                axis="w",
                center_coords=(1.0, 2.0),
                vertices=[(0, 0.5), (5, 1.0)],
                lower_bound=(0, 0, 0),
                upper_bound=(10, 3, 4),
            )

    def test_init_invalid_center_coords_length(self) -> None:
        """Test initialization with invalid center_coords length."""
        with pytest.raises(AssertionError, match="Center coordinates must be a pair"):
            Aperture3D(
                axis="x",
                center_coords=(1.0, 2.0, 3.0),
                vertices=[(0, 0.5), (5, 1.0)],
                lower_bound=(0, 0, 0),
                upper_bound=(10, 3, 4),
            )

    def test_init_invalid_vertex_length(self) -> None:
        """Test initialization with invalid vertex length."""
        with pytest.raises(AssertionError, match="Each vertex must be a list"):
            Aperture3D(
                axis="x",
                center_coords=(1.0, 2.0),
                vertices=[(0,), (5, 1.0)],
                lower_bound=(0, 0, 0),
                upper_bound=(10, 3, 4),
            )

    def test_mirror_perpendicular_axis(self) -> None:
        """Test mirroring across a perpendicular axis."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(1.0, 2.0),
            vertices=[(0, 0.5), (5, 1.0), (10, 0.5)],
            lower_bound=(0, 0, 0),
            upper_bound=(10, 3, 4),
        )

        aperture.mirror(axis=1, position=1.5)  # Mirror across y-axis

        expected_center = [2.0, 2.0]  # y: 2*1.5 - 1.0 = 2.0, z unchanged
        assert xp.allclose(aperture.center_coords, expected_center)

    def test_mirror_aperture_axis(self) -> None:
        """Test mirroring across the aperture's axis."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(1.0, 2.0),
            vertices=[(0, 0.5), (5, 1.0), (10, 0.5)],
            lower_bound=(0, 0, 0),
            upper_bound=(10, 3, 4),
        )

        aperture.mirror(axis=0, position=5.0)  # Mirror across x-axis

        expected_vertices = xp.array(
            [
                (10, 0.5),  # 2*5 - 0 = 10
                (5, 1.0),  # 2*5 - 5 = 5 (unchanged)
                (0, 0.5),  # 2*5 - 10 = 0
            ],
        )
        assert xp.allclose(aperture.vertices, expected_vertices)

    def test_mirror_invalid_axis(self) -> None:
        """Test mirroring with invalid axis."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(1.0, 2.0),
            vertices=[(0, 0.5), (5, 1.0)],
            lower_bound=(0, 0, 0),
            upper_bound=(10, 3, 4),
        )

        with pytest.raises(ValueError, match="Axis must be 0"):
            aperture.mirror(axis=3, position=0.0)

    def test_copy(self) -> None:
        """Test copying the aperture."""
        original = Aperture3D(
            axis="x",
            center_coords=(1.0, 2.0),
            vertices=[(0, 0.5), (5, 1.0)],
            lower_bound=(0, 0, 0),
            upper_bound=(10, 3, 4),
            bc="test_bc",
        )
        copied = original.copy()

        assert copied.axis == original.axis
        assert xp.allclose(copied.center_coords, original.center_coords)
        assert xp.allclose(copied.vertices, original.vertices)
        assert copied._bc == original._bc
        assert copied is not original

    def test_translate(self) -> None:
        """Test translating the aperture."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(1.0, 2.0),
            vertices=[(0, 0.5), (5, 1.0), (10, 0.5)],
            lower_bound=(0, 0, 0),
            upper_bound=(10, 3, 4),
        )

        aperture.translate((2.0, 3.0, 4.0))

        expected_center = [4.0, 6.0]  # y: 1+3, z: 2+4
        expected_vertices = xp.array(
            [
                (2, 0.5),  # x: 0+2
                (7, 1.0),  # x: 5+2
                (12, 0.5),  # x: 10+2
            ],
        )

        assert xp.allclose(aperture.center_coords, expected_center)
        assert xp.allclose(aperture.vertices, expected_vertices)

    def test_get_vertices_slice_aperture_axis(self) -> None:
        """Test getting vertices for a slice along the aperture's axis."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 1.0), (10, 1.0)],
            lower_bound=(0, -2, -2),
            upper_bound=(10, 2, 2),
        )

        vertices, inverted = aperture.get_vertices(slice_axis=0, slice_position=5.0)

        assert inverted
        assert len(vertices) == 100  # Circle approximation with 100 points
        # Check that all points are at distance 1.0 from center (0,0)
        distances = xp.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
        assert xp.allclose(distances, 1.0, atol=1e-10)

    def test_get_vertices_slice_perpendicular_axis(self) -> None:
        """Test getting vertices for a slice along a perpendicular axis."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 2.0), (10, 2.0)],
            lower_bound=(0, -3, -3),
            upper_bound=(10, 3, 3),
        )

        vertices, inverted = aperture.get_vertices(slice_axis=1, slice_position=0.0)

        assert inverted
        assert len(vertices) > 0
        # Should be a shape representing the cross-section

    def test_get_vertices_slice_outside_bounds(self) -> None:
        """Test getting vertices for a slice outside the aperture bounds."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 1.0), (10, 1.0)],
            lower_bound=(0, -2, -2),
            upper_bound=(10, 2, 2),
        )

        vertices, inverted = aperture.get_vertices(slice_axis=0, slice_position=15.0)

        assert inverted
        assert len(vertices) == 0

    def test_contains_inside(self) -> None:
        """Test contains method for points inside the aperture."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 2.0), (10, 2.0)],
            lower_bound=(0, -3, -3),
            upper_bound=(10, 3, 3),
        )

        # Point far from center should be inside (outside the radial boundary)
        assert aperture.contains((5.0, 2.9, 0.0))

    def test_contains_outside(self) -> None:
        """Test contains method for points outside the aperture."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 2.0), (10, 2.0)],
            lower_bound=(0, -3, -3),
            upper_bound=(10, 3, 3),
        )

        # Point close to center should be outside (inside the radial boundary)
        assert not aperture.contains((5.0, 0.5, 0.0))

    def test_contains_outside_bounds(self) -> None:
        """Test contains method for points outside bounding box."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 2.0), (10, 2.0)],
            lower_bound=(0, -3, -3),
            upper_bound=(10, 3, 3),
        )

        # Point outside bounding box
        assert not aperture.contains((15.0, 0.0, 0.0))

    def test_get_indices_in_object(self) -> None:
        """Test getting grid indices inside the aperture."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(5.0, 5.0),
            vertices=[(0, 1.0), (10, 1.0)],
            lower_bound=(0, 3, 3),
            upper_bound=(10, 7, 7),
        )

        # Mock grid object
        class MockGrid:
            """Provide a lightweight grid mock for tests."""

            def node_coordinates(self, index):
                """Return node coordinates for a grid whose lower bound is the origin."""
                return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing

            def __init__(self) -> None:
                """Initialize the mock grid instance."""
                self.indices = [
                    (5, 4, 4),
                    (5, 5, 4),
                    (5, 6, 4),
                    (5, 4, 5),
                    (5, 5, 5),
                    (5, 6, 5),
                    (5, 4, 6),
                    (5, 5, 6),
                    (5, 6, 6),
                ]
                self.grid_spacing = 1.0

        grid = MockGrid()
        indices = aperture.get_indices_in_object(grid)

        # Points far from center (5,5) in yz-plane should be inside
        # Distance > 1.0 from (5,5) in yz-plane
        expected_indices = [
            (5, 4, 4),
            (5, 6, 4),
            (5, 4, 6),
            (5, 6, 6),
        ]
        assert set(indices) == set(expected_indices)

    def test_volume_not_implemented(self) -> None:
        """Test that volume raises NotImplementedError."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 1.0), (10, 1.0)],
            lower_bound=(0, -2, -2),
            upper_bound=(10, 2, 2),
        )

        with pytest.raises(
            NotImplementedError,
            match="Volume calculation not implemented",
        ):
            aperture.volume()

    def test_surface_area_not_implemented(self) -> None:
        """Test that surface_area raises NotImplementedError."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 1.0), (10, 1.0)],
            lower_bound=(0, -2, -2),
            upper_bound=(10, 2, 2),
        )

        with pytest.raises(
            NotImplementedError,
            match="Surface area calculation not implemented",
        ):
            aperture.surface_area()

    def test_inverted_toggle_contains(self) -> None:
        """Test toggling aperture inversion."""
        aperture = Aperture3D(
            axis="x",
            center_coords=(0.0, 0.0),
            vertices=[(0, 2.0), (10, 2.0)],
            lower_bound=(0, -3, -3),
            upper_bound=(10, 3, 3),
            inverted=False,
        )

        assert not aperture.inverted
        assert not aperture.contains((5.0, 2.9, 0.0))
        assert aperture.contains((5.0, 0.5, 0.0))


@pytest.mark.parametrize(
    ("axis", "vertices", "inverted"),
    [
        ("x", ((0.4, 0.2), (0.6, 0.2)), True),
        ("x", ((0.4, 0.2), (0.5, 0.1), (0.6, 0.3)), True),
        ("y", ((0.3, 0.15), (0.7, 0.35)), False),
        ("z", ((0.5, 0.25),), True),
    ],
)
def test_contains_points_agrees_with_contains(axis, vertices, inverted) -> None:
    """The vectorized test used for markers equals the point-by-point one."""
    import numpy as np

    aperture = Aperture3D(
        axis=axis,
        center_coords=(0.5, 0.5),
        vertices=vertices,
        lower_bound=(0.2, 0.1, 0.0),
        upper_bound=(0.8, 0.9, 1.0),
        bc=0.0,
        inverted=inverted,
    )
    points = np.random.default_rng(0).uniform(-0.1, 1.1, (3000, 3))
    points[:100, 0] = 0.2  # on the half-open box faces
    points[100:200, 0] = 0.8
    points[200:300, aperture.axis_index] = vertices[0][0]  # on a vertex plane
    points = points.T  # markers are component-major
    expected = np.array([aperture.contains(point) for point in points.T])
    assert expected.any()
    np.testing.assert_array_equal(aperture.contains_points(points), expected)
    assert aperture.solid_within_bounding_box
