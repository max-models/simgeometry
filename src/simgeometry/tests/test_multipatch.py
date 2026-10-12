"""Tests for multi-patch geometry utilities."""

import pytest

from simgeometry.multipatch import PatchDescriptor, PatchGeometry, PatchLibrary


class TestPatchDescriptor:
    """Tests for PatchDescriptor dataclass."""

    def test_init_valid_2d(self) -> None:
        """Test valid 2D patch creation."""
        patch = PatchDescriptor(
            name="test_patch",
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
            num_gridpoints=(11, 11),
        )
        assert patch.name == "test_patch"
        assert patch.ndim == 2
        assert patch.sidelengths == (1.0, 1.0)

    def test_init_valid_3d(self) -> None:
        """Test valid 3D patch creation."""
        patch = PatchDescriptor(
            name="test_patch_3d",
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(2.0, 3.0, 4.0),
            num_gridpoints=(21, 31, 41),
        )
        assert patch.ndim == 3
        assert patch.sidelengths == (2.0, 3.0, 4.0)

    def test_init_with_resolution(self) -> None:
        """Test patch with explicit resolution."""
        patch = PatchDescriptor(
            name="test",
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
            num_gridpoints=(11, 11),
            resolution=0.1,
        )
        assert patch.resolution == 0.1

    def test_init_invalid_bounds_length(self) -> None:
        """Test error when bounds lengths don't match."""
        with pytest.raises(ValueError):
            PatchDescriptor(
                name="bad",
                lower_bound=(0.0, 0.0),
                upper_bound=(1.0,),  # Length mismatch
                num_gridpoints=(11, 11),
            )

    def test_init_invalid_gridpoints_length(self) -> None:
        """Test error when gridpoints length doesn't match bounds."""
        with pytest.raises(ValueError):
            PatchDescriptor(
                name="bad",
                lower_bound=(0.0, 0.0),
                upper_bound=(1.0, 1.0),
                num_gridpoints=(11,),  # Length mismatch
            )

    def test_init_invalid_bounds_order(self) -> None:
        """Test error when upper_bound <= lower_bound."""
        with pytest.raises(ValueError):
            PatchDescriptor(
                name="bad",
                lower_bound=(1.0, 0.0),
                upper_bound=(0.0, 1.0),  # upper < lower in x
                num_gridpoints=(11, 11),
            )

    def test_init_invalid_bounds_equal(self) -> None:
        """Test error when upper_bound == lower_bound."""
        with pytest.raises(ValueError):
            PatchDescriptor(
                name="bad",
                lower_bound=(0.0, 0.0),
                upper_bound=(0.0, 1.0),  # equal in x
                num_gridpoints=(11, 11),
            )


class TestPatchLibrary:
    """Tests for PatchLibrary geometry factory."""

    def test_single_patch(self) -> None:
        """Test single-patch (backward compatible) mode."""
        patches = PatchLibrary.single(
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
            num_gridpoints=(11, 11),
        )
        assert len(patches) == 1
        assert "default" in patches
        patch = patches["default"]
        assert patch.lower_bound == (0.0, 0.0)
        assert patch.upper_bound == (1.0, 1.0)

    def test_l_shape_2d(self) -> None:
        """Test L-shaped 2D geometry."""
        patches = PatchLibrary.l_shape_2d(
            vertical_length=2.0,
            horizontal_length=2.0,
            thickness=0.5,
            num_points_per_unit=10,
        )
        assert len(patches) == 2
        assert "L_vertical" in patches
        assert "L_horizontal" in patches

        vert = patches["L_vertical"]
        horiz = patches["L_horizontal"]

        # Vertical leg: x in [0, 0.5], y in [0, 2.0]
        assert vert.lower_bound == (0.0, 0.0)
        assert vert.upper_bound == (0.5, 2.0)

        # Horizontal leg: x in [0, 2.0], y in [0, 0.5]
        # The overlap at [0, 0.5] x [0, 0.5] connects the full side
        assert horiz.lower_bound == (0.0, 0.0)
        assert horiz.upper_bound == (2.0, 0.5)

    def test_l_shape_memory_efficient(self) -> None:
        """Verify L-shape uses less memory than bounding box."""
        # L-shape with thickness 0.5 in 2x2 domain
        patches = PatchLibrary.l_shape_2d(
            vertical_length=2.0,
            horizontal_length=2.0,
            thickness=0.5,
            num_points_per_unit=10,
        )
        vert = patches["L_vertical"]
        horiz = patches["L_horizontal"]

        vert_cells = (vert.num_gridpoints[0] - 1) * (vert.num_gridpoints[1] - 1)
        horiz_cells = (horiz.num_gridpoints[0] - 1) * (horiz.num_gridpoints[1] - 1)
        total_l_cells = vert_cells + horiz_cells

        # Bounding box would be 2x2
        bbox_cells = 20 * 20  # 21x21 gridpoints

        # L-shape should use significantly less
        assert total_l_cells < bbox_cells * 0.6  # Less than 60%

    def test_t_shape_2d(self) -> None:
        """Test T-shaped 2D geometry."""
        patches = PatchLibrary.t_shape_2d(
            stem_length=2.0,
            stem_thickness=0.5,
            head_width=1.5,
            head_height=0.5,
            num_points_per_unit=10,
        )
        assert len(patches) == 2
        assert "T_stem" in patches
        assert "T_head" in patches

    def test_cross_2d(self) -> None:
        """Test cross-shaped 2D geometry."""
        patches = PatchLibrary.cross_2d(
            arm_length=2.0,
            arm_thickness=0.5,
            num_points_per_unit=10,
        )
        assert len(patches) == 2
        assert "Cross_horizontal" in patches
        assert "Cross_vertical" in patches

    def test_create_factory_by_enum(self) -> None:
        """Test create() factory method with enum."""
        patches = PatchLibrary.create(
            PatchGeometry.L_SHAPE,
            vertical_length=1.0,
            horizontal_length=1.0,
            thickness=0.25,
        )
        assert len(patches) == 2

    def test_create_factory_by_string(self) -> None:
        """Test create() factory method with string."""
        patches = PatchLibrary.create(
            "l_shape",
            vertical_length=1.0,
            horizontal_length=1.0,
            thickness=0.25,
        )
        assert len(patches) == 2

    def test_create_factory_single(self) -> None:
        """Test create() factory with SINGLE geometry."""
        patches = PatchLibrary.create(
            PatchGeometry.SINGLE,
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
            num_gridpoints=(11, 11),
        )
        assert len(patches) == 1

    def test_create_factory_invalid_geometry(self) -> None:
        """Test error on invalid geometry name."""
        with pytest.raises(ValueError):
            PatchLibrary.create("invalid_geometry")

    def test_patch_geometry_enum(self) -> None:
        """Test PatchGeometry enum values."""
        assert PatchGeometry.SINGLE.value == "single"
        assert PatchGeometry.L_SHAPE.value == "l_shape"
        assert PatchGeometry.T_SHAPE.value == "t_shape"
        assert PatchGeometry.CROSS.value == "cross"
