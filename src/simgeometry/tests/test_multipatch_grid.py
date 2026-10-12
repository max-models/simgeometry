"""Tests for MultiPatchGridComposite."""

import pytest

from simgeometry.grid import Grid, MultiPatchGridComposite


class TestMultiPatchGridComposite:
    """Tests for MultiPatchGridComposite class."""

    @pytest.fixture
    def single_patch_grid(self):
        """Create a single-patch grid for testing."""
        grid = Grid(
            ndim=2,
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
            num_gridpoints=(11, 11),
        )
        return grid

    @pytest.fixture
    def l_shape_grids(self):
        """Create L-shaped grids."""
        grid_vert = Grid(
            ndim=2,
            lower_bound=(0.0, 0.5),
            upper_bound=(0.5, 2.0),
            num_gridpoints=(6, 16),
        )
        grid_horiz = Grid(
            ndim=2,
            lower_bound=(0.5, 0.0),
            upper_bound=(2.0, 0.5),
            num_gridpoints=(16, 6),
        )
        return {"L_vertical": grid_vert, "L_horizontal": grid_horiz}

    def test_init_single_patch(self, single_patch_grid):
        """Test initialization with single patch."""
        composite = MultiPatchGridComposite({"default": single_patch_grid})
        assert composite.ndim == 2
        assert len(composite.patches) == 1
        assert "default" in composite.patches

    def test_init_multiple_patches(self, l_shape_grids):
        """Test initialization with multiple patches."""
        composite = MultiPatchGridComposite(l_shape_grids)
        assert composite.ndim == 2
        assert len(composite.patches) == 2
        assert composite.patch_ids == ("L_horizontal", "L_vertical")  # Sorted

    def test_init_empty_patches_error(self):
        """Test error on empty patches dict."""
        with pytest.raises(ValueError):
            MultiPatchGridComposite({})

    def test_init_inconsistent_ndim_error(self, single_patch_grid):
        """Test error when patches have different ndim."""
        grid_3d = Grid(
            ndim=3,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            num_gridpoints=(11, 11, 11),
        )
        with pytest.raises(ValueError):
            MultiPatchGridComposite(
                {
                    "2d": single_patch_grid,
                    "3d": grid_3d,
                },
            )

    def test_isinstance_grid(self, single_patch_grid):
        """Test that composite is-a Grid (backward compatibility)."""
        composite = MultiPatchGridComposite({"default": single_patch_grid})
        # Most important: isinstance check should pass
        assert isinstance(composite, Grid)

    def test_find_patch_in_single(self, single_patch_grid):
        """Test patch finding in single-patch domain."""
        composite = MultiPatchGridComposite({"default": single_patch_grid})
        assert composite.find_patch((0.5, 0.5)) == "default"
        assert composite.find_patch((1.1, 0.5)) is None  # Outside

    def test_find_patch_in_l_shape(self, l_shape_grids):
        """Test patch finding in L-shaped domain."""
        composite = MultiPatchGridComposite(l_shape_grids)

        # Vertical leg
        assert composite.find_patch((0.25, 1.0)) == "L_vertical"

        # Horizontal leg
        assert composite.find_patch((1.0, 0.25)) == "L_horizontal"

        # In the gap (not covered)
        assert composite.find_patch((1.0, 1.0)) is None

    def test_global_to_local_single_patch(self, single_patch_grid):
        """Test global->local conversion in single patch."""
        composite = MultiPatchGridComposite({"default": single_patch_grid})
        patch_id, local_pos = composite.global_to_local((0.3, 0.4))

        assert patch_id == "default"
        assert local_pos[0] == pytest.approx(0.3)
        assert local_pos[1] == pytest.approx(0.4)

    def test_global_to_local_l_shape_vertical(self, l_shape_grids):
        """Test global->local in L-shaped vertical leg."""
        composite = MultiPatchGridComposite(l_shape_grids)
        patch_id, local_pos = composite.global_to_local((0.1, 1.0))

        assert patch_id == "L_vertical"
        assert local_pos[0] == pytest.approx(0.1)
        assert local_pos[1] == pytest.approx(0.5)  # 1.0 - 0.5 (lower bound)

    def test_global_to_local_l_shape_horizontal(self, l_shape_grids):
        """Test global->local in L-shaped horizontal leg."""
        composite = MultiPatchGridComposite(l_shape_grids)
        patch_id, local_pos = composite.global_to_local((1.5, 0.2))

        assert patch_id == "L_horizontal"
        assert local_pos[0] == pytest.approx(1.0)  # 1.5 - 0.5
        assert local_pos[1] == pytest.approx(0.2)

    def test_global_to_local_auto_patch_detection(self, l_shape_grids):
        """Test global->local with automatic patch detection."""
        composite = MultiPatchGridComposite(l_shape_grids)

        # Should auto-detect patch
        patch_id1, _local1 = composite.global_to_local((0.2, 1.0))
        assert patch_id1 == "L_vertical"

        # Specify patch explicitly
        patch_id2, _local2 = composite.global_to_local(
            (0.2, 1.0), patch_id="L_vertical"
        )
        assert patch_id2 == "L_vertical"

    def test_global_to_local_position_not_found(self, l_shape_grids):
        """Test error when position not in any patch."""
        composite = MultiPatchGridComposite(l_shape_grids)
        with pytest.raises(ValueError):
            composite.global_to_local((1.0, 1.0))  # Gap in L-shape

    def test_global_to_local_invalid_patch_id(self, l_shape_grids):
        """Test error with invalid patch ID."""
        composite = MultiPatchGridComposite(l_shape_grids)
        with pytest.raises(KeyError):
            composite.global_to_local((0.2, 1.0), patch_id="nonexistent")

    def test_local_to_global_single_patch(self, single_patch_grid):
        """Test local->global conversion."""
        composite = MultiPatchGridComposite({"default": single_patch_grid})
        global_pos = composite.local_to_global((0.3, 0.4), "default")

        assert global_pos[0] == pytest.approx(0.3)
        assert global_pos[1] == pytest.approx(0.4)

    def test_local_to_global_l_shape_vertical(self, l_shape_grids):
        """Test local->global in vertical leg."""
        composite = MultiPatchGridComposite(l_shape_grids)
        global_pos = composite.local_to_global((0.1, 0.5), "L_vertical")

        # Lower bound is (0.0, 0.5), so global = local + lower_bound
        assert global_pos[0] == pytest.approx(0.1)
        assert global_pos[1] == pytest.approx(1.0)  # 0.5 + 0.5

    def test_local_to_global_invalid_patch_id(self, l_shape_grids):
        """Test error with invalid patch ID."""
        composite = MultiPatchGridComposite(l_shape_grids)
        with pytest.raises(KeyError):
            composite.local_to_global((0.1, 0.5), "nonexistent")

    def test_roundtrip_conversion(self, l_shape_grids):
        """Test global->local->global roundtrip."""
        composite = MultiPatchGridComposite(l_shape_grids)
        original = (0.2, 1.5)

        patch_id, local = composite.global_to_local(original)
        restored = composite.local_to_global(local, patch_id)

        assert restored[0] == pytest.approx(original[0])
        assert restored[1] == pytest.approx(original[1])

    def test_get_patch(self, l_shape_grids):
        """Test retrieving child grid."""
        composite = MultiPatchGridComposite(l_shape_grids)
        patch = composite.get_patch("L_vertical")

        assert isinstance(patch, Grid)
        assert patch.lower_bound == (0.0, 0.5)

    def test_get_patch_invalid_id(self, l_shape_grids):
        """Test error retrieving invalid patch."""
        composite = MultiPatchGridComposite(l_shape_grids)
        with pytest.raises(KeyError):
            composite.get_patch("nonexistent")

    def test_get_patch_bounds(self, l_shape_grids):
        """Test retrieving patch bounds."""
        composite = MultiPatchGridComposite(l_shape_grids)
        lower, upper = composite.get_patch_bounds("L_horizontal")

        assert lower == (0.5, 0.0)
        assert upper == (2.0, 0.5)

    def test_repr(self, l_shape_grids):
        """Test string representation."""
        composite = MultiPatchGridComposite(l_shape_grids)
        repr_str = repr(composite)

        assert "MultiPatchGridComposite" in repr_str
        assert "ndim=2" in repr_str
        assert "L_horizontal" in repr_str
        assert "L_vertical" in repr_str
