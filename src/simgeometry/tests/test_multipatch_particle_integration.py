"""Tests for multi-patch particle tracing functionality."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry.grid import Grid, MultiPatchGridComposite
from simgeometry.multipatch import PatchGeometry, PatchLibrary


class TestMultiPatchParticleTracer:
    """Tests for MultiPatchParticleTracer class."""

    @pytest.fixture
    def simple_l_shape_grid(self):
        """Create a simple L-shaped grid for testing."""
        grid_vert = Grid(
            ndim=2,
            lower_bound=(0.0, 0.0),
            upper_bound=(0.5, 2.0),
            num_gridpoints=(6, 21),
        )
        grid_horiz = Grid(
            ndim=2,
            lower_bound=(0.0, 0.0),
            upper_bound=(2.0, 0.5),
            num_gridpoints=(21, 6),
        )
        return MultiPatchGridComposite(
            {"L_vertical": grid_vert, "L_horizontal": grid_horiz},
        )

    def test_grid_creation(self, simple_l_shape_grid):
        """Test that multi-patch grid is created correctly."""
        grid = simple_l_shape_grid
        assert grid.ndim == 2
        assert len(grid.patches) == 2
        assert isinstance(grid, Grid)  # Backward compatibility

    def test_find_patch_in_domain(self, simple_l_shape_grid):
        """Test finding patches at various positions."""
        grid = simple_l_shape_grid

        # In vertical leg
        patch = grid.find_patch((0.25, 1.0))
        assert patch == "L_vertical"

        # In horizontal leg
        patch = grid.find_patch((1.0, 0.25))
        assert patch == "L_horizontal"

        # In overlap region
        patch = grid.find_patch((0.25, 0.25))
        assert patch is not None  # Should be in one of the patches

        # Outside domain
        patch = grid.find_patch((-1.0, -1.0))
        assert patch is None

    def test_coordinate_mapping(self, simple_l_shape_grid):
        """Test global to local coordinate mapping."""
        grid = simple_l_shape_grid

        # Global to local in vertical patch
        _patch_id, pos_local = grid.global_to_local((0.25, 1.0), "L_vertical")
        assert pos_local is not None
        assert isinstance(pos_local, np.ndarray)

        # Local back to global
        pos_global = grid.local_to_global(pos_local, "L_vertical")
        np.testing.assert_array_almost_equal(pos_global, (0.25, 1.0))

    def test_boundary_detection(self, simple_l_shape_grid):
        """Test detection of particles leaving the domain."""
        grid = simple_l_shape_grid

        # Just inside domain
        patch = grid.find_patch((0.0001, 0.0001))
        assert patch is not None

        # Just outside domain
        patch = grid.find_patch((-0.01, 0.5))
        assert patch is None


class TestParticleTrajectoryIntegration:
    """Tests for particle trajectory integration."""

    @pytest.fixture
    def test_particle_dict(self):
        """Create a test particle dictionary."""
        return {
            "id": 0,
            "pos": np.array([0.5, 0.5], dtype=float),
            "vel": np.array([0.1, 0.0], dtype=float),
            "charge": -1.0,
            "mass": 1.0,
            "trajectory": [np.array([0.5, 0.5])],
            "times": [0.0],
            "patch_history": ["L_horizontal"],
            "active": True,
        }

    def test_particle_initialization(self, test_particle_dict):
        """Test particle initialization."""
        p = test_particle_dict
        assert p["id"] == 0
        assert p["charge"] == -1.0
        assert p["active"] is True
        assert len(p["trajectory"]) == 1

    def test_particle_position_update(self, test_particle_dict):
        """Test particle position update logic."""
        p = test_particle_dict
        dt = 0.01

        # Simple uniform motion: x_new = x + v * dt
        x_new = p["pos"] + p["vel"] * dt
        expected = np.array([0.5 + 0.1 * 0.01, 0.5])

        np.testing.assert_array_almost_equal(x_new, expected)

    def test_particle_velocity_verlet(self, test_particle_dict):
        """Test velocity Verlet integration scheme."""
        p = test_particle_dict
        dt = 0.01

        # Simple constant acceleration
        acc = np.array([0.01, 0.0])

        # v_half = v + 0.5 * a * dt
        v_half = p["vel"] + 0.5 * acc * dt
        # x_new = x + v_half * dt
        x_new = p["pos"] + v_half * dt
        # v_new = v_half + 0.5 * a * dt (with constant a)
        v_new = v_half + 0.5 * acc * dt

        # After one Verlet step:
        # Position should advance
        assert x_new[0] > p["pos"][0]
        # Velocity should increase in x direction
        assert v_new[0] > p["vel"][0]


class TestPatchGeometries:
    """Tests for different patch geometries."""

    def test_l_shape_geometry(self):
        """Test L-shape patch descriptor."""
        patches = PatchLibrary.l_shape_2d(
            vertical_length=2.0,
            horizontal_length=2.0,
            thickness=0.5,
            num_points_per_unit=10,
        )
        assert len(patches) == 2
        assert "L_vertical" in patches
        assert "L_horizontal" in patches

        # Verify patches share an edge (overlap region)
        vert = patches["L_vertical"]
        horiz = patches["L_horizontal"]

        # Both should start at origin
        assert vert.lower_bound[0] == 0.0
        assert horiz.lower_bound[0] == 0.0
        assert vert.lower_bound[1] == 0.0
        assert horiz.lower_bound[1] == 0.0

    def test_t_shape_geometry(self):
        """Test T-shape patch descriptor."""
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

    def test_cross_geometry(self):
        """Test cross patch descriptor."""
        patches = PatchLibrary.cross_2d(
            arm_length=2.0,
            arm_thickness=0.5,
            num_points_per_unit=10,
        )
        assert len(patches) == 2
        assert "Cross_horizontal" in patches or "Cross_vertical" in patches

    def test_create_composite_grids(self):
        """Test creating composite grids from patch descriptors."""
        patch_dict = PatchLibrary.create(PatchGeometry.L_SHAPE)

        # Convert to Grid objects
        grids = {}
        for name, patch in patch_dict.items():
            grids[name] = Grid(
                ndim=patch.ndim,
                lower_bound=patch.lower_bound,
                upper_bound=patch.upper_bound,
                num_gridpoints=patch.num_gridpoints,
            )

        # Create composite
        composite = MultiPatchGridComposite(grids)
        assert composite.ndim == 2
        assert len(composite.patches) == 2


class TestDomainProperties:
    """Tests for multi-patch domain properties."""

    def test_memory_efficiency_l_shape(self):
        """Verify L-shape uses less memory than bounding box."""
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

        # Bounding box would be 2x2 = 20x20 (with 10 points per unit)
        bbox_cells = 20 * 20

        # L-shape should use significantly less (roughly half)
        assert total_l_cells < bbox_cells * 0.65

    def test_domain_bounds_consistency(self):
        """Test that patch bounds are consistent."""
        patches = PatchLibrary.create(PatchGeometry.L_SHAPE)

        # Create grids
        grids = {}
        for name, patch in patches.items():
            grids[name] = Grid(
                ndim=patch.ndim,
                lower_bound=patch.lower_bound,
                upper_bound=patch.upper_bound,
                num_gridpoints=patch.num_gridpoints,
            )

        composite = MultiPatchGridComposite(grids)

        # Check that patches can be retrieved
        for patch_id in composite.patch_ids:
            grid = composite.get_patch(patch_id)
            assert grid is not None
            assert grid.ndim == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
