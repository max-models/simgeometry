"""Tests for geometry."""

import cunumpy as xp
import pytest

from simgeometry.boundingbox import BoundingBox
from simgeometry.grid import Grid
from simgeometry.objects import Circle, Rectangle


def test_domain(verbose: bool = False) -> None:
    # Test domain
    """Verify basic domain construction."""
    domain = BoundingBox(lower_bound=(0, 0), upper_bound=(1, 1))
    xmin = 0.5
    ymid = 0.5
    expected_positions = {
        (xmin - 1, ymid - 1): [-1, -1],
        (xmin - 1, ymid): [-1, 0],
        (xmin - 1, ymid + 1): [-1, 1],
        (xmin, ymid - 1): [0, -1],
        (xmin, ymid): [0, 0],
        (xmin, ymid + 1): [0, 1],
        (xmin + 1, ymid - 1): [1, -1],
        (xmin + 1, ymid): [1, 0],
        (xmin + 1, ymid + 1): [1, 1],
    }

    for (x, y), expected in expected_positions.items():
        result = domain.get_relative_position(coords=[x, y])
        assert xp.array_equal(result, expected), f"Failed for position ({x}, {y})"

    domain = BoundingBox(lower_bound=(0, 0), upper_bound=(1, 1))
    xmin = 0.5
    ymid = 0.5
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            assert domain.get_relative_position([xmin + dx, ymid + dy]) == [dx, dy]


def test_objects(verbose: bool = False) -> None:
    # Test Rectangle
    """Verify object construction helpers."""
    square = Rectangle(midpoint=(0, 0), side_lengths=(1, 2), bc={})

    assert xp.array_equal(square.midpoint, [0, 0])
    assert xp.array_equal(square.side_lengths, [1, 2])

    # Test Circle
    circle = Circle(midpoint=[0, 0], radius=1, bc={})

    assert xp.array_equal(circle.midpoint, [0, 0])
    # For scalar values, use xp.isclose
    assert xp.isclose(circle.radius, 1)

    print("Circle:", circle)
    print("Square:", square)


def test_grid(verbose: bool = False) -> None:
    """Verify grid construction."""
    grid = Grid(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=(10, 10))
    assert grid.ndim == 2
    assert xp.allclose(grid.lower_bound, (0, 0))
    assert xp.allclose(grid.upper_bound, (1, 1))
    assert xp.allclose(grid.num_gridpoints, (10, 10))

    print(grid)


def test_1d_grid_accepts_scalar_arguments() -> None:
    """1D grid construction accepts scalar-style arguments."""
    domain = BoundingBox(lower_bound=0.0, upper_bound=1.0)
    grid = Grid(ndim=1, lower_bound=0.0, upper_bound=1.0, num_gridpoints=10)

    assert domain.lower_bound == (0.0,)
    assert domain.upper_bound == (1.0,)
    assert grid.num_gridpoints == (10,)


def test_3d_grid_indices_cover_z_extent() -> None:
    """3D index generation must iterate over the z-axis grid size."""
    grid = Grid(
        ndim=3,
        lower_bound=(0.0, 0.0, 0.0),
        upper_bound=(1.0, 1.0, 1.0),
        num_gridpoints=(2, 3, 4),
    )

    indices = grid.indices
    assert len(indices) == 2 * 3 * 4
    assert (1, 2, 3) in indices
    assert max(k for _, _, k in indices) == 3


if __name__ == "__main__":
    pytest.main()
