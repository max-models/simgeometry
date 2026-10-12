"""MPI tests for Grid decomposition."""

from __future__ import annotations

import pytest
from maybempi import MPI

from simgeometry.grid import Grid

SIZE = MPI.COMM_WORLD.Get_size()


@pytest.mark.skipif(SIZE != 4, reason="this test must run on exactly 4 MPI ranks")
def test_grid_uses_process_coordinates_for_multidimensional_splits() -> None:
    """Each rank should index per-axis splits with its process coordinates."""
    comm = MPI.COMM_WORLD
    grid = Grid(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(0.06, 0.02),
        num_gridpoints=(18, 12),
        comm=comm,
        domain_decomposition=(True, True),
    )

    assert grid.num_gridpoints_local == (9, 6)
    assert grid.proc_index_bounds == grid.domain_decomposition_class.get_index_bounds(
        grid.num_gridpoints,
    )


@pytest.mark.parametrize("periodic", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("decompose", [(True, True), (False, False)])
def test_grid_neighbours_respect_periodicity_and_replication(
    periodic: tuple[bool, bool],
    decompose: tuple[bool, bool],
) -> None:
    """Grid neighbours wrap on periodic axes and match the decomposition's."""
    comm = MPI.COMM_WORLD
    grid = Grid(
        ndim=2,
        lower_bound=(0.0, 0.0),
        upper_bound=(1.0, 1.0),
        num_gridpoints=(12, 8),
        comm=comm,
        domain_decomposition=decompose,
        periodic=periodic,
    )
    decomposition = grid.domain_decomposition_class

    assert grid.neighbor_ranks == list(decomposition.neighbour_ranks)
    for axis, (left, right) in enumerate(grid.neighbor_ranks):
        if periodic[axis]:
            assert MPI.PROC_NULL not in (left, right)
        elif decomposition.proc_sizes[axis] == 1:
            assert (left, right) == (MPI.PROC_NULL, MPI.PROC_NULL)
    for left, right in grid.neighbor_ranks:
        assert all(r == MPI.PROC_NULL or 0 <= r < SIZE for r in (left, right))
