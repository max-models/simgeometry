"""Tests for the process grid in ``simgeometry.domain_decomposition``."""

from __future__ import annotations

import pytest
from maybempi import MPI

from simgeometry.domain_decomposition import DomainDecomposition

size = MPI.COMM_WORLD.Get_size()


@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
def test_domain_decomposition_mpi6() -> None:
    """Six ranks on three decomposed axes: 2 * 1 * 3 processes."""
    ddcomp = DomainDecomposition(comm=MPI.COMM_WORLD, decompose=[True, True, True])
    assert ddcomp.proc_sizes == [2, 1, 3]


@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
@pytest.mark.parametrize("periodic", [True, False])
def test_domain_decomposition_neighbours_1d(periodic: bool) -> None:
    """One-dimensional neighbours wrap on a periodic axis and are null at walls."""
    comm = MPI.COMM_WORLD
    ddcomp = DomainDecomposition(comm=comm, decompose=[True], periodic=[periodic])
    rank = ddcomp.mpi_rank
    left, right = ddcomp.neighbour_ranks[0]

    if periodic:
        assert left == (rank - 1) % size
        assert right == (rank + 1) % size
    else:
        assert left == (MPI.PROC_NULL if rank == 0 else rank - 1)
        assert right == (MPI.PROC_NULL if rank == size - 1 else rank + 1)


@pytest.mark.parametrize("periodic", [(True, False), (False, False)])
def test_undecomposed_layout_is_replicated_on_every_rank(
    periodic: tuple[bool, bool],
) -> None:
    """Without decomposed axes every rank sits at process coordinate 0."""
    comm = MPI.COMM_WORLD
    ddcomp = DomainDecomposition(comm=comm, decompose=[False, False], periodic=periodic)

    assert ddcomp.replicated == (size > 1)
    assert ddcomp.proc_sizes == [1, 1]
    assert list(ddcomp.proc_coord) == [0, 0]
    assert all(list(ddcomp.get_proc_coord(rank)) == [0, 0] for rank in range(size))
    assert ddcomp.get_index_bounds((5, 3)) == [(0, 5), (0, 3)]
    rank = comm.Get_rank()
    for is_periodic, neighbours in zip(periodic, ddcomp.neighbour_ranks):
        if is_periodic:
            assert neighbours == (rank, rank)
        else:
            assert neighbours == (MPI.PROC_NULL, MPI.PROC_NULL)


def test_layout_without_communicator() -> None:
    """``comm=None`` is a single-rank layout with periodic self-neighbours."""
    ddcomp = DomainDecomposition(
        comm=None, decompose=[True, False], periodic=(True, False)
    )
    assert ddcomp.get_index_bounds((7, 5)) == [(0, 7), (0, 5)]
    assert ddcomp.neighbour_ranks == [(0, 0), (MPI.PROC_NULL, MPI.PROC_NULL)]


@pytest.mark.parametrize("decompose", [[True, True], [True, False], [False, False]])
@pytest.mark.parametrize("periodic", [(True, False), (False, True)])
def test_layout_agrees_with_the_process_grid(
    decompose: list[bool],
    periodic: tuple[bool, bool],
) -> None:
    """The array layout puts the blocks and neighbours where the process grid does."""
    ddcomp = DomainDecomposition(
        comm=MPI.COMM_WORLD, decompose=decompose, periodic=periodic
    )
    layout = ddcomp.layout((12, 8, 3), halo=(1, 1, 0))

    assert list(layout.process_grid[:2]) == ddcomp.proc_sizes
    assert layout.process_grid[2] == 1
    assert list(layout.index_bounds[:2]) == ddcomp.get_index_bounds((12, 8))
    assert layout.index_bounds[2] == (0, 3)
    assert list(layout.neighbours[:2]) == ddcomp.neighbour_ranks
    assert layout.periodic == (*periodic, False)
    assert tuple(layout.process_coord[:2]) == tuple(ddcomp.proc_coord)


def test_layout_has_a_whole_component_axis() -> None:
    """Axes beyond the decomposition's are not split and need ``ndim`` extents."""
    ddcomp = DomainDecomposition(comm=None, decompose=[True, True])
    assert ddcomp.layout((6, 4, 2)).local_shape == (6, 4, 2)
    with pytest.raises(ValueError):
        ddcomp.layout((6,))


@pytest.mark.parametrize("periodic", [False, True])
def test_subdomain_edges_sit_on_the_node_cuts(periodic: bool) -> None:
    """Markers and fields share one split: the edges are the cut nodes' positions."""
    ddcomp = DomainDecomposition(comm=None, ndim=1, periodic=[periodic])
    ddcomp._proc_sizes = [3]
    num_nodes = 10
    cuts = ddcomp.node_cuts((num_nodes,))[0]
    assert cuts == (0, 4, 7, 10)
    spacing = 2.0 / (num_nodes if periodic else num_nodes - 1)
    edges = ddcomp.subdomain_edges(0, 1.0, 3.0, num_nodes)
    assert edges[0] == 1.0 and edges[-1] == 3.0
    assert edges[1:-1] == pytest.approx([1.0 + 4 * spacing, 1.0 + 7 * spacing])
    assert [ddcomp.get_index_bounds((num_nodes,), rank)[0] for rank in range(3)] == [
        (0, 4),
        (4, 7),
        (7, 10),
    ]
