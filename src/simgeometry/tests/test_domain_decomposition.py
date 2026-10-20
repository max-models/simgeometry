"""Tests for the process grid in ``simgeometry.domain_decomposition``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from maybempi import MPI

from simgeometry.domain_decomposition import (
    DomainDecomposition,
    PositionDecomposition,
)

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


# --- Owner lookup ----------------------------------------------------------


def _decomposition(
    proc_sizes: list[int], periodic: bool, rank: int = 0
) -> DomainDecomposition:
    """Return a decomposition with a chosen process grid, as seen from ``rank``."""
    ndim = len(proc_sizes)
    ddcomp = DomainDecomposition(comm=None, ndim=ndim, periodic=[periodic] * ndim)
    ddcomp._proc_sizes = list(proc_sizes)
    ddcomp._mpi_size = math.prod(proc_sizes)
    ddcomp._mpi_rank = rank
    return ddcomp


def _reference_owners(
    ddcomp: DomainDecomposition, positions: np.ndarray, lower, upper, nodes
) -> np.ndarray:
    """Brute force: test every rank's box, the outermost boxes reaching to infinity."""
    owners = np.full(positions.shape[1], -1)
    for rank in range(math.prod(ddcomp.proc_sizes)):
        coord = ddcomp.get_proc_coord(rank)
        lo, hi = ddcomp.subdomain_bounds(rank, lower, upper, nodes)
        inside = np.ones(positions.shape[1], dtype=bool)
        for dim, num_procs in enumerate(ddcomp.proc_sizes):
            if coord[dim] > 0:
                inside &= positions[dim] >= lo[dim]
            if coord[dim] < num_procs - 1:
                inside &= positions[dim] < hi[dim]
        assert not np.any(owners[inside] >= 0), "boxes overlap"
        owners[inside] = rank
    return owners


CASES = [
    ([4], (1.0,), (3.0,), (10,)),
    ([3], (-1.0,), (2.0,), (11,)),
    ([2, 3], (0.0, 0.0), (1.0, 2.0), (9, 8)),
    ([3, 1, 2], (0.0, -1.0, 0.0), (1.0, 1.0, 3.0), (10, 5, 7)),
    ([2, 2, 2], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (5, 6, 7)),
    ([1, 1], (0.0, 0.0), (1.0, 1.0), (5, 5)),
]


def _sample_positions(ddcomp, lower, upper, nodes) -> np.ndarray:
    """Random positions beyond the box plus every edge combination and its neighbours."""
    rng = np.random.default_rng(3)
    ndim = len(lower)
    width = np.array(upper) - np.array(lower)
    random = rng.uniform(
        np.array(lower)[:, None] - 0.3 * width[:, None],
        np.array(upper)[:, None] + 0.3 * width[:, None],
        (ndim, 300),
    )
    per_axis = []
    for dim in range(ndim):
        edges = ddcomp.subdomain_edges(dim, lower[dim], upper[dim], nodes[dim])
        values = []
        for edge in edges:
            values += [edge, np.nextafter(edge, -np.inf), np.nextafter(edge, np.inf)]
        values += [lower[dim] - 1.0, upper[dim] + 1.0]
        per_axis.append(values)
    grid = np.stack(
        np.meshgrid(*[np.array(v) for v in per_axis], indexing="ij"), axis=0
    ).reshape(ndim, -1)
    if grid.shape[1] > 3000:
        grid = grid[:, rng.choice(grid.shape[1], 3000, replace=False)]
    return np.concatenate([random, grid], axis=1)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(("proc_sizes", "lower", "upper", "nodes"), CASES)
def test_owner_ranks_match_the_brute_force_reference(
    proc_sizes, lower, upper, nodes, periodic
) -> None:
    """Vectorised owners, the owned mask and the scalar lookup agree with a reference."""
    ddcomp = _decomposition(proc_sizes, periodic)
    positions = _sample_positions(ddcomp, lower, upper, nodes)
    expected = _reference_owners(ddcomp, positions, lower, upper, nodes)

    owners = ddcomp.owner_ranks(positions, lower, upper, nodes)
    assert owners.dtype == np.int64
    assert owners.tolist() == expected.tolist()

    # The (n, ndim) layout gives the same answer.
    particle_major = ddcomp.owner_ranks(
        positions.T.copy(), lower, upper, nodes, layout="particle_major"
    )
    assert particle_major.tolist() == expected.tolist()

    # The scalar rule is the vector rule.
    for point, owner in list(zip(positions.T, expected))[::7]:
        assert ddcomp.owner_rank_for_position(point, lower, upper, nodes) == owner

    # Each rank's mask is exactly "owner == me".
    for rank in range(math.prod(proc_sizes)):
        mine = _decomposition(proc_sizes, periodic, rank=rank)
        mask = mine.owned_mask(positions, lower, upper, nodes)
        assert mask.tolist() == (expected == rank).tolist()
        assert mask.tolist() == (owners == rank).tolist()


def test_out_of_range_positions_go_to_the_nearest_block() -> None:
    """Below the lower bound is the first block, at or above the upper bound the last."""
    ddcomp = _decomposition([2, 2], periodic=False)
    args = ((0.0, 0.0), (1.0, 1.0), (5, 5))
    points = np.array([[-5.0, 1.0, 5.0, -5.0], [-5.0, 1.0, 0.2, 0.9]])
    # Row-major ranks over (2, 2): (0,0)=0 (0,1)=1 (1,0)=2 (1,1)=3.
    assert ddcomp.owner_ranks(points, *args).tolist() == [0, 3, 2, 1]
    assert ddcomp.owner_rank_for_position((-1.0, -1.0), *args) == 0
    assert ddcomp.owner_rank_for_position((1.0, 1.0), *args) == 3
    assert ddcomp.owner_rank_for_position((2.0, 0.1), *args) == 2
    nan = np.array([[np.nan], [np.nan]])
    assert ddcomp.owner_ranks(nan, *args).tolist() == [3]
    assert _decomposition([2, 2], False, rank=3).owned_mask(nan, *args).tolist() == [
        True
    ]


def test_layout_is_explicit_when_n_equals_ndim() -> None:
    """A (2, 2) array is read by the layout argument, never guessed."""
    ddcomp = _decomposition([2, 1], periodic=False)
    args = ((0.0, 0.0), (1.0, 1.0), (5, 5))
    # component-major: points (0.1, 0.2) and (0.9, 0.8)
    positions = np.array([[0.1, 0.9], [0.2, 0.8]])
    assert ddcomp.owner_ranks(positions, *args).tolist() == [0, 1]
    # the same array as particle-major: points (0.1, 0.9) and (0.2, 0.8)
    assert ddcomp.owner_ranks(positions, *args, layout="particle_major").tolist() == [
        0,
        0,
    ]


def test_owner_ranks_validate_their_input() -> None:
    """Wrong shapes and layouts raise instead of being reinterpreted."""
    ddcomp = _decomposition([2, 1], periodic=False)
    args = ((0.0, 0.0), (1.0, 1.0), (5, 5))
    with pytest.raises(ValueError, match="components"):
        ddcomp.owner_ranks(np.zeros((3, 4)), *args)
    with pytest.raises(ValueError, match="components"):
        ddcomp.owner_ranks(np.zeros((4, 3)), *args, layout="particle_major")
    with pytest.raises(ValueError, match="2D"):
        ddcomp.owner_ranks(np.zeros(2), *args)
    with pytest.raises(ValueError, match="layout"):
        ddcomp.owner_ranks(np.zeros((2, 2)), *args, layout="rows")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Bounds"):
        ddcomp.owner_ranks(np.zeros((2, 2)), (0.0,), (1.0,), (5,))


def test_replicated_decomposition_owns_everything_on_rank_zero() -> None:
    """Without a decomposed axis every position is owned by every rank, listed as rank 0."""
    ddcomp = DomainDecomposition(comm=None, ndim=2)
    ddcomp._proc_sizes = [1, 1]
    ddcomp._replicated = True
    ddcomp._mpi_rank = 2
    ddcomp._mpi_size = 4
    args = ((0.0, 0.0), (1.0, 1.0), (5, 5))
    points = np.array([[-1.0, 0.5, 2.0], [0.3, 0.3, 0.3]])
    assert ddcomp.owner_ranks(points, *args).tolist() == [0, 0, 0]
    assert ddcomp.owned_mask(points, *args).tolist() == [True, True, True]


def test_position_decomposition_binds_the_grid() -> None:
    """The bound object satisfies the protocol and agrees with the methods."""
    ddcomp = _decomposition([2, 2], periodic=False, rank=1)
    args = ((0.0, 0.0), (1.0, 1.0), (5, 5))
    bound = ddcomp.position_decomposition(*args)
    assert isinstance(bound, PositionDecomposition)
    assert (bound.comm, bound.rank, bound.size) == (None, 1, 4)
    points = np.random.default_rng(0).uniform(-0.2, 1.2, (2, 50))
    assert bound.owners(points).tolist() == ddcomp.owner_ranks(points, *args).tolist()
    assert (
        bound.owned_mask(points).tolist() == ddcomp.owned_mask(points, *args).tolist()
    )
    with pytest.raises(ValueError, match="Bounds"):
        ddcomp.position_decomposition((0.0,), (1.0,), (5,))


def test_owner_ranks_on_cupy() -> None:
    """With CuPy installed the lookup runs on the device and agrees with NumPy."""
    cp = pytest.importorskip("cupy")
    ddcomp = _decomposition([2, 3], periodic=False, rank=4)
    args = ((0.0, 0.0), (1.0, 2.0), (9, 8))
    points = np.random.default_rng(1).uniform(-0.3, 2.3, (2, 200))
    owners = ddcomp.owner_ranks(cp.asarray(points), *args)
    assert isinstance(owners, cp.ndarray)
    assert owners.get().tolist() == ddcomp.owner_ranks(points, *args).tolist()
    mask = ddcomp.owned_mask(cp.asarray(points), *args)
    assert mask.get().tolist() == ddcomp.owned_mask(points, *args).tolist()


@pytest.mark.skipif(size != 4, reason="this test must run on exactly 4 MPI ranks")
@pytest.mark.parametrize("periodic", [False, True])
def test_owner_ranks_mpi4(periodic: bool) -> None:
    """On 4 real ranks every position has one owner, which agrees with the mask."""
    comm = MPI.COMM_WORLD
    ddcomp = DomainDecomposition(
        comm=comm, decompose=[True, True], periodic=(periodic, periodic)
    )
    assert ddcomp.proc_sizes == [2, 2]
    args = ((0.0, 0.0), (1.0, 2.0), (9, 8))
    points = np.random.default_rng(8).uniform(-0.3, 2.3, (2, 400))
    bound = ddcomp.position_decomposition(*args)
    assert bound.rank == comm.Get_rank() and bound.size == 4 and bound.comm is comm

    owners = bound.owners(points)
    mine = bound.owned_mask(points)
    assert mine.tolist() == (owners == comm.Get_rank()).tolist()
    # Every rank, by the same data, sees every position owned by exactly one rank.
    counts = np.zeros(points.shape[1], dtype=np.int64)
    comm.Allreduce(mine.astype(np.int64), counts, op=MPI.SUM)
    assert np.all(counts == 1)
    # The owner lies inside the rank's own subdomain bounds for in-box positions.
    lo, hi = ddcomp.subdomain_bounds(comm.Get_rank(), *args)
    inside = np.all(
        (points >= np.array(lo)[:, None]) & (points < np.array(hi)[:, None]), axis=0
    )
    in_box = np.all((points >= 0) & (points < np.array([1.0, 2.0])[:, None]), axis=0)
    assert np.all(mine[inside & in_box])
