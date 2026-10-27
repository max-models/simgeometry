"""Process grid of the MPI ranks and the physical subdomain each one owns.

The process grid decides which rank owns which part of the domain. It is used for
particles (:meth:`DomainDecomposition.owner_ranks`) and, through
:meth:`DomainDecomposition.layout`, for the :class:`mpiarray.DistributedArray` of the
fields, so that markers and fields agree on who owns what.

:meth:`DomainDecomposition.preview` builds the decomposition of a parallel run
without a communicator, for inspection and planning; :meth:`DomainDecomposition.show_3d`
and :meth:`DomainDecomposition.show` draw its subdomains.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import cunumpy as xp
import mpiarray as mpa
import numpy as np
from maybempi import MPI

Layout = Literal["component_major", "particle_major"]
"""Axis order of a positions array: ``(ndim, n)`` or ``(n, ndim)``."""


@runtime_checkable
class PositionDecomposition(Protocol):
    """What a particle container needs from a decomposition of physical space.

    The bounds of the domain are already bound in, so a container only passes
    positions. ``positions`` have shape ``(ndim, n)`` (component-major). This is
    the structural interface of the object returned by
    :meth:`DomainDecomposition.position_decomposition`; other decompositions
    can implement it without importing simgeometry.
    """

    comm: Any
    rank: int
    size: int

    def owners(self, positions: Any) -> Any:
        """Return the rank owning each position, as an integer array of shape ``(n,)``."""
        ...

    def owned_mask(self, positions: Any) -> Any:
        """Return a boolean array: which positions this rank owns."""
        ...


class _BoundPositionDecomposition:
    """A :class:`DomainDecomposition` with the grid bounds bound in."""

    def __init__(
        self,
        decomposition: DomainDecomposition,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
        num_gridpoints: tuple[int, ...],
    ) -> None:
        decomposition._check_grid(lower_bound, upper_bound, num_gridpoints)
        self._decomposition = decomposition
        self._grid = (
            tuple(float(v) for v in lower_bound),
            tuple(float(v) for v in upper_bound),
            tuple(int(v) for v in num_gridpoints),
        )

    @property
    def comm(self) -> MPI.Comm | None:
        """Return the communicator of the decomposition."""
        return self._decomposition.comm

    @property
    def rank(self) -> int:
        """Return this rank."""
        return self._decomposition.mpi_rank

    @property
    def size(self) -> int:
        """Return the number of ranks."""
        return self._decomposition.mpi_size

    def owners(self, positions: Any, layout: Layout = "component_major") -> Any:
        """Return the owning rank of every position (see ``DomainDecomposition.owner_ranks``)."""
        return self._decomposition.owner_ranks(positions, *self._grid, layout=layout)

    def owned_mask(self, positions: Any, layout: Layout = "component_major") -> Any:
        """Return which positions this rank owns (see ``DomainDecomposition.owned_mask``)."""
        return self._decomposition.owned_mask(positions, *self._grid, layout=layout)


class DomainDecomposition:
    """Represent the MPI process grid of a decomposed domain."""

    def _calculate_neighbor_ranks(self) -> list[tuple[int, int]]:
        """
        Given a Cartesian decomposition of MPI ranks and your own rank,
        compute the neighbor ranks in each dimension.

        Returns
        -------
        neighbor_ranks : List[Tuple[int, int]]
            For each dim i, a tuple (left_rank, right_rank). If you're at the
            boundary on the left (resp. right), left_rank (resp. right_rank)
            is MPI.PROC_NULL.
        """
        proc_sizes = self.proc_sizes
        rank = self.mpi_rank

        ndim = len(proc_sizes)
        # compute strides so that flat_rank = sum(coords[i] * strides[i])
        strides = [math.prod(proc_sizes[i + 1 :]) for i in range(ndim)]

        # decode flat rank → multi‐dim coords
        coords = []
        rem = rank
        for i in range(ndim):
            stride = strides[i]
            c = rem // stride
            coords.append(c)
            rem %= stride

        def rank_from_coords(process_coords: list[int]) -> int:
            """Convert process coordinates to a flat MPI rank."""
            return sum(process_coords[j] * strides[j] for j in range(ndim))

        neighbors = []
        for i in range(ndim):
            # LEFT neighbor
            if coords[i] > 0:
                left_coords = coords.copy()
                left_coords[i] -= 1
                left_rank = rank_from_coords(left_coords)
            else:
                if self.periodic[i]:
                    left_coords = coords.copy()
                    left_coords[i] = proc_sizes[i] - 1
                    left_rank = rank_from_coords(left_coords)
                else:
                    left_rank = MPI.PROC_NULL

            # RIGHT neighbor
            if coords[i] < proc_sizes[i] - 1:
                right_coords = coords.copy()
                right_coords[i] += 1
                right_rank = rank_from_coords(right_coords)
            else:
                if self.periodic[i]:
                    right_coords = coords.copy()
                    right_coords[i] = 0
                    right_rank = rank_from_coords(right_coords)
                else:
                    right_rank = MPI.PROC_NULL

            neighbors.append((left_rank, right_rank))

        return neighbors

    def _calculate_proc_sizes(self) -> list[int]:
        """
        Calculate the optimal number of processes in each dimension based on the
        total number of processes and the domain decomposition flags.

        Args:
            size: Total number of MPI processes.
            domain_decomposition: List of booleans indicating whether to decompose
                                along each dimension (e.g., [True, True, False]
                                for decomposing in x and y but not z in 3D).

        Returns:
            proc_sizes: List of integers indicating the number of processes in each
                        dimension, matching the length of domain_decomposition.
                        For example: [2, 6, 1] for 12 processes in a 2x6 grid in 3D.

        Raises:
            ValueError: If the size cannot be evenly distributed among any set of requested dimensions.
        """

        size = self.mpi_size
        domain_decomposition = self.decompose

        ndim = self.ndim
        proc_sizes = [1] * ndim  # Initialize proc_sizes with 1 in each dimension

        # Priority Decomposition: Start with all requested dimensions
        active_decomposition = domain_decomposition[:]

        while True:
            # Calculate the target number of processes per active dimension
            decompose_dims = sum(active_decomposition)

            if decompose_dims == 0:
                # No decomposable dimensions left, fallback to 1D decomposition
                proc_sizes = [1] * ndim
                for i in range(ndim):
                    if domain_decomposition[i]:
                        proc_sizes[i] = size
                        break
                return proc_sizes

            # Reset proc_sizes and remaining_size for current attempt
            proc_sizes = [1] * ndim
            remaining_size = size

            for i in range(ndim):
                if active_decomposition[i]:
                    # Calculate the approximate target number of processes in this dimension
                    target_procs = round(remaining_size ** (1 / decompose_dims))

                    # Adjust target_procs to be a divisor of the remaining_size
                    while remaining_size % target_procs != 0 and target_procs > 1:
                        target_procs -= 1

                    # Assign the calculated target_procs to this dimension
                    proc_sizes[i] = target_procs
                    remaining_size //= target_procs
                    decompose_dims -= 1

            # Check if the decomposition works
            product_of_procs = math.prod(
                [proc_sizes[i] for i in range(ndim) if active_decomposition[i]],
            )

            if product_of_procs == size:
                return proc_sizes

            # If not, disable the last dimension in active_decomposition and try again
            for j in reversed(range(ndim)):
                if active_decomposition[j]:
                    active_decomposition[j] = False
                    break

    def _setup(
        self,
        comm: MPI.Comm | None,
        size: int,
        rank: int,
        decompose: list[bool] | None,
        periodic: tuple[bool, ...] | None,
        ndim: int | None,
        proc_sizes: list[int] | None = None,
    ) -> None:
        """Build the process grid of ``size`` ranks, as seen from ``rank``."""
        self._comm = comm
        self._mpi_size = size
        self._mpi_rank = rank

        assert ndim is not None or decompose is not None, (
            "DomainDecomposition must take either ndim or decompose!"
        )

        if ndim is None:
            ndim = len(decompose)
        self._ndim = ndim

        if decompose is None:
            self._decompose = [True] * self.ndim
        else:
            self._decompose = list(decompose)
        assert len(self.decompose) == self.ndim

        if periodic is None:
            periodic = [False] * self.ndim
        self._periodic = tuple(periodic)
        assert len(self.periodic) == self.ndim

        if proc_sizes is None:
            self._proc_sizes = self._calculate_proc_sizes()
        else:
            self._proc_sizes = list(proc_sizes)

        # With no decomposed axis the process grid has a single slot and every
        # rank holds the whole domain; all ranks then share process coordinate 0.
        self._replicated = self.mpi_size > 1 and math.prod(self.proc_sizes) == 1

        # Ranks are placed row-major on the process grid, as mpiarray does.
        if self.replicated:
            self._neighbour_ranks = [
                (
                    (self.mpi_rank, self.mpi_rank)
                    if is_periodic
                    else (MPI.PROC_NULL, MPI.PROC_NULL)
                )
                for is_periodic in self.periodic
            ]
        else:
            self._neighbour_ranks = self._calculate_neighbor_ranks()

    def __init__(
        self,
        comm: MPI.Comm | None,
        decompose: list[bool] | None = None,
        periodic: tuple[bool, ...] | None = None,
        ndim: int | None = None,
    ) -> None:
        """Initialize the process grid of ``comm`` for the decomposed axes.

        Without a communicator (``comm=None``) there is a single process. To
        inspect the split of a parallel run without one, use :meth:`preview`.
        """
        if comm is None:
            size, rank = 1, 0
        else:
            size, rank = comm.Get_size(), comm.Get_rank()
        self._setup(comm, size, rank, decompose, periodic, ndim)

    @classmethod
    def preview(
        cls,
        size: int | None = None,
        *,
        proc_sizes: Sequence[int] | None = None,
        decompose: Sequence[bool] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        rank: int = 0,
    ) -> DomainDecomposition:
        """Return the decomposition a ``size``-rank run would use, without a communicator.

        This is for inspection and planning, e.g. to draw the subdomains in
        the docs (:meth:`show_3d`, :meth:`show`) or to check the local array
        sizes of a run before launching it. Nothing communicates: the result
        has no communicator (``comm`` is ``None``) and :meth:`layout` raises.

        With ``size``, the processes per axis are chosen exactly as for a
        communicator of that size, so ``proc_sizes``, the index ranges and the
        subdomains are those of the real run. Alternatively, ``proc_sizes``
        fixes the processes per axis directly (the axes with more than one are
        then the decomposed ones, unless ``decompose`` says otherwise).

        The decomposition is seen from ``rank`` (``mpi_rank``,
        ``neighbour_ranks``, ``proc_coord``, ``owned_mask``). The read-only
        queries that take a rank, or return one per position, work for every
        rank: :meth:`get_index_bounds`, :meth:`subdomain_bounds`,
        :meth:`subdomain_boxes`, :meth:`get_proc_coord`, :meth:`owner_ranks`.

        Example:
            >>> preview = DomainDecomposition.preview(8, decompose=[True] * 3)
            >>> preview.proc_sizes
            [2, 2, 2]
            >>> preview.get_index_bounds((9, 9, 9), rank=7)
            [(5, 9), (5, 9), (5, 9)]
        """
        if proc_sizes is not None:
            proc_sizes = [int(p) for p in proc_sizes]
            if any(p < 1 for p in proc_sizes):
                raise ValueError(f"proc_sizes must be positive, got {proc_sizes}")
            if ndim is not None and ndim != len(proc_sizes):
                raise ValueError(f"proc_sizes {proc_sizes} do not have {ndim} axes")
            ndim = len(proc_sizes)
            if decompose is None:
                decompose = [p > 1 for p in proc_sizes]
            if any(p > 1 and not d for p, d in zip(proc_sizes, decompose)):
                raise ValueError(
                    f"proc_sizes {proc_sizes} split an axis that decompose "
                    f"{list(decompose)} does not"
                )
            total = math.prod(proc_sizes)
            if size is None:
                size = total
            elif total not in (1, size):
                raise ValueError(f"proc_sizes {proc_sizes} do not multiply to {size}")
        if size is None:
            raise ValueError("Pass size or proc_sizes")
        if size < 1:
            raise ValueError(f"size must be positive, got {size}")
        if not 0 <= rank < size:
            raise ValueError(f"rank {rank} is not in [0, {size})")
        decomposition = cls.__new__(cls)
        decomposition._setup(
            None,
            int(size),
            int(rank),
            None if decompose is None else list(decompose),
            None if periodic is None else tuple(periodic),
            ndim,
            proc_sizes,
        )
        return decomposition

    def layout(
        self,
        shape: tuple[int, ...],
        halo: int | Sequence[int] = 0,
        periodic: Sequence[bool] | None = None,
    ) -> mpa.Layout:
        """Return the :class:`mpiarray.Layout` of an array of ``shape`` on this process grid.

        Axes after ``ndim`` (for example the component axis of a field) are not
        split. ``periodic`` defaults to this decomposition's flags, padded with
        ``False`` for the extra axes.
        """
        extra = len(shape) - self.ndim
        if extra < 0:
            raise ValueError(f"shape {shape} has fewer than {self.ndim} axes")
        if periodic is None:
            periodic = (*self.periodic, *([False] * extra))
        if self.is_preview:
            raise ValueError(
                "A preview has no communicator, so it has no array layout; use "
                "get_index_bounds() or node_cuts() for the index ranges"
            )
        comm = MPI.COMM_SELF if self.comm is None else self.comm
        if self.replicated:
            # No axis is split: every rank holds the whole array.
            return mpa.Layout(
                shape, comm=comm, split=None, halo=halo, periodic=periodic
            )
        return mpa.Layout(
            shape,
            comm=comm,
            halo=halo,
            periodic=periodic,
            process_grid=(*self.proc_sizes, *([1] * extra)),
        )

    def node_cuts(self, shape: Sequence[int]) -> tuple[tuple[int, ...] | None, ...]:
        """Return where the blocks of a grid of ``shape`` nodes start and end, per axis.

        ``(0, c1, ..., n)`` along a split axis (process coordinate ``c`` owns
        nodes ``cuts[c] <= i < cuts[c + 1]``), ``None`` along the others. This
        is the one split of the grid: the field arrays (:meth:`layout`), the
        solver's matrix rows and the markers (:meth:`subdomain_edges`) all
        follow it.
        """
        if len(shape) != self.ndim:
            raise ValueError(f"Expected a shape with {self.ndim} axes, got {shape}")
        cuts: list[tuple[int, ...] | None] = []
        for length, num_procs in zip(shape, self.proc_sizes, strict=True):
            if num_procs == 1 or self.replicated:
                cuts.append(None)
                continue
            starts = [
                mpa.chunk_bounds(int(length), num_procs, c)[0] for c in range(num_procs)
            ]
            cuts.append((*starts, int(length)))
        return tuple(cuts)

    def get_proc_coord(self, rank: int) -> tuple[int, ...]:
        """Return process coordinates for one rank."""
        if self.replicated:
            rank = 0
        coords = []
        for size in reversed(self.proc_sizes):
            coords.append(rank % size)
            rank //= size
        return tuple(reversed(coords))

    def rank_from_proc_coord(self, proc_coord: tuple[int, ...]) -> int:
        """Return the flat MPI rank for process coordinates."""
        if len(proc_coord) != self.ndim:
            raise ValueError(f"Expected {self.ndim} process coordinates")
        strides = [math.prod(self.proc_sizes[i + 1 :]) for i in range(self.ndim)]
        return int(sum(proc_coord[i] * strides[i] for i in range(self.ndim)))

    def get_index_bounds(
        self,
        shape: tuple[int, ...],
        rank: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return global index bounds for one rank over a decomposed shape."""
        if rank is None:
            rank = self.mpi_rank
        if len(shape) != self.ndim:
            raise ValueError(f"Expected shape with {self.ndim} dimensions")
        coord = self.get_proc_coord(rank)
        cuts = self.node_cuts(shape)
        return [
            (0, int(length)) if axis_cuts is None else axis_cuts[c : c + 2]
            for length, axis_cuts, c in zip(shape, cuts, coord, strict=True)
        ]

    def subdomain_edges(
        self, dim: int, lower: float, upper: float, num_nodes: int
    ) -> list[float]:
        """Return the ``proc_sizes[dim] + 1`` edges of the subdomains along one axis.

        The edges sit on the nodes where :meth:`node_cuts` splits the
        ``num_nodes`` nodes of the axis (``n - 1`` cells over ``[lower,
        upper]``, ``n`` on a periodic axis). Process coordinate ``c`` owns
        ``[edges[c], edges[c + 1])``, the cells whose lower node it owns; the
        last one also owns ``upper`` and everything beyond, the first
        everything below ``lower`` (see :meth:`owner_ranks`).
        """
        num_procs = self.proc_sizes[dim]
        if num_procs == 1 or self.replicated:
            return [lower, upper]
        cells = num_nodes if self.periodic[dim] else num_nodes - 1
        spacing = (upper - lower) / cells
        starts = [
            mpa.chunk_bounds(int(num_nodes), num_procs, c)[0] for c in range(num_procs)
        ]
        return [lower + start * spacing for start in starts] + [upper]

    def subdomain_bounds(
        self,
        rank: int,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
        num_gridpoints: tuple[int, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the physical ``(lower, upper)`` corners of one rank's subdomain."""
        proc_coord = self.get_proc_coord(rank)
        lower = []
        upper = []
        for dim in range(self.ndim):
            edges = self.subdomain_edges(
                dim,
                float(lower_bound[dim]),
                float(upper_bound[dim]),
                int(num_gridpoints[dim]),
            )
            lower.append(edges[proc_coord[dim]])
            upper.append(edges[proc_coord[dim] + 1])
        return tuple(lower), tuple(upper)

    def _check_grid(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
    ) -> None:
        """Raise if the grid description does not have ``ndim`` entries."""
        if len(lower_bound) != self.ndim or len(upper_bound) != self.ndim:
            raise ValueError("Bounds must match decomposition dimensions")
        if len(num_gridpoints) != self.ndim:
            raise ValueError("num_gridpoints must match decomposition dimensions")

    def subdomain_boxes(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
    ) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        """Return the physical ``(lower, upper)`` corners of every rank's subdomain.

        Entry ``r`` is :meth:`subdomain_bounds` of rank ``r``, for all
        ``mpi_size`` ranks. The boxes tile the domain without gaps or overlaps
        (when no axis is decomposed, every rank holds the whole domain).
        """
        self._check_grid(lower_bound, upper_bound, num_gridpoints)
        grid = (tuple(lower_bound), tuple(upper_bound), tuple(num_gridpoints))
        return [self.subdomain_bounds(rank, *grid) for rank in range(self.mpi_size)]

    def show_3d(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
        *,
        shape: Any = None,
        opacity: float = 0.25,
        shape_opacity: float = 1.0,
        resolution: int = 48,
        legend: bool | None = None,
        plotter: Any = None,
        display: bool = True,
        save: str | None = None,
    ) -> Any:
        """Draw every rank's subdomain as a translucent box, coloured by rank.

        Needs the ``interactive`` extra (PyVista and plasma-plots). The boxes
        are drawn with ``plasma_plots.pyvista_plots.pyvista_surfaces``, one
        mesh per rank labelled ``"rank r"``, with their edges. ``shape`` (a 3D
        shape or collection) is drawn inside them. ``legend=None`` shows the
        rank legend for up to 16 ranks and colours by a ``rank`` scalar bar
        above that. Returns the ``pyvista.Plotter``; it is shown only with
        ``display=True``, and ``save`` writes a screenshot. Combine it with
        :meth:`preview` to see the split of a parallel run; no MPI operation is
        performed.
        """
        from simgeometry._decomposition_views import show_decomposition_3d

        return show_decomposition_3d(
            self,
            lower_bound,
            upper_bound,
            num_gridpoints,
            shape=shape,
            opacity=opacity,
            shape_opacity=shape_opacity,
            resolution=resolution,
            legend=legend,
            plotter=plotter,
            display=display,
            save=save,
        )

    def show(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
        *,
        shape: Any = None,
        ax: Any = None,
        opacity: float = 0.35,
        labels: bool = True,
        display: bool = True,
        save: str | None = None,
    ) -> Any:
        """Draw a 2D decomposition with Matplotlib: one rectangle per rank.

        Each subdomain is filled in its rank's colour and labelled with its
        rank (``labels``); ``shape`` (a 2D shape or collection) is outlined on
        top. Returns the Matplotlib axes. No MPI operation is performed.
        """
        from simgeometry._decomposition_views import show_decomposition_2d

        return show_decomposition_2d(
            self,
            lower_bound,
            upper_bound,
            num_gridpoints,
            shape=shape,
            ax=ax,
            opacity=opacity,
            labels=labels,
            display=display,
            save=save,
        )

    def _components(self, positions: Any, layout: Layout) -> Any:
        """Return ``positions`` as a float ``(ndim, n)`` array on the active backend."""
        if layout not in ("component_major", "particle_major"):
            raise ValueError(
                f"layout must be 'component_major' or 'particle_major', got {layout!r}"
            )
        positions = xp.asarray(positions, dtype=float)
        if positions.ndim != 2:
            raise ValueError(f"positions must be 2D, got shape {positions.shape}")
        if layout == "particle_major":
            positions = positions.T
        if positions.shape[0] != self.ndim:
            raise ValueError(
                f"positions have {positions.shape[0]} components for layout "
                f"{layout!r}, expected {self.ndim}"
            )
        return positions

    def owner_ranks(
        self,
        positions: Any,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
        layout: Layout = "component_major",
    ) -> Any:
        """Return the rank owning each position, vectorised.

        ``positions`` is a NumPy or CuPy array (whatever the active cunumpy
        backend is) of shape ``(ndim, n)`` for ``layout="component_major"``
        (the default, the layout of the marker arrays) or ``(n, ndim)`` for
        ``layout="particle_major"``. The layout is never guessed, so ``n ==
        ndim`` is not ambiguous. The result is an ``int64`` array of shape
        ``(n,)`` in the same memory as ``positions``; ranks are row-major over
        :attr:`proc_sizes`.

        Along each axis the subdomains are the blocks of
        :meth:`subdomain_edges`: block ``c`` owns ``[edges[c], edges[c + 1])``.
        **Out-of-range rule**: a position below the lower bound belongs to the
        first block and one at or above the upper edge (including exactly the
        upper bound) to the last block, that is, the block index is clipped into
        ``[0, proc_sizes - 1]`` per axis. A position outside the box is thus
        owned by the nearest subdomain, so it is never shipped across the whole
        domain and exactly one rank owns it; deciding what happens to it (kill,
        reflect, wrap on a periodic axis) is the particle boundary condition's
        job and must happen before the lookup. NaN positions go to the last
        block. Without a decomposed axis every position belongs to rank 0.
        """
        self._check_grid(lower_bound, upper_bound, num_gridpoints)
        positions = self._components(positions, layout)
        owners = xp.zeros(positions.shape[1], dtype=xp.int64)
        for dim, num_procs in enumerate(self.proc_sizes):
            if num_procs == 1 or self.replicated:
                continue
            edges = self.subdomain_edges(
                dim,
                float(lower_bound[dim]),
                float(upper_bound[dim]),
                int(num_gridpoints[dim]),
            )
            # The inner edges split the axis; searchsorted already yields
            # 0 below the first and num_procs - 1 at or above the last.
            coord = xp.searchsorted(
                xp.asarray(edges[1:-1], dtype=float), positions[dim], side="right"
            )
            owners += coord * math.prod(self.proc_sizes[dim + 1 :])
        return owners

    def owned_mask(
        self,
        positions: Any,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
        layout: Layout = "component_major",
    ) -> Any:
        """Return which positions this rank owns, by the rule of :meth:`owner_ranks`.

        Equal to ``owner_ranks(...) == mpi_rank`` (every rank owns everything
        when no axis is decomposed), but only two comparisons per decomposed
        axis against this rank's own edges, which is much cheaper than finding
        every owner when most positions stay put.
        """
        self._check_grid(lower_bound, upper_bound, num_gridpoints)
        positions = self._components(positions, layout)
        owned = xp.ones(positions.shape[1], dtype=bool)
        coords = self.proc_coord
        for dim, num_procs in enumerate(self.proc_sizes):
            if num_procs == 1 or self.replicated:
                continue
            edges = self.subdomain_edges(
                dim,
                float(lower_bound[dim]),
                float(upper_bound[dim]),
                int(num_gridpoints[dim]),
            )
            coord = coords[dim]
            column = positions[dim]
            last = num_procs - 1
            if coord > 0:
                # ``~(<)`` rather than ``>=`` keeps NaN on the last block.
                owned &= (
                    ~(column < edges[coord])
                    if coord == last
                    else (column >= edges[coord])
                )
            if coord < last:
                owned &= column < edges[coord + 1]
        return owned

    def owner_rank_for_position(
        self,
        position: Sequence[float] | np.ndarray,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
        num_gridpoints: tuple[int, ...],
    ) -> int:
        """Return the rank owning one physical position (see :meth:`owner_ranks`)."""
        self._check_grid(lower_bound, upper_bound, num_gridpoints)
        point = xp.asarray(position, dtype=float).reshape(-1, 1)
        owners = self.owner_ranks(point, lower_bound, upper_bound, num_gridpoints)
        return int(owners[0])

    def position_decomposition(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        num_gridpoints: Sequence[int],
    ) -> PositionDecomposition:
        """Return this decomposition with the grid bounds bound in.

        The result implements :class:`PositionDecomposition` (``comm``,
        ``rank``, ``size``, ``owners(positions)``, ``owned_mask(positions)``),
        which is what a particle container needs; it does not change when the
        bounds argument sequences are mutated afterwards.
        """
        return _BoundPositionDecomposition(
            self, tuple(lower_bound), tuple(upper_bound), tuple(num_gridpoints)
        )

    @property
    def comm(self) -> MPI.Comm | None:
        """Return the MPI communicator."""
        return self._comm

    @property
    def mpi_size(self) -> int:
        """Return the MPI size."""
        return self._mpi_size

    @property
    def mpi_rank(self) -> int:
        """Return the MPI rank."""
        return self._mpi_rank

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return self._ndim

    @property
    def decompose(self) -> list[bool]:
        """Return flags indicating decomposed axes."""
        return self._decompose

    @property
    def is_preview(self) -> bool:
        """Return whether this is a :meth:`preview` of several ranks, without a communicator."""
        return self.comm is None and self.mpi_size > 1

    @property
    def replicated(self) -> bool:
        """Return whether every rank holds the whole domain (no decomposed axis)."""
        return self._replicated

    @property
    def proc_sizes(self) -> list[int]:
        """Return process counts along each axis."""
        return self._proc_sizes

    @property
    def proc_coord(self) -> tuple[int, ...]:
        """Return this rank process coordinates."""
        return self.get_proc_coord(self.mpi_rank)

    @property
    def neighbour_ranks(self) -> list[tuple[int, int]]:
        """Return neighboring MPI ranks."""
        return self._neighbour_ranks

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Return periodicity flags for each axis."""
        return self._periodic


def calculate_proc_sizes(size: int, domain_decomposition: list[bool]) -> list[int]:
    """
    Calculate the optimal number of processes in each dimension based on the
    total number of processes and the domain decomposition flags.

    Args:
        size: Total number of MPI processes. (e.g., [True, True, False])

    Returns:
        proc_sizes: Number of processes in each dimension, (e.g., [2, 6, 1])

    Raises:
        ValueError: If the size cannot be evenly distributed among any set of requested dimensions.
    """
    ndim = len(domain_decomposition)
    proc_sizes = [1] * ndim  # Initialize proc_sizes with 1 in each dimension

    # Priority Decomposition: Start with all requested dimensions
    active_decomposition = domain_decomposition[:]

    while True:
        # Calculate the target number of processes per active dimension
        decompose_dims = sum(active_decomposition)

        if decompose_dims == 0:
            # No decomposable dimensions left, fallback to 1D decomposition
            proc_sizes = [1] * ndim
            for i in range(ndim):
                if domain_decomposition[i]:
                    proc_sizes[i] = size
                    break
            return proc_sizes

        # Reset proc_sizes and remaining_size for current attempt
        proc_sizes = [1] * ndim
        remaining_size = size

        for i in range(ndim):
            if active_decomposition[i]:
                # Calculate the approximate target number of processes in this dimension
                target_procs = round(remaining_size ** (1 / decompose_dims))

                # Adjust target_procs to be a divisor of the remaining_size
                while remaining_size % target_procs != 0 and target_procs > 1:
                    target_procs -= 1

                # Assign the calculated target_procs to this dimension
                proc_sizes[i] = target_procs
                remaining_size //= target_procs
                decompose_dims -= 1

        # Check if the decomposition works
        product_of_procs = math.prod(
            [proc_sizes[i] for i in range(ndim) if active_decomposition[i]],
        )

        if product_of_procs == size:
            return proc_sizes

        # If not, disable the last dimension in active_decomposition and try again
        for j in reversed(range(ndim)):
            if active_decomposition[j]:
                active_decomposition[j] = False
                break
