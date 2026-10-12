"""Boundingbox helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

import cunumpy as xp
from maybempi import MPI

from ._bounds import Bounds
from ._normalize import ensure_dim_tuple
from ._types import Array, ConfigValue
from .domain_decomposition import DomainDecomposition


class BoundingBox:
    """Represent axis-aligned bounds for a simulation region."""

    # Applications may supply their own seeded stream. The default follows
    # the active cunumpy backend and works without any Picnic configuration.
    random = staticmethod(xp.random.random)

    def __init__(
        self,
        lower_bound: tuple,
        upper_bound: tuple,
        comm: MPI.Comm | None = None,
        domain_decomposition: ConfigValue = None,
        periodic: bool | None = None,
    ) -> None:
        """
        Initialize the simulation domain.

        Parameters:
        ndim (int): Number of dimensions (e.g., 1, 2, or 3).
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
                                For example, [(x_min, y_min, z_min), (y_max, y_max, z_max)] for 3D.
        """

        lower_bound = ensure_dim_tuple(lower_bound)
        upper_bound = ensure_dim_tuple(upper_bound)
        if len(lower_bound) != len(upper_bound):
            raise ValueError(
                "lower_bound and upper_bound must have the same length, "
                f"got {len(lower_bound)} and {len(upper_bound)}",
            )
        ndim = len(lower_bound)
        if ndim not in [1, 2, 3]:
            raise ValueError("Number of dimensions must be 1, 2, or 3")
        if domain_decomposition is not None:
            domain_decomposition = ensure_dim_tuple(
                domain_decomposition,
                ndim=ndim,
            )
        if periodic is not None:
            periodic = ensure_dim_tuple(periodic, ndim=ndim)
        for dir in range(ndim):
            if upper_bound[dir] < lower_bound[dir]:
                raise ValueError("upper bound must be >= lower bound in each dimension")
        # Set attributes
        self._bounds = Bounds(
            lower_bound=xp.array(lower_bound, dtype=float),
            upper_bound=xp.array(upper_bound, dtype=float),
        )

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._comm = comm

        if isinstance(comm, MPI.Comm):
            self._mpi_rank = comm.Get_rank()
            self._mpi_size = comm.Get_size()
        else:
            self._mpi_rank = 0
            self._mpi_size = 1

        self._periodic = periodic
        self._domain_decomposition = DomainDecomposition(
            comm=comm,
            decompose=domain_decomposition,
            periodic=periodic,
            ndim=self.ndim,
        )
        self._proc_sizes = self.domain_decomposition_class.proc_sizes

        # TODO: Use a tuple here instead
        self._sidelengths = tuple(
            upper_bound[dir] - lower_bound[dir] for dir in range(ndim)
        )

    def contains(self, coords: tuple) -> bool:
        """
        Check if the given coordinates are inside the domain.

        Parameters:
        coords (float): Coordinates for each dimension.

        Returns:
        bool: True if all coordinates are within the domain bounds, False otherwise.
        """

        assert len(coords) == self.ndim, (
            f"Expected {self.ndim} dimensions, got {len(coords)}."
        )

        return all(
            self.lower_bound[i] <= coord < self.upper_bound[i]
            for i, coord in enumerate(coords)
        )

    def get_relative_position(self, coords: Array) -> Array:
        """
        Determine the relative position of the given coordinates with respect to the domain bounds.

        Parameters:
            coords (float): Coordinates for each dimension.

        Returns:
            list: A list representing the relative position.
                [0, 0] if inside,
                [-1, 0] if to the left,
                [1, 0] if to the right,
                [0, -1] if below,
                [0, 1] if above,
                etc.
        """
        assert len(coords) == self.ndim, (
            f"Expected {self.ndim} dimensions, got {len(coords)}."
        )

        position = [0] * self.ndim

        for i, coord in enumerate(coords):
            if coord < self.lower_bound[i]:
                position[i] = -1
            elif coord >= self.upper_bound[i]:
                position[i] = 1
            else:
                position[i] = 0

        return position

    def draw_random_position(self) -> Array:
        """Draw one random position inside the bounding box."""
        pos = self.random(self.ndim)
        for i, (b0, b1) in enumerate(zip(self.lower_bound, self.upper_bound)):
            pos[i] = b0 + pos[i] * (b1 - b0)
        return pos

    def draw_random_positions(self, num_positions: int, out: Array = None) -> Array:
        """Draw `num_positions` random positions inside the bounding box.

        The result has shape ``(ndim, num_positions)``; ``out`` must have it.
        """
        shape = (self.ndim, num_positions)
        if out is None:
            out = self.random(shape)
            ret = True
        else:
            assert out.shape == shape, (
                f"Expected out with shape {shape}, got {out.shape}"
            )
            out[:] = self.random(shape)
            ret = False

        for dir in range(self.ndim):
            out[dir] = out[dir] * self.bounds.sidelengths[dir] + self.lower_bound[dir]

        if ret:
            return out

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return self._bounds.ndim

    @property
    def bounds(self) -> Bounds:
        """Return the lower and upper bounds."""
        return self._bounds

    @property
    def lower_bound(self) -> tuple:
        """Return the lower coordinate bounds."""
        return self._lower_bound

    @property
    def upper_bound(self) -> tuple:
        """Return the upper coordinate bounds."""
        return self._upper_bound

    @property
    def comm(self) -> MPI.Comm | None:
        """Return the MPI communicator."""
        return self._comm

    @property
    def mpi_rank(self) -> int:
        """Return the MPI rank."""
        return self._mpi_rank

    @property
    def mpi_size(self) -> int:
        """Return the MPI size."""
        return self._mpi_size

    @property
    def domain_decomposition_class(self) -> ConfigValue:
        """Return the domain decomposition class."""
        return self._domain_decomposition

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Return periodicity flags for each axis."""
        return self._periodic

    @property
    def proc_sizes(self) -> list[int]:
        """Return process counts along each axis."""
        return self._proc_sizes

    @property
    def midpoint(self) -> tuple[float, ...]:
        """Return the center point of the bounding box."""
        return tuple(
            0.5 * (self.lower_bound[dir] + self.upper_bound[dir])
            for dir in range(self.ndim)
        )

    @property
    def sidelengths(self) -> tuple[float, ...]:
        """Return the side lengths."""
        return self._sidelengths
