"""Grid helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from maybempi import MPI
from numpy import ndarray

from simgeometry._normalize import ensure_dim_tuple
from simgeometry._types import Array, ConfigValue, Printable
from simgeometry.boundingbox import BoundingBox


class Grid(BoundingBox):
    """Represent grid coordinates for a bounded simulation domain."""

    def __init__(
        self,
        ndim: int,
        lower_bound: tuple,
        upper_bound: tuple,
        num_gridpoints: tuple,
        num_ghost_points: int = 0,
        comm: MPI.Comm | None = None,
        domain_decomposition: ConfigValue = None,
        periodic: bool | tuple[bool, ...] | None = None,
    ) -> None:
        """
        Initialize a grid.

        Parameters:
        ndim (int): Number of dimensions (e.g., 1, 2, or 3).
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
        num_gridpoints (list of int or int in case of ndim==1): Number of grid points in each dimension.
                                      For example, [num_points_x, num_points_y, num_points_z] for 3D.
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            comm=comm,
            domain_decomposition=domain_decomposition,
            periodic=periodic,
        )

        num_gridpoints = ensure_dim_tuple(
            num_gridpoints,
            ndim=ndim,
        )

        self.num_gridpoints = num_gridpoints  # xp.array(num_gridpoints, dtype=int)
        self._num_ghost_points = num_ghost_points
        self.num_gridpoints_ndarray = np.array(num_gridpoints, dtype=int)
        # On a bounded axis the first and last node sit on the boundary; on a
        # periodic axis the last node is a neighbour of the first, not a copy
        # of it, so there are ``n`` cells over the length rather than ``n-1``.
        # This is the same convention as ``FieldConfig.node_spacing`` and
        # the field solvers, so markers and the solver agree on where the
        # nodes are.
        periodic_axes = self.periodic or (False,) * ndim
        self.grid_spacing = np.array(
            [
                (self.upper_bound[dir] - self.lower_bound[dir])
                / (
                    num_gridpoints[dir]
                    if periodic_axes[dir]
                    else num_gridpoints[dir] - 1
                )
                for dir in range(ndim)
            ],
            dtype=float,
        )

        self._proc_index_bounds = [
            (int(start), int(end))
            for start, end in self.domain_decomposition_class.get_index_bounds(
                self.num_gridpoints
            )
        ]
        self._num_gridpoints_local = [
            end - start for start, end in self._proc_index_bounds
        ]
        # Same neighbours as the decomposition: periodic wrap, PROC_NULL at walls.
        self._neighbor_ranks = list(self.domain_decomposition_class.neighbour_ranks)
        self._num_gridpoints_local = tuple(self._num_gridpoints_local)

        # Add attribute: number of local gridpoints including ghost points
        self._num_gridpoints_local_with_ghosts = tuple(
            n + 2 * self._num_ghost_points for n in self._num_gridpoints_local
        )

    def show(self, **kwargs):
        """Display grid lines and optional nodes, or an axis-aligned slice."""
        from simgeometry.collection import GeometryCollection

        geometry = GeometryCollection({}, bounds=(self.lower_bound, self.upper_bound))
        return geometry.show(grid=self, labels=False, **kwargs)

    def show_3d(self, **kwargs):
        """Display a three-dimensional grid and optional nodes."""
        from simgeometry.collection import GeometryCollection

        geometry = GeometryCollection({}, bounds=(self.lower_bound, self.upper_bound))
        return geometry.show_3d(grid=self, labels=False, **kwargs)

    @property
    def indices(self) -> list[tuple[int, int]]:
        """Return the indices."""
        indices = []
        if self.ndim == 1:
            return list(range(self.num_gridpoints[0]))
        elif self.ndim == 2:
            for i in range(self.num_gridpoints[0]):
                for j in range(self.num_gridpoints[1]):
                    indices.append((i, j))
            return indices
        elif self.ndim == 3:
            for i in range(self.num_gridpoints[0]):
                for j in range(self.num_gridpoints[1]):
                    for k in range(self.num_gridpoints[2]):
                        indices.append((i, j, k))
            return indices

    def node_points(self) -> np.ndarray:
        """Return the coordinates of every node as an ``(ndim, N)`` host array.

        The nodes are in the order of :attr:`indices` (C order of the grid);
        column ``n`` equals :meth:`node_coordinates` of the ``n``-th index.
        """
        lower = np.asarray(self.lower_bound, dtype=float)
        spacing = np.asarray(xp.to_numpy(self.grid_spacing), dtype=float)
        index = np.indices(tuple(int(n) for n in self.num_gridpoints)).reshape(
            self.ndim, -1
        )
        return lower[:, None] + index.astype(float) * spacing[:, None]

    def indices_from_mask(self, mask: Array) -> list[int] | list[tuple[int, ...]]:
        """Return the :attr:`indices` of the nodes where the grid-shaped ``mask`` is set.

        ``mask`` is a boolean array of shape :attr:`num_gridpoints` (or flat, in
        C order); the result keeps the order of :attr:`indices`.
        """
        mask = np.asarray(xp.to_numpy(mask), dtype=bool).reshape(
            tuple(int(n) for n in self.num_gridpoints)
        )
        if self.ndim == 1:
            return np.flatnonzero(mask).tolist()
        return [tuple(index) for index in np.argwhere(mask).tolist()]

    def node_coordinates(self, index: int | tuple[int, ...]) -> np.ndarray:
        """Return the physical coordinates of a grid node.

        ``index`` is a tuple of integer node indices (a plain integer in 1D).
        The result is ``lower_bound + index * grid_spacing`` as a float array of
        length ``ndim``, on the host.
        """
        index_array = np.atleast_1d(np.asarray(index, dtype=float))
        if index_array.shape != (self.ndim,):
            raise ValueError(
                f"Expected an index with {self.ndim} entries, got {index}.",
            )
        lower = np.asarray(self.lower_bound, dtype=float)
        spacing = np.asarray(xp.to_numpy(self.grid_spacing), dtype=float)
        return lower + index_array * spacing

    def get_cell_indices(self, *coords: Printable) -> tuple[int, ...]:
        """
        Get the cell indices for a given position, overridden to include cell size calculations.

        Parameters:
        coords (float): Coordinates for each dimension.

        Returns:
        tuple: Indices of the cell containing the position.
        """

        if len(coords) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(coords)}.")

        indices = []

        if self.contains(coords):
            for coord, min_bound, cell_size in zip(
                coords,
                self.lower_bound,
                self.grid_spacing,
            ):
                index = int((coord - min_bound) / cell_size)
                indices.append(index)

        return tuple(indices)

    def get_grid_points(self) -> list[ndarray]:
        """Return grid points for one axis."""
        grid_points_list = []
        periodic_axes = self.periodic or (False,) * self.ndim
        for dim in range(self.ndim):
            min_bound = self.lower_bound[dim]
            max_bound = self.upper_bound[dim]
            num_points = self.num_gridpoints[dim]
            if periodic_axes[dim]:
                # ``num_points`` cells over the length: the last node is one
                # spacing before the upper bound (see ``grid_spacing``).
                grid_points = min_bound + xp.arange(num_points) * self.grid_spacing[dim]
            else:
                grid_points = xp.linspace(min_bound, max_bound, num_points)
            grid_points_list.append(grid_points)
        return grid_points_list

    @property
    def meshgrid(self) -> tuple[Array, ...]:
        """Return the meshgrid."""
        return xp.meshgrid(*self.get_grid_points())

    @property
    def interpolator(self) -> ConfigValue:
        """Return the interpolator."""
        return self._interpolator

    @property
    def num_gridpoints_local(self) -> tuple[int, ...]:
        """Return the local grid point counts."""
        return self._num_gridpoints_local

    @property
    def num_gridpoints_local_with_ghosts(self) -> tuple[int, ...]:
        """Return local grid point counts including ghosts."""
        return self._num_gridpoints_local_with_ghosts

    @property
    def proc_index_bounds(self) -> list[tuple[int, int]]:
        """Return this rank global index bounds."""
        return self._proc_index_bounds

    @property
    def neighbor_ranks(self) -> list[tuple[int, int]]:
        """Return neighboring MPI ranks."""
        return self._neighbor_ranks
