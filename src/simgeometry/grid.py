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
        periodic: bool | None = None,
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


class MultiPatchGridComposite(Grid):
    """
    Composite grid wrapping multiple non-overlapping patches.

    Inherits all Grid functionality and is backward compatible with code
    expecting a Grid instance. Single-patch mode (default) has zero overhead.

    Attributes:
        patches: Dict mapping patch IDs to child Grid objects
        patch_bounds: Dict mapping patch IDs to (lower_bound, upper_bound) tuples
    """

    def __init__(
        self,
        patches: dict[str, Grid],
        comm: MPI.Comm | None = None,
    ) -> None:
        """
        Initialize a composite multi-patch grid.

        Parameters:
            patches: Dict mapping patch IDs (str) to Grid objects.
                    All patches must have the same ndim.
            comm: MPI communicator (inherited from first patch if None)

        Raises:
            ValueError: If patches dict is empty or patches have inconsistent ndim
        """
        if not patches:
            raise ValueError("patches dict cannot be empty")

        # Verify consistency
        first_patch = next(iter(patches.values()))
        ndim = first_patch.ndim
        for pid, grid in patches.items():
            if grid.ndim != ndim:
                raise ValueError(f"Patch '{pid}' has ndim={grid.ndim}, expected {ndim}")

        # Store patches
        self.patches = patches
        self._patch_ids = tuple(sorted(patches.keys()))

        # For backward compatibility: initialize parent Grid with bounds
        # of the "first" patch (alphabetically). This ensures isinstance checks pass.
        self._primary_patch = patches[self._patch_ids[0]]

        if comm is None:
            comm = self._primary_patch.comm

        super().__init__(
            ndim=ndim,
            lower_bound=self._primary_patch.lower_bound,
            upper_bound=self._primary_patch.upper_bound,
            num_gridpoints=self._primary_patch.num_gridpoints,
            num_ghost_points=self._primary_patch._num_ghost_points,
            comm=comm,
            domain_decomposition=None,
            periodic=self._primary_patch.periodic,
        )

    @property
    def patch_ids(self) -> tuple[str, ...]:
        """Return sorted tuple of patch identifiers."""
        return self._patch_ids

    def find_patch(self, global_position: Array | tuple[float, ...]) -> str | None:
        """
        Find which patch contains the given global position.

        Parameters:
            global_position: Global coordinate(s)

        Returns:
            Patch ID (string) if position is in a patch, None otherwise.

        Example:
            patch_id = grid.find_patch((0.2, 0.3))  # Returns 'L_horizontal' or similar
        """
        if isinstance(global_position, (list, tuple)):
            global_position = xp.array(global_position)

        for patch_id, patch in self.patches.items():
            if patch.contains(tuple(global_position)):
                return patch_id

        return None

    def global_to_local(
        self,
        global_position: Array | tuple[float, ...],
        patch_id: str | None = None,
    ) -> tuple[str, Array]:
        """
        Convert global coordinates to local patch coordinates.

        Parameters:
            global_position: Position in global (composite domain) coordinates
            patch_id: If None, automatically find patch. If provided, assumes
                     position is in this patch.

        Returns:
            (patch_id, local_position) tuple where local_position is relative
            to the patch's lower_bound.

        Raises:
            ValueError: If position is not in any patch
            KeyError: If specified patch_id doesn't exist

        Example:
            patch_id, local_pos = grid.global_to_local((0.3, 0.2))
            # patch_id might be 'L_horizontal'
            # local_pos is coordinates relative to L_horizontal's lower_bound
        """
        if isinstance(global_position, (list, tuple)):
            global_position = np.array(global_position, dtype=float)

        # Determine patch if not specified
        if patch_id is None:
            patch_id = self.find_patch(global_position)
            if patch_id is None:
                raise ValueError(
                    f"Position {global_position} is not in any patch of this composite grid",
                )
        elif patch_id not in self.patches:
            raise KeyError(f"Patch ID '{patch_id}' not found in composite grid")

        # Get patch and compute local coordinates
        patch = self.patches[patch_id]
        local_position = np.asarray(global_position, dtype=float) - np.array(
            patch.lower_bound,
            dtype=float,
        )

        return patch_id, local_position

    def local_to_global(
        self,
        local_position: Array | tuple[float, ...],
        patch_id: str,
    ) -> Array:
        """
        Convert local patch coordinates to global coordinates.

        Parameters:
            local_position: Position in local patch coordinates
            patch_id: Patch identifier

        Returns:
            Position in global (composite domain) coordinates

        Raises:
            KeyError: If patch_id doesn't exist

        Example:
            global_pos = grid.local_to_global((0.1, 0.2), patch_id='L_horizontal')
            # Returns coordinates in the composite L-shaped domain
        """
        if patch_id not in self.patches:
            raise KeyError(f"Patch ID '{patch_id}' not found in composite grid")

        if isinstance(local_position, (list, tuple)):
            local_position = np.array(local_position, dtype=float)

        patch = self.patches[patch_id]
        global_position = np.asarray(local_position, dtype=float) + np.array(
            patch.lower_bound,
            dtype=float,
        )

        return global_position

    def get_patch(self, patch_id: str) -> Grid:
        """
        Retrieve a child Grid object for a specific patch.

        Parameters:
            patch_id: Patch identifier

        Returns:
            Grid object for the patch

        Raises:
            KeyError: If patch_id doesn't exist
        """
        if patch_id not in self.patches:
            raise KeyError(f"Patch ID '{patch_id}' not found in composite grid")
        return self.patches[patch_id]

    def get_patch_bounds(
        self,
        patch_id: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """
        Get bounds for a specific patch.

        Parameters:
            patch_id: Patch identifier

        Returns:
            (lower_bound, upper_bound) tuple

        Raises:
            KeyError: If patch_id doesn't exist
        """
        patch = self.get_patch(patch_id)
        return (patch.lower_bound, patch.upper_bound)

    def __repr__(self) -> str:
        """Return string representation."""
        patch_list = ", ".join(self._patch_ids)
        return f"MultiPatchGridComposite(ndim={self.ndim}, patches=[{patch_list}])"
