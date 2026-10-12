"""Simulation domain helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

from typing import Any

import cunumpy as xp
from maybempi import MPI

from simgeometry._particle_boundaries import apply_particle_boundaries
from simgeometry._types import Array, ConfigValue
from simgeometry.boundingbox import BoundingBox
from simgeometry.grid import Grid
from simgeometry.shapes import Shape

# The particle boundaries ``apply_bc_arrays`` knows.
_PARTICLE_BC_TYPES = ("kill", "reflect", "periodic", "reinject")


#: The ``apply_particle_boundaries`` kernel's code of each particle boundary type.
PARTICLE_BOUNDARY_CODES = {"kill": 0, "reflect": 1, "periodic": 2, "reinject": 3}


class SimulationDomain(BoundingBox):
    """Class representing the simulation domain with boundary conditions."""

    boundary_kernel = staticmethod(apply_particle_boundaries)

    def __init__(
        self,
        ndim: int,
        lower_bound: tuple,
        upper_bound: tuple,
        num_gridpoints: int | tuple[int, ...],
        objects: list[Shape] | None = None,
        bcs_field: Any | None = None,
        bcs_particles: Any | None = None,
        comm: MPI.Comm | None = None,
        domain_decomposition: ConfigValue = None,
    ) -> None:
        """
        Initialize the simulation domain with boundary conditions.

        Parameters:
        ndim (int): Number of dimensions (e.g., 1, 2, or 3).
        bounds (list of tuples): List of tuples specifying min and max for each dimension.
        bcs (dict): Dictionary specifying the boundary conditions for each dimension.
        """
        # A periodic field axis has ``n`` cells over the length (its last node
        # neighbours its first); the grid, and everything deposited on or
        # gathered from it, must use that node layout.
        axis_names = ["x", "y", "z"][:ndim]
        periodic = tuple(
            bcs_field is not None
            and bcs_field.get_bc(dir=axis, side="min").bc_type == "periodic"
            for axis in axis_names
        )
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            comm=comm,
            domain_decomposition=domain_decomposition,
            periodic=periodic,
        )
        if objects is None:
            objects = []
        self._objects = objects
        self._dimensions = ["x", "y", "z"][: self.ndim]
        self._bcs_field = bcs_field
        self._bcs_particles = bcs_particles

        if self.bcs_particles is not None:
            for dim in self._dimensions:
                for side in ("min", "max"):
                    bc_type = self.bcs_particles.get_bc(dir=dim, side=side).bc_type
                    if bc_type not in _PARTICLE_BC_TYPES:
                        raise ValueError(
                            f"Unknown particle boundary {bc_type!r} on {dim} {side}; "
                            f"expected one of {_PARTICLE_BC_TYPES}.",
                        )

        if ndim == 1:
            bounds_injection_zone = ((1e-3,), (5e-3,))
        elif ndim == 2:
            bounds_injection_zone = ((1e-3, 0e-3), (5e-3, 20e-3))
        elif ndim == 3:
            bounds_injection_zone = ((1e-3, 0e-3, 0e-3), (5e-3, 20e-3, 20e-3))

        self.injection_zone = BoundingBox(
            lower_bound=bounds_injection_zone[0],
            upper_bound=bounds_injection_zone[1],
        )

        self.grid = Grid(
            ndim=ndim,
            lower_bound=self.lower_bound,
            upper_bound=upper_bound,
            num_gridpoints=num_gridpoints,
            comm=comm,
            domain_decomposition=domain_decomposition,
            periodic=periodic,
        )

    @property
    def objects(self) -> list[Shape]:
        """Return objects embedded in the domain."""
        return self._objects

    def plot(
        self,
        ax: Any = None,
        show_grid: bool = False,
        objects: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw the domain outline, its grid lines and its objects into ``ax``.

        Keyword arguments style the outline (``color``, ``linewidth``, ...).
        Returns the axes. Draw a field first with ``field.plot(ax=ax)``.
        """
        from simgeometry.plotting import draw_domain

        return draw_domain(self, ax=ax, show_grid=show_grid, objects=objects, **kwargs)

    @property
    def bcs_field(self) -> Any | None:
        """Return field boundary conditions."""
        return self._bcs_field

    @property
    def bcs_particles(self) -> Any | None:
        """Return particle boundary conditions."""
        return self._bcs_particles

    def particle_boundary_type(
        self,
        dir: int,
        side: str,
        particle_bcs: Any | None = None,
    ) -> str:
        """Return the particle boundary type for one axis side."""
        particle_bcs = particle_bcs or self.bcs_particles
        if particle_bcs is None:
            raise ValueError("No particle boundary conditions are configured")
        return particle_bcs.get_bc(
            dir=self._dimensions[dir],
            side=side,
        ).bc_type

    def particle_boundary_codes(
        self, particle_bcs: Any | None = None
    ) -> tuple[Array, Array]:
        """The kernel codes of the lower and upper side of every axis (int64 arrays).

        ``PARTICLE_BOUNDARY_CODES``: 0 kill, 1 reflect, 2 periodic, 3 reinject,
        as ``apply_particle_boundaries`` and the fused pushes take them.
        """
        return tuple(
            xp.asarray(
                [
                    PARTICLE_BOUNDARY_CODES[
                        self.particle_boundary_type(axis, side, particle_bcs)
                    ]
                    for axis in range(self.ndim)
                ],
                dtype=xp.int64,
            )
            for side in ("min", "max")
        )

    @staticmethod
    def _wrap_periodic(values: Array, lower: float, upper: float) -> Array:
        """Wrap ``values`` into ``[lower, upper)``.

        The domain holds ``lower <= x < upper``.  The modulo can round a
        value just below ``lower`` up to exactly ``upper``, which would then
        count as outside; such a value is the periodic image of ``lower``.
        """
        wrapped = lower + (values - lower) % (upper - lower)
        return xp.where(wrapped >= upper, lower, wrapped)

    def apply_bc_arrays(
        self,
        positions: Array,
        velocities: Array,
        alives: ConfigValue,
        particle_bcs: Any | None = None,
    ) -> Array:
        """Apply particle boundaries and return markers reinjected this push.

        ``positions`` is ``(ndim, N)`` and ``velocities`` ``(nvel, N)``. One
        pass of the ``apply_particle_boundaries`` kernel reflects, wraps or
        kills the markers in place; markers crossing a reinject side get a new
        position from the injection zone here.
        """
        particle_bcs = particle_bcs or self.bcs_particles
        assert positions.shape[0] == self.ndim, (
            f"Expected positions with shape ({self.ndim}, N), got {positions.shape}"
        )
        num_markers = positions.shape[1]
        if particle_bcs is None or num_markers == 0:
            return xp.zeros(num_markers, dtype=bool)

        lower_codes, upper_codes = self.particle_boundary_codes(particle_bcs)
        reinjected = xp.zeros(num_markers, dtype=bool)
        self.boundary_kernel(
            positions,
            velocities,
            alives,
            reinjected,
            xp.asarray(self.lower_bound, dtype=float),
            xp.asarray(self.upper_bound, dtype=float),
            lower_codes,
            upper_codes,
        )
        num_reinjected = int(xp.count_nonzero(reinjected))
        if num_reinjected > 0:
            positions[:, reinjected] = self.injection_zone.draw_random_positions(
                num_reinjected,
            )
        return reinjected
