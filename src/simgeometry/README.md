# simgeometry

Domain, grid, shape, and decomposition primitives for simulation geometry.

The package uses only NumPy, cunumpy, maybempi, and mpiarray outside the
standard library. Internal imports are relative.
Matplotlib is needed only when calling `Shape2D.plot` or
`SimulationDomain.plot`.

`SimulationDomain` accepts boundary-condition objects through their public
`get_bc(dir=..., side=...).bc_type` interface. Its default particle-boundary
implementation runs on the active cunumpy backend. Applications with an
accelerated implementation can set `SimulationDomain.boundary_kernel` to a
callable with the same eight-array signature. Likewise,
`BoundingBox.random` accepts a callable taking a shape and returning random
values; applications can configure their own seeded stream.

## Modules

- `__init__.py` - Package exports for geometry, grids, and simulation domains.
- `boundingbox.py` - Boundingbox helpers for geometry, grids, and simulation domains.
- `plotting.py` - Optional Matplotlib drawing for shapes and domains.
- `domain_decomposition.py` - The process grid of the MPI ranks (`DomainDecomposition`): neighbours, the physical subdomain and the marker owner of each rank, and `layout()`, the `mpiarray.Layout` of a field on the same grid.
- `grid.py` - Grid helpers for geometry, grids, and simulation domains.
- `point.py` - Point helpers for geometry, grids, and simulation domains.
- `pointlist.py` - Pointlist helpers for geometry, grids, and simulation domains.
- `shapes.py` - Shapes helpers for geometry, grids, and simulation domains.
- `simulation_domain.py` - Simulation domain helpers for geometry, grids, and simulation domains.
- `utils.py` - Utils helpers for geometry, grids, and simulation domains.

## Subpackages

- `objects/` - Concrete geometric objects that can be embedded in a simulation domain.
- `tests/` - Test modules for `simgeometry`.
