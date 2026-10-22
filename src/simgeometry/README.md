# simgeometry

Domain, grid, shape, and decomposition primitives for simulation geometry.

The package uses only NumPy, cunumpy, maybempi, and mpiarray outside the
standard library. Internal imports use the `simgeometry` package name.
Optional extras provide labelled evaluation (`evaluate`), sampled meshes (`mesh`),
Matplotlib views (`plot`), and a PyVista viewer (`interactive`). Dependencies are
loaded only when the corresponding operations are called.

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
- `domain_decomposition.py` - The process grid of the MPI ranks (`DomainDecomposition`): neighbours, the physical subdomain and the marker owner of each rank, and `layout()`, the `mpiarray.Layout` of a field on the same grid. `DomainDecomposition.preview(size)` builds the split of a `size`-rank run without a communicator, and `show_3d()`/`show()` draw its subdomains.
- `grid.py` - Grid helpers for geometry, grids, and simulation domains.
- `crossings.py` - `crossings()`: the first boundary crossing of many segments at once, for particle pushers.
- `panels.py` - `boundary_panels()`: the straight edges of a 2D outline with lengths and normals, for boundary-element solvers.
- `shapes.py` - Shapes helpers for geometry, grids, and simulation domains.
- `simulation_domain.py` - Simulation domain helpers for geometry, grids, and simulation domains.
- `utils.py` - Utils helpers for geometry, grids, and simulation domains.
- `_level_set.py` - Exact signed-distance formulas of the primitives and the Illinois root finding on level sets.
- `collection.py` - `GeometryCollection`: named objects, overlap precedence, styles, and conductor voltages (`with_voltages`, `voltages_at`, `dirichlet_values`).

The shared `GeometryAPI` adds sections, transforms, Boolean operations, plotting,
evaluation, surface queries, sampling and export. `GeometryCollection` assigns
names, overlap precedence and conductor voltages. Analytic shapes have a
`level_set` (exact where `level_set_exact`), which Booleans, transforms, sections and
collections propagate and the crossing and distance queries use when present.
`cut_edges` owns solver-independent intersections;
`fractions` estimates cell/face occupancy. Picnic owns stencil coefficients and
its minimum-distance regularization. Tutorials live in `tutorials/` and are
executed into the documentation site.

## Subpackages

- `objects/` - Concrete geometric objects that can be embedded in a simulation domain, including the solids of revolution `Cylinder`, `Tube`, `Cone` and `Torus` (`objects/axisymmetric.py` holds their common frame).
- `tests/` - Test modules for `simgeometry`.
