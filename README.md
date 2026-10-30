# simgeometry


Geometry primitives for numerical simulations: bounded domains,
structured grids, MPI domain decomposition, and 1D, 2D, and 3D shapes.
Box, ball, and polygon containment have NumPy, Pyccel, CUDA, and Metal
implementations. Originally developed for
[Picnic](https://github.com/max-models/picnic).

## Install

``` bash
pip install simgeometry
```

For development, install this checkout with `pip install -e '.[test]'`.
NumPy, cunumpy, maybempi, mpiarray and scope-profiler are runtime
dependencies. Matplotlib is optional. Install `simgeometry[plot]` for
plotting, labelled evaluation and surface meshing, or
`simgeometry[interactive]` for interactive 3D clipping with PyVista.
Install `simgeometry[mpi]` for MPI, `simgeometry[pyccel]` for compiled
CPU containment kernels, or `simgeometry[gpu]` for CUDA.

## Example

``` python
from simgeometry.grid import Grid
from simgeometry.objects import Circle

boundary = Circle(midpoint=(0.5, 0.5), radius=0.25)
grid = Grid(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=(32, 32))
mask = boundary.node_mask(grid)
```

`SimulationDomain` accepts field and particle boundary objects through
their `get_bc(dir=..., side=...).bc_type` interface. Applications may
supply a seeded random function with `BoundingBox.random` and an
accelerated particle boundary implementation with
`SimulationDomain.boundary_kernel`.

## Inspect, slice, and compose

``` python
from simgeometry import Sphere

shell = Sphere((0, 0, 0), 1) - Sphere((0, 0, 0), 0.4)
shell.show_3d(opacity=0.6)
shell.slice(z=0.2).show()
inside = shell.evaluate(x=[0, 0.8], y=0, z=0)
```

Shapes support named collections, Boolean operations, affine transforms,
surface queries, point sampling, mesh export, cut-edge geometry and
approximate cell/face fractions. See the [executed
examples](https://max-models.github.io/simgeometry/examples/) and [API
guide](https://max-models.github.io/simgeometry/getting-started/geometry-api/)
for examples and the accuracy limits of sampled operations.

## Level sets, solids of revolution and conductors

``` python
from simgeometry import GeometryCollection, Tube

ring = lambda z: Tube((0, 0, z - 0.4), (0, 0, z + 0.4), 1.0, 1.6, fillet=0.1)
lens = GeometryCollection(
    {"entrance": ring(-1.2), "focus": ring(0.0), "exit": ring(1.2)},
    voltages={"entrance": 0.0, "focus": 10e3, "exit": 0.0},
)
scan = lens.with_voltages(focus=lambda t: 10e3 * min(t / 1e-6, 1.0))
values = scan.dirichlet_values(grid, t=0.5e-6)  # labels and voltages per node
distance = lens["focus"].level_set(points)  # exact signed distance
```

`Cylinder`, `Tube`, `Cone` and `Torus` take any axis. Analytic shapes
have a `level_set(points)`, negative inside, with `level_set_exact`
where it is the Euclidean signed distance; Booleans and transforms
propagate it. Edge crossings refine it with a few secant steps instead
of bisecting `contains`, and `signed_distance` returns an exact one
directly. The [conductors
example](https://max-models.github.io/simgeometry/examples/conductors-and-level-sets/)
builds an Einzel lens.

## Preview a domain decomposition

``` python
from simgeometry import Sphere
from simgeometry.domain_decomposition import DomainDecomposition

preview = DomainDecomposition.preview(8, decompose=[True, True, True])
preview.proc_sizes  # [2, 2, 2], as on 8 MPI ranks
boxes = preview.subdomain_boxes((0, 0, 0), (2, 1, 1), (33, 17, 17))
preview.show_3d((0, 0, 0), (2, 1, 1), (33, 17, 17), shape=Sphere((1, 0.5, 0.5), 0.35))
```

`DomainDecomposition.preview(size)` computes the split a `size`-rank run
would use without a communicator, for inspection and planning: index
ranges, subdomains and owner ranks for every rank. `show_3d` draws one
box per rank (with `simgeometry[interactive]`) and `show` one rectangle
per rank of a 2D decomposition. See the [parallel grids
example](https://max-models.github.io/simgeometry/examples/parallel-grids/).

## Profiling

simgeometry ships
[scope-profiler](https://github.com/max-models/scope-profiler) regions
for its expensive operations (`simgeometry.node_mask`,
`simgeometry.crossings`, `simgeometry.to_mesh`, …). They are off by
default and cost almost nothing. Turn them on in an application that
profiles itself with the `simgeometry` tag:

``` python
import scope_profiler as sp

sp.setup(enable_tags=["simgeometry"])
```

or `SCOPE_PROFILER_ENABLE_TAGS=simgeometry`, or
`scope-profiler run --enable-tag simgeometry`.

## Development

``` bash
pytest src/simgeometry/tests
```

Documentation: <https://max-models.github.io/simgeometry/>
