# simgeometry


Geometry primitives for numerical simulations: bounded domains,
structured grids, MPI domain decomposition, multipatch geometry, and 1D,
2D, and 3D shapes. Box, ball, and polygon containment have NumPy,
Pyccel, CUDA, and Metal implementations. Originally developed for
[Picnic](https://github.com/max-models/picnic).

## Install

``` bash
pip install simgeometry
```

For development, install this checkout with `pip install -e '.[test]'`.
NumPy, cunumpy, maybempi, and mpiarray are runtime dependencies.
Matplotlib is optional. Install `simgeometry[plot]` for plotting,
labelled evaluation and surface meshing, or `simgeometry[interactive]`
for interactive 3D clipping with PyVista. Install `simgeometry[mpi]` for
MPI, `simgeometry[pyccel]` for compiled CPU containment kernels, or
`simgeometry[gpu]` for CUDA.

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
tutorials](https://max-models.github.io/simgeometry/tutorials/) and [API
guide](https://max-models.github.io/simgeometry/getting-started/geometry-api/)
for examples and the accuracy limits of sampled operations.

## Development

``` bash
pytest src/simgeometry/tests
```

Documentation: <https://max-models.github.io/simgeometry/>
