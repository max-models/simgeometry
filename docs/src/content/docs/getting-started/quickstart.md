---
title: Quickstart
description: Build a grid and query a shape with simgeometry.
---

Install `simgeometry`, then use its geometry modules from Python:

```python
from simgeometry.grid import Grid
from simgeometry.objects import Circle

grid = Grid(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=(32, 32))
circle = Circle(midpoint=(0.5, 0.5), radius=0.25)
mask = circle.node_mask(grid)
```

`mask` marks the grid nodes inside the circle.

Continue with [grids and coordinates](/simgeometry/tutorials/grids-and-coordinates/) to
understand node spacing and periodic axes, or [shapes and node
masks](/simgeometry/tutorials/shapes-and-node-masks/) for point queries and index lists.
For MPI layouts and multiple patches, see [parallel and multipatch
grids](/simgeometry/tutorials/parallel-and-multipatch-grids/).
