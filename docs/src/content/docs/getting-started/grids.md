---
title: Grids and coordinates
description: Work with structured nodes, physical coordinates, and periodic axes.
---

`Grid` defines a bounded, structured set of nodes. Pass one lower bound, upper
bound, and node count per axis:

```python
import numpy as np

from simgeometry.grid import Grid

grid = Grid(
    ndim=2,
    lower_bound=(-1.0, 2.0),
    upper_bound=(1.0, 4.0),
    num_gridpoints=(11, 21),
)

assert np.allclose(grid.grid_spacing, (0.2, 0.1))
assert np.allclose(grid.node_coordinates((5, 10)), (0.0, 3.0))
```

For a bounded axis with `n` nodes, spacing is `(upper - lower) / (n - 1)`;
both endpoints are nodes. `node_coordinates(index)` includes the lower bound.
In 1D, pass a plain integer index.

`node_points()` returns a host NumPy array with shape `(ndim, N)`: each column
is one node, in C order. `get_grid_points()` returns one coordinate array per
axis. `indices_from_mask(mask)` converts a grid-shaped boolean mask to node
indices in the same order.

## Periodic axes

```python
periodic_grid = Grid(
    ndim=2,
    lower_bound=(0.0, 0.0),
    upper_bound=(1.0, 1.0),
    num_gridpoints=(4, 3),
    periodic=(True, False),
)

assert periodic_grid.get_grid_points()[0].tolist() == [0.0, 0.25, 0.5, 0.75]
assert periodic_grid.get_grid_points()[1].tolist() == [0.0, 0.5, 1.0]
```

A periodic axis has `n` cells over its length; its upper endpoint is not a
duplicate node. Each nonperiodic axis still includes both endpoints.

`BoundingBox.contains()` uses half-open bounds (`lower <= x < upper`). Thus a
node can lie at a nonperiodic upper endpoint even though `contains()` reports
that coordinate as outside the box.
