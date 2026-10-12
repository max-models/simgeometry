---
title: Shapes and node masks
description: Select grid nodes and test points against geometric objects.
---

Import concrete shapes from `simgeometry.objects`. The package provides `Line`
in 1D; `Circle`, `Rectangle`, `Triangle`, `Polygon`, `Aperture2D`, and
`FunctionObject2D` in 2D; and `Sphere`, `Cuboid`, `Aperture3D`, and
`FunctionObject3D` in 3D.

```python
import numpy as np

from simgeometry.grid import Grid
from simgeometry.objects import Circle

grid = Grid(
    ndim=2,
    lower_bound=(0.0, 0.0),
    upper_bound=(1.0, 1.0),
    num_gridpoints=(11, 11),
)
circle = Circle(midpoint=(0.5, 0.5), radius=0.25)

assert circle.contains((0.5, 0.5))
mask = circle.node_mask(grid)  # host NumPy bool array, shape (11, 11)
indices = circle.get_indices_in_object(grid)  # (i, j) tuples in grid order
assert mask.dtype == np.bool_
assert indices == grid.indices_from_mask(mask)
```

Use `contains()` for one point and `contains_points()` for a batch. Batch
points are component-major: an array of shape `(ndim, N)` where each column
is a point. The result has one boolean value per column.

```python
points = np.array([[0.5, 0.0], [0.5, 0.0]])
assert circle.contains_points(points).tolist() == [True, False]
```

Shapes also accept `inverted=True` to select the region outside their
geometry. For example, `Circle(..., inverted=True)` treats a circular cavity
as excluded. The shape's bounding box continues to describe the circle itself.

`Circle`, `Rectangle`, `Triangle`, and other concrete shapes support geometry
operations such as `copy()`, `translate()`, and `mirror()`. See the
[API reference](/simgeometry/api/simgeometry/) for each shape's constructor and
available methods.
