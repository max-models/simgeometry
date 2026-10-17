---
title: Geometry inspection and queries
description: Plotting, sections, labelled sampling, surface queries and cut-edge geometry.
---

Install `simgeometry[plot]` for plotting, evaluation and surface meshing, or
`simgeometry[interactive]` for the optional PyVista viewer. The base package keeps
NumPy, cunumpy, maybempi and mpiarray as its only runtime dependencies.

```python
from simgeometry import Circle, GeometryCollection, Sphere

ring = Circle((0, 0), 1) - Circle((0, 0), 0.4)
ring.show()

shell = Sphere((0, 0, 0), 1) - Sphere((0, 0, 0), 0.4)
shell.show_3d(opacity=0.5)
shell.show(z=0.2)
shell.slice(origin=(0, 0, 0), normal=(1, 1, 1)).show()
```

## Coordinates and return values

| Operation | Input | Result |
| --- | --- | --- |
| `contains(point)` | One coordinate vector | Boolean |
| `contains_points(points)` | `(ndim, n)` array | Boolean array on the selected cunumpy backend |
| `evaluate(x=..., y=..., z=...)` | Scalars, coordinate sequences or DataArrays | Host `xarray.DataArray` |
| `node_mask(grid)` | Grid | Host boolean array with grid shape |
| `slice(z=value)` | One fixed coordinate | Queryable section |
| `slice(origin=..., normal=...)` | 3D plane | Queryable section with plane coordinates |
| `closest_point(points)` | One vector or `(ndim, n)` | Host array in the same layout |
| `signed_distance(points)` | One vector or `(ndim, n)` | Scalar or host `(n,)` array, negative inside |
| `surface_normal(points)` | Boundary points | Host outward unit vectors; NaN if undefined |
| `intersect_segment(start, end)` | Two coordinate vectors | Ordered hits, fractions, normals and crossing directions |
| `cut_edges(grid)` | Grid | All detected edge hits with object identities |
| `cell_fractions(grid)` | Grid | Approximate solid volume and face fractions |
| `to_mesh()` | Resolution and optional bounds | Vertices, connectivity, centres, normals and sampling spacing |

Inspection and meshing run on the calling process and do not implicitly gather
MPI data. Host copies may be made when the containment backend uses a GPU.

### Evaluation

Scalars fix an axis. Sequences form a tensor-product mesh. DataArrays sharing a
dimension (for example `points`) describe scattered samples. Omitted coordinates
use `grid=` nodes, or uniformly sample the shape bounds with `resolution` points
per axis. Supply `bounds=(lower, upper)` for a different sampling region.

Shapes evaluate to occupancy. Named collections evaluate to labels: zero is
background; 1..N follow insertion order; the last listed object wins in overlaps.
`quantity="occupancy"` requests the collection's union mask.
`quantity="signed_distance"` evaluates boundary distance.

Axis-aligned sections retain remaining world-coordinate names. For example,
`sphere.slice(x=0).evaluate(y=ys, z=zs)` samples a yz plane. Oblique sections use
local x/y plane coordinates. `section.to_world()` and `section.to_local()` map
between the plane and world space; the latter is an orthogonal projection.

### Construction and ownership

`a | b`, `a & b`, and `a - b` construct union, intersection, and difference.
`translated()`, `rotated()` and `scaled()` return independent snapshots, as does
`slice()`. Rotation angles are degrees; rotation/scaling use world origin zero
unless `origin=` is given. Affine transformations accept invertible matrices,
including nonuniform scales and reflections.

Existing primitive `translate()` and `mirror()` methods remain in-place methods.
A `GeometryCollection` retains references to its named objects. Its labels always
follow insertion order. `styles={name: {"color": ..., "alpha": ...}}` supplies
per-object colors and opacity to 3D plots; 2D label maps use the per-object colors
and a shared opacity. Collection bounds are the viewing region, not a membership
clip. Set explicit bounds if referenced objects will move beyond their original
extent. Boolean bounds are conservative; they need not tightly bound the result.

### Plotting

`show()` selects a 2D or 3D view; `show(z=...)` draws a slice.
`plot(ax=...)` composes into existing axes without displaying.
`show_3d()` supports opacity, wireframes, labels, normals, bounds and clipping.
These return Matplotlib axes; use `display=False, save="image.png"` in scripts.
Existing primitive 2D `plot()` methods keep their polygon drawing behavior.

`domain.show(grid=True)` overlays its grid. `grid.show()` displays the grid itself.
On other shapes/collections pass `grid=grid`. `nodes=True` adds grid node markers.

For an interactive clipping-plane widget:

```python
viewer = shell.show_3d(backend="pyvista", clip=True)
viewer.close()
```

The PyVista backend returns a plotter owned by the caller. A static plane specified
as `clip=(origin, normal)` retains its positive half-space. Clipping leaves an open
surface and does not insert caps. Plotting meshes and normal arrows are faceted
approximations; increase `resolution` to inspect small features. PyVista's
[plane widget](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh_clip_plane)
provides the interactive clipping behavior.

## Exact geometry and numerical approximations

- Balls, axis-aligned boxes and simple polygons have analytic segment intersections
  and closest-point distances. Similarity transforms preserve exact closest-point
  queries. General affine transformations preserve analytic segment intersections.
- Composite segment queries merge operand hits and discard hidden interfaces.
  General membership functions scan `samples` intervals and then bisect detected
  crossings. `hits.exact` identifies the analytic path. Sampled searches can miss
  thin features or tangencies. Fractions within `tolerance` are merged.
- Primitive normal queries are analytic. General normal queries estimate membership
  on nearby directions. Normals at corners are not unique; a primitive can select
  one incident face. A generic query away from a surface returns NaN.
- Closest-point and distance queries on general shapes use a sampled contour or
  triangular mesh. `resolution` controls the surface sampling; no exact signed
  distance is inferred from an arbitrary boolean function.
- Surface meshing uses scikit-image's
  [marching contours/cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html).
  Primitive balls and boxes use continuous implicit values; other shapes use
  membership samples. Mesh spacing is metadata, not a certified error bound.
  A surface touching sampling bounds can be open.
- `sample_interior()` uses uniform rejection sampling and a finite `max_draws`.
  Supply bounds for unbounded/inverted objects. `sample_surface()` samples by
  length or area on the mesh. A seed or NumPy Generator makes sampling repeatable.
- Export formats are NPZ, OBJ, and 3D STL. Export does not certify watertightness.
- `cell_fractions(samples_per_axis=...)` uses midpoint subcell quadrature. Its
  values are solid occupancy fractions, not an exact finite-volume reconstruction.
  Face arrays have one extra entry along their normal axis, including both
  periodic boundary copies. Increase subcell sampling to check convergence.

## Solver integration

`shape.outlines(max_spacing=...)` returns the boundary of a 2D shape as closed,
counter-clockwise `(2, n)` loops, one per piece (an `Aperture2D` has two), with no
edge longer than `max_spacing`; circles put every vertex on the circle. `max_spacing` can
also be a function of `(2, m)` points returning `(m,)` spacings, which grades the vertices
(fine near an aperture tip, coarse far from it); see the
[boundary outlines tutorial](/simgeometry/tutorials/boundary-outlines/). Boundary-element
and panel solvers can discretise electrodes and walls from it directly. Boolean shapes have
no outline; transformed 2D shapes transform their source's.

`simgeometry.cut_edges.bracketed_edge_geometry()` is the batched geometry path for
Picnic's Shortley–Weller stencil. It visits fluid nodes with solid neighbours and
assumes a single crossing in each bracket. Fractions have no numerical lower clamp.
Picnic applies its own minimum edge fraction before assembling coefficients.

The full `cut_edges()` inspection can detect multiple crossings and objects lying
between fluid nodes. It visits every edge, so it is intended for inspection and
geometry preparation rather than repeated use in a solver timestep. Collection
hits include each object's interfaces, including interfaces hidden by overlaps;
Boolean shapes instead expose only the resulting solid's boundary.

`diagnose(grid)` reports node overlaps, objects without solid nodes, multiple-hit
edges, and very short edge distances. These observations do not prove that all
features are resolved. See the [cut-edge tutorial](/simgeometry/tutorials/grid-intersections-and-cut-edges/)
and the [sampling/export tutorial](/simgeometry/tutorials/sampling-export-and-diagnostics/).
