---
title: Geometry inspection and queries
description: Plotting, sections, labelled sampling, surface queries and cut-edge geometry.
---

Install `simgeometry[plot]` for plotting, evaluation and surface meshing, or
`simgeometry[interactive]` for the optional PyVista viewer, drawn with plasma-plots. The base package keeps
NumPy, cunumpy, maybempi and mpiarray as its only runtime dependencies.

```python
from simgeometry import Circle, GeometryCollection, Sphere

ring = Circle((0, 0), 1) - Circle((0, 0), 0.4)
ring.show()

shell = Sphere((0, 0, 0), 1) - Sphere((0, 0, 0), 0.4)
shell.show_3d(opacity=0.5)
shell.show(z=0.2)
shell.slice(origin=(0, 0, 0), normal=(1, 1, 1)).show()
shell.slice(origin=(0, 0, 0), normal=(1, 1, 1)).show_3d(backend="pyvista")
```

## Coordinates and return values

| Operation | Input | Result |
| --- | --- | --- |
| `contains(point)` | One coordinate vector | Boolean |
| `contains_points(points)` | `(ndim, n)` array | Boolean array on the selected cunumpy backend |
| `level_set(points)` | `(ndim, n)` array | Signed level set on the selected cunumpy backend, negative inside |
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
| `yee_fractions(grid)` | Grid | Edge-length and face-area fractions of a Yee grid and the surface's traces, exact for planes |
| `collection.dirichlet_values(grid, t=...)` | Grid or `(ndim, n)` | Host labels and voltages; NaN outside conductors |
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
`section.outlines()` returns the section's boundary loops as `(2, n)` arrays in
plane coordinates, sampled at `resolution` points per axis (outer boundaries
counter-clockwise, holes clockwise, as for 2D Boolean shapes);
`outlines(world=True)` maps them onto the plane as `(3, n)` arrays.
`section.show_3d()` draws the section in context: the sliced shape translucently,
the cutting plane over the shape's bounds and the outline on it
(`backend="pyvista"` for an interactive view).

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

Without `clip=True`, wireframes or extra PyVista options, the PyVista backend draws the
surfaces with plasma-plots' `pyvista_surfaces`, one colour per object. For a collection
with voltages, `color_by="voltage"` colours each conductor by its voltage at time `t`
(other objects grey):

```python
viewer = lens.show_3d(backend="pyvista", color_by="voltage", t=0.0)
```

`level_set_field(resolution=...)` samples the level set on a grid in the layout of
plasma-plots' 3D views (`eta1`, `eta2`, `eta3` with physical `X`, `Y`, `Z`), so any of
its levels can be drawn as an isosurface:

```python
import plasma_plots

field = shell.level_set_field(resolution=64)
field.plasma.plot.isosurface(values=[0.0, 0.1], show_domain=False)
```

In a notebook, `plasma_plots.scene(plotter)` shows a plotter as an interactive scene,
also on these docs pages.

## Level sets

Analytic shapes have `level_set(points)` (`has_level_set` is true): negative inside,
positive outside and zero on the boundary, with the sign of `signed_distance`, so it
flips for inverted shapes. Its magnitude never exceeds the distance to the boundary,
and where `level_set_exact` is true it is the Euclidean signed distance.

| Shape | Level set |
| --- | --- |
| `Line`, `Circle`, `Sphere`, `Rectangle`, `Cuboid` (with or without fillets) | exact |
| `Polygon`, `Triangle` | exact: distance to the nearest edge, signed by `contains` (one pass per edge) |
| `Aperture2D` | exact while the plates do not touch |
| `Cylinder`, `Tube`, `Cone`, `Torus` (any axis, with or without fillets) | exact |
| `a \| b`, `a & b`, `a - b` | `min(a, b)`, `max(a, b)`, `max(a, -b)`: a bound, not exact |
| rotations, translations, reflections, uniform scaling | exact if the source is |
| nonuniform scaling, shears | the source's times the smallest stretch: a bound |
| sections, collections | the source's on the plane, the minimum over the objects: a bound |
| `FunctionObject2D/3D`, `Aperture3D`, `Polyline` | none |

Where a shape has a level set, grid-edge crossings (`intersection_fractions`,
`bracketed_edge_geometry`, the sampled path of `cut_edges` and `crossings`) start from
the linear interpolation of the level set at the two ends of the edge and refine the
root with a few Illinois (modified regula falsi) steps, instead of bisecting `contains`
48 times. Membership still decides which edges are cut; an edge whose level-set values
do not change sign (a node exactly on the surface, a periodic seam) is bisected as
before. Crossings land on the surface to round-off after about 8 evaluations per edge
instead of 50. `signed_distance` returns an exact level set directly, surface normals
of other level-set shapes are its gradient, and `to_mesh` contours it, so meshes of
these shapes place their vertices between samples.

`cell_fractions` keeps its subcell quadrature: interpolating corner level-set values
(AMReX-style) is more accurate in smooth cut cells but less accurate at corners and
misses features thinner than a cell, which the subcell samples catch.

## Conductors

`GeometryCollection(objects, voltages={"electrode": 1000.0, "wall": 0.0})` makes
named objects conductors. A voltage is a number or a function of the time `t`.
`collection.with_voltages(electrode=500.0)` returns the same objects, bounds and styles
with other voltages (a mapping works for any name; `None` removes a voltage), so a
parameter scan does not rebuild the geometry. `voltages_at(t)` evaluates the voltages
and `conductors` lists the names with one.

`collection.dirichlet_values(grid, t=t)` returns `DirichletValues` with `labels` and
`voltages` per node (grid-shaped), or per point for `(ndim, n)` points. Labels follow
`labels()`: 0 outside every object, the last listed object winning in overlaps. The
voltage is NaN outside the conductors, including inside an object without a voltage;
`values.conductor` marks the nodes a field solver fixes. See the
[conductors tutorial](/simgeometry/tutorials/conductors-and-level-sets/) for an
Einzel lens.

## Exact geometry and numerical approximations

- Balls, axis-aligned boxes (also rounded with `fillet`) and simple polygons have
  analytic segment intersections
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
  distance is inferred from an arbitrary boolean function. Shapes with an exact level
  set return it from `signed_distance`.
- Surface meshing uses scikit-image's
  [marching contours/cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html).
  Shapes with a level set use it as continuous implicit values; other shapes use
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

`shape.outlines(max_spacing=...)` returns the boundary of a 2D shape as `(2, n)` loops,
one per piece (an `Aperture2D` has two), with no edge longer than `max_spacing`; circles put
every vertex on the circle. Loops keep the shape on their left: outer boundaries run
counter-clockwise and holes clockwise. `max_spacing` can also be a function of `(2, m)`
points returning `(m,)` spacings, which grades the vertices (fine near an aperture tip,
coarse far from it); see the
[boundary outlines tutorial](/simgeometry/tutorials/boundary-outlines/). Transformed 2D
shapes transform their source's loops. Boolean 2D shapes (`a | b`, `a & b`, `a - b`,
nested) have outlines through shapely (`pip install 'simgeometry[clip]'`): each operand
enters as the polygon of its own outline, so curved operands are approximated by inscribed
polygons. Inverted operands, open curves and function objects have no outline to combine.
`exact_outline` is true where the outline is the exact boundary (polygons, rectangles,
triangles, apertures, polylines and their affine images).

A `Polyline` is an open curve of zero thickness, for thin plates and wires: it contains
no points, `closed` is `False`, and its outline runs from its first vertex to its last.
`Aperture2D.slit(axial_position, half_width, thickness, length, axis=...)` builds two
rectangular plates; `axis` is the beam axis (0 for x, 1 for y).

`shape.boundary_panels(max_spacing=...)` turns the outline into straight panels
(`simgeometry.panels.Panels`): `start`, `end`, `midpoint`, `length`, unit `tangent` and
`normal`, and the loop each panel belongs to (`piece`). The normal is to the right of the
panel's direction, so it points out of the geometry, into the hole at a hole; for an open
curve it is the right-hand side of the curve. Boundary-element and panel solvers can
discretise electrodes and walls from it directly.

`shape.crossings(start, end)` finds where each of N steps, given as `(ndim, N)` start and
end positions, first crosses the shape's boundary, entering or leaving. It returns `hit`,
`fraction` in (0, 1], `point` and a unit `normal` pointing out of the region where
`contains` is true. Because the whole step is tested, a fast particle cannot pass through a
thin plate between two end points. Polygonal 2D shapes, polylines, circles, spheres,
axis-aligned boxes and their affine transforms are exact. Other shapes are sampled at
`samples` points per step and bisected, so features thinner than `1 / samples` of a step can
be missed, and their normals are estimated (`exact` is `False`). A crossing at the very
start of a step is not reported, so a particle stopped on a surface is not counted twice;
grazing contacts are not crossings.

`simgeometry.cut_edges.bracketed_edge_geometry()` is the batched geometry path for
Picnic's Shortley–Weller stencil. It visits fluid nodes with solid neighbours and
assumes a single crossing in each bracket. Fractions have no numerical lower clamp.
Picnic applies its own minimum edge fraction before assembling coefficients. Its
`points` and `normals` are the crossings in world coordinates and the outward normals
of the entered object, `(ndim, n)` arrays in the order of `fractions` (periodic images
wrapped into the domain); `crossing_owners` gives each crossing's object.

The full `cut_edges()` inspection can detect multiple crossings and objects lying
between fluid nodes. It visits every edge, so it is intended for inspection and
geometry preparation rather than repeated use in a solver timestep. Collection
hits include each object's interfaces, including interfaces hidden by overlaps;
Boolean shapes instead expose only the resulting solid's boundary.

On a 3D grid, `show_3d()` on either result draws the translucent surface and the
crossings, with `normals=True` the outward normals and with `nodes=True` the solid grid
nodes; `backend="pyvista"` returns a PyVista plotter, otherwise Matplotlib 3D axes.

`shape.yee_fractions(grid)` gives the cut-cell geometry of a staggered (Yee) grid on the
grid's nodes, 2D or 3D, for conformal finite-difference and finite-integration schemes
(Dey–Mittra edges and faces, embedded boundaries). `edges[a]` is the fraction of each edge
along axis `a` inside the shape (shape: cells along `a`, nodes across), `faces[a]` the
fraction of each face normal to `a` (nodes along `a`, cells across; in 2D one family, the
cells). `traces[a]` is the length of the curve where the surface crosses each face and
`projected_traces[a]` that length weighted by `|n x e_a|`, the sine between the surface
normal and the face's; summed over all faces, each times the spacing along its normal, the
projected traces give twice the surface's area. On a periodic axis the last edge or face
closes the period (no duplicate). Edge fractions use every crossing along the edge (so a
slit narrower than a cell keeps its gap); face fractions and traces integrate lines across
the face, piecewise Gauss–Legendre between the crossings on the face's edges, in both
directions.
Both are exact for planes; for curved surfaces they converge at second order or better (a
circle's area and perimeter to `2e-7` at 80 cells across). A surface lying on grid lines
(walls of an axis-aligned box or slit on node planes) is treated like a staircase: the edges
and faces in it count as inside, whatever `contains` returns on the surface itself, and its trace
counts for the face on its outside. Shapes without analytic crossings are
sampled with `samples` points per segment. Where a shape has a level set, segments farther
from the surface than half their length are not searched.

`diagnose(grid)` reports node overlaps, objects without solid nodes, multiple-hit
edges, and very short edge distances. These observations do not prove that all
features are resolved. See the [cut-edge tutorial](/simgeometry/tutorials/grid-intersections-and-cut-edges/)
and the [sampling/export tutorial](/simgeometry/tutorials/sampling-export-and-diagnostics/).
