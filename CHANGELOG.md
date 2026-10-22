# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `DomainDecomposition.preview(size, decompose=..., periodic=..., rank=...)` (or
  `proc_sizes=`): the decomposition a `size`-rank run would use, without a communicator, for
  inspection and planning; `subdomain_boxes(...)` returns every rank's subdomain corners,
  `show_3d(...)` draws them as boxes coloured by rank (plasma-plots, optional `shape=`) and
  `show(...)` as rectangles in 2D. The tutorial "Parallel grids" shows an 8-rank scene.
- `bracketed_edge_geometry(...).points` and `.normals`: the crossings in world coordinates
  and the outward normals there, `(ndim, n)` like `cut_edges()`; `crossing_owners` indexes
  each crossing's object. `BracketedEdgeGeometry` now keeps `spacing`, `periodic` and
  `objects`, and `CutEdges` its `geometry`.
- `CutEdges.show_3d()` and `BracketedEdgeGeometry.show_3d()`: the translucent surface, the
  crossings and, optionally, the outward normals and solid nodes in 3D (PyVista or
  Matplotlib). The tutorials "Grid intersections and cut edges" and "Conductors and level
  sets" use them.
- The PyVista backend of `show_3d` draws the surfaces with plasma-plots' `pyvista_surfaces`
  (plasma-plots is now part of the `interactive` extra); `color_by="voltage"` colours a
  collection's conductors by their voltage at time `t`.
- `level_set_field()`: the level set on a grid in the layout of plasma-plots' 3D views, for
  `field.plasma.plot.isosurface(...)`; `evaluate(quantity="level_set")`.
- Interactive 3D scenes in the tutorials "Inspecting geometry" and "Conductors and level
  sets" (starlight-theme v0.5.0, `plasma_plots.scene`).
- More interactive 3D scenes in the tutorials: the new solids and a 3D node mask, 3D
  transforms and Booleans, an oblique slice through a shell, interior and surface samples,
  3D cut edges with normals, the level set of a slotted electrode and its grid-edge crossings.
- `DomainDecomposition.owner_ranks(positions, lower_bound, upper_bound, num_gridpoints,
  layout)` and `owned_mask(...)`: vectorised position-to-rank lookup for NumPy and CuPy arrays
  (`(ndim, n)` by default, `layout="particle_major"` for `(n, ndim)`).
- `DomainDecomposition.position_decomposition(lower_bound, upper_bound, num_gridpoints)`: the
  decomposition with the grid bound in, implementing the `PositionDecomposition` protocol
  (`comm`, `rank`, `size`, `owners(positions)`, `owned_mask(positions)`).
- `Shape2D.outlines(max_spacing=None)`: the closed boundary loops of a 2D shape as
  counter-clockwise `(2, n)` arrays, with edges refined to at most `max_spacing`, or graded
  by a spacing function of position; also on transformed 2D shapes. Plotting draws shapes
  from it.
- Tutorial "Boundary outlines": uniform and graded outlines, their convergence, and
  outlines of transformed shapes.
- `outlines()` on 2D Boolean shapes (union, intersection, difference, nested), through
  shapely (the `clip` extra): outer boundaries counter-clockwise, holes clockwise.
- `shape.crossings(start, end)`: the first boundary crossing of many segments at once
  (fraction, point and outward normal), so particles cannot step through thin objects;
  analytic for polygons, polylines, circles, spheres, boxes and their affine transforms,
  sampled for other shapes.
- `shape.boundary_panels(max_spacing)`: the outline as straight panels with midpoints,
  lengths, tangents and outward normals (`simgeometry.panels.Panels`).
- `Polyline`: an open curve of zero thickness (thin plates, wires), with an open outline,
  panels and crossings.
- `Aperture2D.slit(...)`: two rectangular plates. `exact_outline` and `closed` flags on 2D
  shapes.
- The tutorial covers holes, polylines, panels and crossings.
- scope-profiler regions for the expensive operations (`simgeometry.node_mask`,
  `simgeometry.crossings`, `simgeometry.to_mesh`, ...), off unless the application enables
  the `simgeometry` tag (`sp.setup(enable_tags=["simgeometry"])`). scope-profiler is now a
  runtime dependency.
- `shape.level_set(points)`: an analytic level set, negative inside and positive outside like
  `signed_distance`, with `has_level_set` and `level_set_exact` (the Euclidean signed
  distance). Exact for lines, circles, spheres, rectangles, cuboids, polygons, triangles,
  apertures and the new solids; Booleans combine level sets by min/max, transforms scale
  them by their smallest stretch (exact for rigid motions and uniform scaling), sections and
  collections pass them on. Function objects and `Aperture3D` have none.
- `Cylinder`, `Tube` (hollow cylinder), `Cone` (frustum, or cone with one radius 0) and
  `Torus`, along any axis, with exact level sets, volumes, surface areas and tight bounding
  boxes; `fillet=` rounds the edges of cylinders and tubes.
- Conductors on `GeometryCollection`: `voltages={name: value or function of t}`,
  `with_voltages(...)` (same geometry, other voltages), `voltages_at(t)`, `conductors`, and
  `dirichlet_values(grid or points, t=...)` with the label and voltage of each node
  (`DirichletValues`; NaN outside conductors).
- Tutorial "Conductors and level sets": an Einzel lens of three ring electrodes with voltages.
- `benchmarks/level_set_crossings.py`: edge crossings on level sets against bisection.

### Changed

- `__version__` is read from the installed package metadata (`importlib.metadata`), not hardcoded,
  so it can no longer disagree with `pyproject.toml` (the 0.1.1 wheel reported 0.1.0).
- One out-of-range rule for owners: per axis, positions below the lower bound belong to the
  first block and positions at or above the upper edge to the last (the block index is clipped).
  `owner_rank_for_position` now calls `owner_ranks`; positions below the lower bound used to go
  to the last block.
- `Aperture2D(axis=...)` now sets the beam axis: `axis=1` builds the plates with the beam
  along y. Before, `axis` was ignored and the beam always ran along x, as it still does
  for `axis=0`.
- 2D shapes have no `get_vertices()` any more; use `outlines()`, which returns `(2, n)`
  arrays. `PointList` and `Point` are gone.
- `intersection_fractions` (and so `bracketed_edge_geometry`) refines crossings of shapes with
  a level set by linear interpolation and Illinois steps on it, about 8 evaluations per edge
  instead of 50 (2.5 to 3.3 times faster on NumPy, equally accurate); edges without a sign
  change of the level set are bisected as before. A level-set crossing is the root itself, not
  the end of the last bisection interval inside the object.
- The sampled paths of `cut_edges`/`intersect_segment` and `crossings` refine on the level set
  when there is one; `crossings` then takes its normal from the level-set gradient, as does
  `surface_normal`. `signed_distance` returns an exact level set directly, and `to_mesh`
  contours the level set of any shape that has one (it did so for balls and boxes only).
- The `FunctionObject3D` example is a rounded cube; use `Torus` for a torus.

### Removed

- Multipatch geometry: the `simgeometry.multipatch` module and the multipatch support in
  `simgeometry.grid` (`MultiPatchGridComposite`). The tutorial is now "Parallel grids".

### Fixed

- `Polygon.contains_points` with the compiled kernels accepts vertices that are not
  C-ordered (a transposed `(2, n)` array).
- `Aperture2D` documents its `axis` as `0` or `1` and its vertices as `(axial, radial)`.
- `Polygon.contains_points` with the compiled kernels returns an empty mask for no points
  instead of raising (this also broke `intersect_segment` for segments missing a polygon).

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
