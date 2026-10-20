# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

- One out-of-range rule for owners: per axis, positions below the lower bound belong to the
  first block and positions at or above the upper edge to the last (the block index is clipped).
  `owner_rank_for_position` now calls `owner_ranks`; positions below the lower bound used to go
  to the last block.
- `Aperture2D(axis=...)` now sets the beam axis: `axis=1` builds the plates with the beam
  along y. Before, `axis` was ignored and the beam always ran along x, as it still does
  for `axis=0`.
- 2D shapes have no `get_vertices()` any more; use `outlines()`, which returns `(2, n)`
  arrays. `PointList` and `Point` are gone.

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
