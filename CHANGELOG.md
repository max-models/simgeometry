# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `Shape2D.outlines(max_spacing=None)`: the closed boundary loops of a 2D shape as
  counter-clockwise `(2, n)` arrays, with edges refined to at most `max_spacing`, or graded
  by a spacing function of position; also on transformed 2D shapes. Plotting draws shapes
  from it.

### Removed

- Multipatch geometry: the `simgeometry.multipatch` module and the multipatch support in
  `simgeometry.grid` (`MultiPatchGridComposite`). The tutorial is now "Parallel grids".

### Fixed

- `Polygon.contains_points` with the compiled kernels accepts vertices that are not
  C-ordered (a transposed `(2, n)` array).
- `Aperture2D` documents its `axis` as `0` or `1` and its vertices as `(axial, radial)`.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
