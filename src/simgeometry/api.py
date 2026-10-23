"""Shared inspection and construction methods for geometry objects.

Array queries use component-major coordinates, ``(ndim, n)``. Inspection,
meshing and labelled evaluation return host arrays; containment retains the
selected cunumpy backend. Optional dependencies are imported on demand.
"""

from __future__ import annotations

import copy


class GeometryAPI:
    """Convenience operations shared by shapes, sections and collections."""

    def evaluate(
        self,
        x=None,
        y=None,
        z=None,
        *,
        grid=None,
        bounds=None,
        resolution=128,
        quantity="occupancy",
    ):
        """Sample geometry as an xarray.DataArray (requires the ``evaluate`` extra).

        Scalars fix axes, sequences form a tensor product, and DataArrays with
        shared dimension names describe scattered points. Unspecified axes use
        grid nodes, or ``resolution`` samples across ``bounds``/the shape bounds.
        ``quantity`` is occupancy, labels (collections), signed_distance, or
        level_set (the analytic level set, or the signed distance without one).
        This is a local host operation, with no implicit MPI collectives.
        """
        from simgeometry.evaluation import evaluate

        return evaluate(
            self,
            x=x,
            y=y,
            z=z,
            grid=grid,
            bounds=bounds,
            resolution=resolution,
            quantity=quantity,
        )

    def level_set_field(self, *, resolution=64, bounds=None):
        """The level set on a 3D grid, laid out for plasma-plots' 3D views.

        ``shape.level_set_field().plasma.plot.isosurface(values=[0])`` draws the
        surface (requires the ``interactive`` extra and ``import plasma_plots``).
        """
        from simgeometry.evaluation import level_set_field

        return level_set_field(self, resolution=resolution, bounds=bounds)

    def slice(self, *, x=None, y=None, z=None, origin=None, normal=None):
        """Return a queryable section with local-to-world coordinate mappings.

        Fix exactly one coordinate, or give a 3D plane's origin and normal.
        Axis-aligned sections retain the remaining coordinate names; oblique
        sections use x/y as their local plane coordinates.
        """
        from simgeometry.operations import Section

        return Section(self, x=x, y=y, z=z, origin=origin, normal=normal)

    def show(self, **kwargs):
        """Display geometry, dispatching to 3D unless a slice is requested.

        Returns Matplotlib axes, or a PyVista plotter for ``backend='pyvista'``.
        Set ``display=False`` for scripts. ``save=path`` saves the figure.
        """
        from simgeometry.visualization import show

        return show(self, **kwargs)

    def show_3d(self, **kwargs):
        """Display a surface with optional wireframe, normals and clipping.

        ``backend='pyvista'`` enables an interactive clipping-plane widget via
        ``clip=True`` (requires the ``interactive`` extra). Matplotlib supports
        a static ``clip=(origin, normal)`` plane. Meshes are resolution-dependent.
        """
        from simgeometry.visualization import show_3d

        return show_3d(self, **kwargs)

    def plot(self, ax=None, **kwargs):
        """Draw without displaying and return axes for composition."""
        return self.show(ax=ax, display=False, **kwargs)

    def copy(self):
        """Return an independent deep copy."""
        return copy.deepcopy(self)

    def transformed(self, matrix, offset=None):
        """Return an affine-transformed snapshot; the source is unchanged."""
        from simgeometry.operations import TransformedShape

        return TransformedShape(self, matrix, offset)

    def translated(self, offset):
        """Return a translated snapshot (unlike legacy in-place translate)."""
        import numpy as np

        return self.transformed(np.eye(self.ndim), offset)

    def rotated(self, angle, *, axis=(0, 0, 1), origin=None):
        """Return a rotated snapshot; angle is in degrees, about origin (default 0)."""
        from simgeometry.operations import rotation

        return rotation(self, angle, axis=axis, origin=origin)

    def scaled(self, factors, *, origin=None):
        """Return a snapshot scaled by a nonzero scalar or per-axis factors."""
        import numpy as np

        factors = np.broadcast_to(factors, (self.ndim,)).astype(float)
        centre = np.zeros(self.ndim) if origin is None else np.asarray(origin, float)
        return self.transformed(np.diag(factors), centre * (1 - factors))

    def union(self, other):
        from simgeometry.operations import BooleanShape

        return BooleanShape(self, other, "union")

    def intersection(self, other):
        from simgeometry.operations import BooleanShape

        return BooleanShape(self, other, "intersection")

    def difference(self, other):
        from simgeometry.operations import BooleanShape

        return BooleanShape(self, other, "difference")

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def intersect_segment(self, start, end, *, samples=256, tolerance=1e-10):
        """Return ordered surface hits, including multiple crossings and tangencies.

        Primitive intersections are analytic. Generic membership functions use
        ``samples`` intervals and may miss features smaller than that sampling.
        Result.exact distinguishes these cases. Coincident surface segments do
        not have a discrete set of hits; their end contacts may be returned.
        """
        from simgeometry.queries import intersect_segment

        return intersect_segment(self, start, end, samples=samples, tolerance=tolerance)

    def crossings(self, start, end, *, samples=64, iterations=40):
        """First boundary crossing of every segment ``start[:, i] -> end[:, i]``.

        ``start`` and ``end`` are ``(ndim, N)``, such as particle positions before and
        after a step. Returns a :class:`~simgeometry.crossings.Crossings` with ``hit``,
        ``fraction``, ``point`` and ``normal`` (out of the region where ``contains`` is
        true) on the active array backend. Polygons, circles, spheres, boxes and their
        affine transforms are analytic; other shapes sample ``samples`` points per
        segment and bisect ``iterations`` times, or refine on their level set (with its
        gradient as the normal) if they have one.
        """
        from simgeometry.crossings import crossings

        return crossings(self, start, end, samples=samples, iterations=iterations)

    def closest_point(self, points, *, resolution=64, bounds=None):
        """Closest surface coordinates; primitives are exact, others use a mesh.

        Mesh fallback requires ``mesh`` extras. Its accuracy is controlled by
        resolution and may miss unresolved components. Points are (ndim, n),
        or one (ndim,) point; the returned array has the same layout.
        """
        from simgeometry.queries import closest_point

        return closest_point(self, points, resolution=resolution, bounds=bounds)

    def signed_distance(self, points, *, resolution=64, bounds=None):
        """Distance to the boundary, negative inside, positive outside.

        Exact for balls, boxes, simple polygons and every shape whose level set is
        exact (``level_set_exact``, such as cylinders, tubes, cones, tori and their
        rigid motions), which it returns directly; otherwise distance to the
        sampled surface mesh. A membership function is not itself a distance.
        """
        from simgeometry.queries import signed_distance

        return signed_distance(self, points, resolution=resolution, bounds=bounds)

    def surface_normal(self, points, *, scale=None):
        """Outward unit normals at boundary points; NaN where undefined.

        Primitive faces use analytic normals. Other geometries estimate the
        normal from nearby membership samples. At corners normals are not unique.
        """
        from simgeometry.queries import surface_normal

        return surface_normal(self, points, scale=scale)

    def to_mesh(self, *, resolution=64, bounds=None):
        """Return sampled contours (2D) or triangles (3D), with spacing metadata.

        Requires the ``mesh`` extra. Open contours/surfaces at the sampling bounds
        remain open; no artificial caps are added. Refine to resolve thin features.
        """
        from simgeometry.mesh import to_mesh

        return to_mesh(self, resolution=resolution, bounds=bounds)

    def export(self, path, **kwargs):
        """Export a sampled mesh as NPZ, OBJ, or STL (3D only)."""
        return self.to_mesh(**kwargs).save(path)

    def sample_interior(self, count, *, rng=None, bounds=None, max_draws=1_000_000):
        """Uniform rejection samples, returned as (ndim, count).

        Supply bounds for inverted/unbounded shapes. Raises if max_draws is
        exhausted; it never silently returns too few samples.
        """
        from simgeometry.sampling import sample_interior

        return sample_interior(self, count, rng=rng, bounds=bounds, max_draws=max_draws)

    def sample_surface(self, count, *, rng=None, resolution=64, bounds=None):
        """Uniform-by-length/area samples from the faceted boundary mesh."""
        from simgeometry.sampling import sample_surface

        return sample_surface(
            self, count, rng=rng, resolution=resolution, bounds=bounds
        )

    def cut_edges(self, grid, **kwargs):
        """Locate all detected grid-edge surface hits, without solver clamping."""
        from simgeometry.cut_edges import cut_edges

        return cut_edges(self, grid, **kwargs)

    def diagnose(self, grid, **kwargs):
        """Report sampled overlaps, unresolved components, and short cut distances."""
        from simgeometry.cut_edges import diagnose

        return diagnose(self, grid, **kwargs)

    def cell_fractions(self, grid, *, samples_per_axis=4):
        """Estimate solid cell-volume and face-area fractions by subcell sampling.

        Increase samples_per_axis to check convergence; unresolved thin features
        can be missed. This is not an exact finite-volume interface reconstruction.
        """
        from simgeometry.fractions import cell_fractions

        return cell_fractions(self, grid, samples_per_axis=samples_per_axis)

    def yee_fractions(self, grid, **kwargs):
        """Edge-length and face-area fractions of a Yee grid, and the surface's traces.

        Exact for planes; see :func:`simgeometry.fractions.yee_fractions`.
        """
        from simgeometry.fractions import yee_fractions

        return yee_fractions(self, grid, **kwargs)
