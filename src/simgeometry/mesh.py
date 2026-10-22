"""Sampled surface meshes and portable export formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.evaluation import sampling_axes
from simgeometry.operations import mask


@dataclass(frozen=True)
class SurfaceMesh:
    """Vertices (n, ndim), segments/triangles, and sampling spacing.

    Empty cells mean no boundary was resolved in the requested region.
    ``spacing`` describes the sampling, not a guaranteed error bound. Surfaces
    touching the sampling bounds can be open. Coordinates are physical.
    """

    vertices: np.ndarray
    cells: np.ndarray
    spacing: tuple[float, ...]

    @property
    def centres(self):
        """Segment/triangle centres, shape (ncells, ndim)."""
        return self.vertices[self.cells].mean(axis=1)

    @property
    def normals(self):
        """Outward unit normals of mesh segments/triangles (faceted approximation)."""
        cells = self.vertices[self.cells]
        edge = cells[:, 1] - cells[:, 0]
        normal = (
            np.column_stack((edge[:, 1], -edge[:, 0]))
            if self.vertices.shape[1] == 2
            else np.cross(edge, cells[:, 2] - cells[:, 0])
        )
        length = np.linalg.norm(normal, axis=1)
        return normal / np.where(length > 0, length, np.nan)[:, None]

    def save(self, path):
        """Write .npz, .obj (segments or triangles), or ASCII .stl (3D)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            np.savez(
                path, vertices=self.vertices, cells=self.cells, spacing=self.spacing
            )
        elif suffix == ".obj":
            vertices = np.pad(self.vertices, ((0, 0), (0, 3 - self.vertices.shape[1])))
            lines = ["# simgeometry sampled surface\n"]
            lines += ["v " + " ".join(f"{v:.17g}" for v in p) + "\n" for p in vertices]
            prefix = "f" if self.cells.shape[1] == 3 else "l"
            lines += [
                prefix + " " + " ".join(str(int(i) + 1) for i in cell) + "\n"
                for cell in self.cells
            ]
            path.write_text("".join(lines))
        elif suffix == ".stl" and self.vertices.shape[1] == 3:
            lines = ["solid simgeometry\n"]
            for triangle in self.vertices[self.cells]:
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                length = np.linalg.norm(normal)
                normal = normal / length if length else normal
                lines.append(
                    "facet normal "
                    + " ".join(f"{v:.17g}" for v in normal)
                    + "\nouter loop\n"
                )
                lines += [
                    "vertex " + " ".join(f"{v:.17g}" for v in p) + "\n"
                    for p in triangle
                ]
                lines.append("endloop\nendfacet\n")
            lines.append("endsolid simgeometry\n")
            path.write_text("".join(lines))
        else:
            raise ValueError("Use .npz, .obj, or .stl for 3D triangles")
        return path


@sp.profile("simgeometry.to_mesh", tags=_PROFILE_TAGS)
def to_mesh(shape, *, resolution=64, bounds=None):
    if shape.ndim not in (2, 3):
        raise ValueError("Surface meshing requires 2D or 3D geometry")
    try:
        from skimage.measure import find_contours, marching_cubes
    except ImportError as exc:
        raise ImportError("Surface meshing requires simgeometry[mesh]") from exc
    from simgeometry.queries import level_values

    if bounds is None:
        lower, upper = np.asarray(shape.lower_bound), np.asarray(shape.upper_bound)
        pad = (upper - lower) * 0.05
        bounds = (lower - pad, upper + pad)
    axes = sampling_axes(shape, resolution, bounds)
    dimensions = tuple(len(a) for a in axes)
    spacing = tuple(float(a[1] - a[0]) for a in axes)
    points = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])
    # A level set (continuous, positive inside once negated) places the surface
    # between samples; a membership mask has an uncertainty of about a sampling cell.
    if getattr(shape, "has_level_set", False):
        values = -level_values(shape, points)
    else:
        values = mask(shape, points).astype(float) - 0.5
    values = values.reshape(dimensions)
    if not (values.min() < 0 < values.max()):
        return SurfaceMesh(
            np.empty((0, shape.ndim)), np.empty((0, shape.ndim), dtype=int), spacing
        )
    if shape.ndim == 2:
        vertices, cells = [], []
        offset = 0
        for contour in find_contours(values, 0, positive_orientation="high"):
            vertices.extend(
                contour * np.asarray(spacing) + np.array([a[0] for a in axes])
            )
            cells.extend((offset + i, offset + i + 1) for i in range(len(contour) - 1))
            offset += len(contour)
        return SurfaceMesh(
            np.asarray(vertices).reshape(-1, 2),
            np.asarray(cells, dtype=int).reshape(-1, 2),
            spacing,
        )
    vertices, faces, _, _ = marching_cubes(
        values,
        level=0,
        spacing=spacing,
        allow_degenerate=False,
        gradient_direction="ascent",
    )
    vertices += np.array([a[0] for a in axes])
    return SurfaceMesh(vertices, faces, spacing)
