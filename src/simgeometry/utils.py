"""Utils helpers for geometry, grids, and simulation domains."""

from __future__ import annotations

from numpy import ndarray


def vertices_to_bounds(
    vertices: ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Convert a list of vertices to bounding box coordinates.

    Parameters:
    vertices (ndarray): Array of shape (N, 2) representing N vertices in 2D.

    Returns:
    Tuple[Tuple[float, float], Tuple[float, float]]: Lower and upper bounds as tuples.
    """
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    lower_bound = (float(xs.min()), float(ys.min()))
    upper_bound = (float(xs.max()), float(ys.max()))
    return lower_bound, upper_bound
