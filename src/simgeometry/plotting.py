"""Optional Matplotlib drawing for geometric shapes and domains."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from simgeometry.shapes import Shape2D
from simgeometry.simulation_domain import SimulationDomain

# Draw order: grid lines, objects, then the domain outline.
ZORDER_GRID = 1.5
ZORDER_OBJECT = 3.0
ZORDER_FRAME = 3.5


def _axes(ax: Any | None) -> Any:
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    return ax


def _region(
    shape: Shape2D,
    bounds: tuple[Sequence[float], Sequence[float]] | None,
) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` of the region to draw ``shape`` in.

    An inverted shape covers everything outside its geometry, so it is drawn
    over ``bounds`` (the simulation domain) when given; otherwise, and for a
    regular shape, over the shape's bounding box.
    """
    lower, upper = (
        bounds
        if bounds is not None and shape.inverted
        else (shape.lower_bound, shape.upper_bound)
    )
    x_min, y_min = (float(v) for v in lower)
    x_max, y_max = (float(v) for v in upper)
    return x_min, y_min, x_max, y_max


def _signed_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _draw_function_object(
    shape: Any,
    ax: Any,
    resolution: int = 128,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    **kwargs: Any,
) -> Any:
    x_min, y_min, x_max, y_max = _region(shape, bounds)
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    mask = np.asarray([[shape.contains((x, y)) for x in xs] for y in ys], dtype=bool)
    if not mask.any():
        return ax
    style = {
        "origin": "lower",
        "extent": [x_min, x_max, y_min, y_max],
        "cmap": "Greys",
        "vmin": 0.0,
        "vmax": 1.0,
        "aspect": "auto",
        "zorder": ZORDER_OBJECT,
        **kwargs,
    }
    ax.imshow(np.ma.masked_where(~mask, mask.astype(float)), **style)
    return ax


def draw_shape(
    shape: Shape2D,
    ax: Any | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a 2D shape as filled polygons (a mask for ``FunctionObject2D``).

    An inverted shape is drawn as the region ``bounds = (lower, upper)`` (the
    simulation domain; the shape's bounding box by default) with its geometry
    cut out.
    """
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path

    from simgeometry.objects.function_object import FunctionObject2D

    ax = _axes(ax)
    if isinstance(shape, FunctionObject2D):
        return _draw_function_object(shape, ax, bounds=bounds, **kwargs)

    style = {
        "facecolor": "gray",
        "edgecolor": "black",
        "zorder": ZORDER_OBJECT,
        **kwargs,
    }
    polygons = [loop.T for loop in shape.outlines()]
    if not shape.inverted:
        for vertices in polygons:
            ax.add_patch(Polygon(vertices, closed=True, **style))
        ax.autoscale_view()
        return ax

    # Fill the region counter-clockwise and the geometry clockwise, so that
    # the geometry is a hole under matplotlib's nonzero winding rule.
    x_min, y_min, x_max, y_max = _region(shape, bounds)
    outer = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    holes = [v if _signed_area(v) < 0.0 else v[::-1] for v in polygons if len(v) >= 3]
    path = Path.make_compound_path(
        *(Path(np.vstack([ring, ring[:1]]), closed=True) for ring in (outer, *holes))
    )
    ax.add_patch(PathPatch(path, **style))
    ax.autoscale_view()
    return ax


def draw_domain(
    domain: SimulationDomain,
    ax: Any | None = None,
    show_grid: bool = False,
    objects: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw the domain outline, optionally its grid lines, and its objects."""
    from matplotlib.collections import LineCollection

    ax = _axes(ax)
    style = {"color": "black", "linewidth": 1.0, "zorder": ZORDER_FRAME, **kwargs}
    lower = [float(bound) for bound in domain.lower_bound]
    upper = [float(bound) for bound in domain.upper_bound]

    if domain.ndim == 1:
        ax.axvline(lower[0], **style)
        ax.axvline(upper[0], **style)
        return ax
    if domain.ndim != 2:
        raise ValueError(
            f"draw_domain draws 1D and 2D domains, got ndim={domain.ndim}; "
            "use Plotter3D for 3D geometry"
        )

    (x0, y0), (x1, y1) = lower, upper
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **style)

    if show_grid:
        xs, ys = (np.asarray(points) for points in domain.grid.get_grid_points())
        segments = [[(x, y0), (x, y1)] for x in xs[1:-1]]
        segments += [[(x0, y), (x1, y)] for y in ys[1:-1]]
        ax.add_collection(
            LineCollection(
                segments,
                colors="gray",
                linewidths=0.5,
                alpha=0.6,
                zorder=ZORDER_GRID,
            )
        )

    if objects:
        for shape in domain.objects:
            if isinstance(shape, Shape2D):
                draw_shape(shape, ax=ax, bounds=(lower, upper))
    return ax
