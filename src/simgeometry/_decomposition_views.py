"""Views of a domain decomposition: one box (3D) or rectangle (2D) per rank.

Used by :meth:`DomainDecomposition.show_3d` and :meth:`DomainDecomposition.show`.
Nothing here communicates; the boxes come from ``subdomain_boxes``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_LEGEND_RANKS = 16


def _rank_colors(size: int) -> list[tuple[float, float, float]]:
    """Return one distinct colour per rank: tab10 up to 10 ranks, else tab20/turbo."""
    from matplotlib import colormaps

    if size <= 10:
        cmap = colormaps["tab10"]
        return [tuple(cmap(i)[:3]) for i in range(size)]
    if size <= 20:
        cmap = colormaps["tab20"]
        return [tuple(cmap(i)[:3]) for i in range(size)]
    cmap = colormaps["turbo"]
    return [tuple(cmap(i / (size - 1))[:3]) for i in range(size)]


def _objects(shape: Any) -> dict[str, Any]:
    """Return the named objects of a collection, or the shape itself."""
    return dict(getattr(shape, "named_objects", None) or {"shape": shape})


def show_decomposition_3d(
    decomposition: Any,
    lower_bound: Any,
    upper_bound: Any,
    num_gridpoints: Any,
    *,
    shape: Any = None,
    opacity: float = 0.25,
    shape_opacity: float = 1.0,
    resolution: int = 48,
    legend: bool | None = None,
    plotter: Any = None,
    display: bool = True,
    save: Any = None,
) -> Any:
    """Draw the subdomains of a 3D decomposition with plasma-plots; return the plotter."""
    if decomposition.ndim != 3:
        raise ValueError("show_3d needs a three-dimensional decomposition")
    try:
        import pyvista as pv
        from plasma_plots.pyvista_plots import pyvista_surfaces
    except ImportError as exc:
        raise ImportError("Interactive 3D needs simgeometry[interactive]") from exc

    boxes = decomposition.subdomain_boxes(lower_bound, upper_bound, num_gridpoints)
    if legend is None:
        legend = len(boxes) <= _MAX_LEGEND_RANKS
    meshes = {}
    for rank, (lower, upper) in enumerate(boxes):
        box = pv.Box(
            bounds=(lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]),
            quads=True,
        )
        box.point_data["rank"] = np.full(box.n_points, rank)
        meshes[f"rank {rank}"] = box
    if plotter is None:
        plotter = pv.Plotter(off_screen=not display)
    pyvista_surfaces(
        meshes,
        colors=None if not legend else _rank_colors(len(boxes)),
        scalars=None if legend else "rank",
        cmap="turbo",
        opacity=opacity,
        show_edges=True,
        legend=legend,
        plotter=plotter,
    )
    if shape is not None:
        if shape.ndim != 3:
            raise ValueError("shape must be three-dimensional")
        bounds = (np.asarray(lower_bound, float), np.asarray(upper_bound, float))
        shapes = {
            name: obj.to_mesh(
                resolution=resolution,
                bounds=bounds if getattr(obj, "inverted", False) else None,
            )
            for name, obj in _objects(shape).items()
        }
        pyvista_surfaces(
            shapes,
            colors=["0.55"] * len(shapes),
            opacity=shape_opacity,
            legend=False,
            plotter=plotter,
        )
    plotter.show_bounds(
        bounds=np.array([lower_bound, upper_bound], dtype=float).T.ravel(),
        location="outer",
        xtitle="x",
        ytitle="y",
        ztitle="z",
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        fmt="%.2g",
    )
    # x to the right, y into the picture and z up, so rank 0 is in front.
    plotter.view_vector((1.0, -1.0, 1.0))
    plotter.reset_camera()
    plotter.camera.zoom(0.9)
    if legend and plotter.legend is not None:
        plotter.legend.SetPosition(0.82, 0.70)  # clear of the boxes
    if display or save:
        plotter.show(screenshot=str(save) if save else None, auto_close=False)
    return plotter


def show_decomposition_2d(
    decomposition: Any,
    lower_bound: Any,
    upper_bound: Any,
    num_gridpoints: Any,
    *,
    shape: Any = None,
    ax: Any = None,
    opacity: float = 0.35,
    labels: bool = True,
    display: bool = True,
    save: Any = None,
) -> Any:
    """Draw the subdomains of a 2D decomposition with Matplotlib; return the axes."""
    if decomposition.ndim != 2:
        raise ValueError("show needs a two-dimensional decomposition")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from simgeometry.visualization import _finish

    if ax is None:
        ax = plt.figure(layout="constrained").add_subplot()
    boxes = decomposition.subdomain_boxes(lower_bound, upper_bound, num_gridpoints)
    colors = _rank_colors(len(boxes))
    for rank, ((x0, y0), (x1, y1)) in enumerate(boxes):
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=(*colors[rank], opacity),
                edgecolor=colors[rank],
                linewidth=1.5,
            )
        )
        if labels:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                f"rank {rank}",
                ha="center",
                va="center",
                fontsize="small",
            )
    if shape is not None:
        if shape.ndim != 2:
            raise ValueError("shape must be two-dimensional")
        for obj in _objects(shape).values():
            closed = getattr(obj, "closed", True)
            for loop in obj.outlines():
                if closed:
                    loop = np.concatenate([loop, loop[:, :1]], axis=1)
                ax.plot(*loop, color="black", linewidth=1.2, solid_capstyle="round")
    ax.set_xlim(lower_bound[0], upper_bound[0])
    ax.set_ylim(lower_bound[1], upper_bound[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return _finish(ax, display, save)
