"""Optional 1D/2D inspection and 3D surface viewers.

Matplotlib is the default backend. The PyVista backend draws the surfaces and point
sets with plasma-plots (``pyvista_surfaces``, ``pyvista_points``) and adds interactive
clipping widgets.
Plots are sampled views. Increase resolution before interpreting small features.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def _finish(ax, display, save):
    import matplotlib.pyplot as plt

    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if display:
        plt.show()
    return ax


def show(
    shape,
    *,
    x=None,
    y=None,
    z=None,
    ax=None,
    resolution=None,
    bounds=None,
    grid=None,
    nodes=False,
    labels=True,
    normals=False,
    show_bounds=False,
    color=None,
    color_by="name",
    opacity=0.65,
    display=True,
    save=None,
    **kwargs,
):
    """Plot occupancy or a slice; dimensions and coordinates are labelled."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    selectors = {k: v for k, v in zip("xyz", (x, y, z)) if v is not None}
    if resolution is None:
        resolution = 48 if shape.ndim == 3 and not selectors else 128
    if shape.ndim == 3 and not selectors:
        return show_3d(
            shape,
            ax=ax,
            resolution=resolution,
            bounds=bounds,
            grid=grid,
            nodes=nodes,
            labels=labels,
            normals=normals,
            show_bounds=show_bounds,
            color=color,
            color_by=color_by,
            opacity=opacity,
            display=display,
            save=save,
            **kwargs,
        )
    if color_by not in (None, "name"):
        raise ValueError("color_by must be 'name' or None")
    if grid is True:
        raise ValueError("Pass a Grid object, or use domain.show(grid=True)")
    if grid is not None and grid.ndim != shape.ndim:
        raise ValueError("Grid and geometry dimensions must match")
    grid_axes = list(grid.get_grid_points()) if grid is not None else None
    if selectors:
        if len(selectors) != 1:
            raise ValueError("Select one coordinate for a geometry slice")
        source_names = getattr(shape, "axis_names", tuple("xyz"[: shape.ndim]))
        name = next(iter(selectors))
        if name not in source_names:
            raise ValueError(f"Available slice axes are {source_names}")
        axis = source_names.index(name)
        shape = shape.slice(**selectors)
        if grid_axes is not None:
            grid_axes.pop(axis)
        if bounds is not None:
            bounds = tuple(np.delete(v, axis) for v in bounds)
    names = getattr(shape, "axis_names", tuple("xyz"[: shape.ndim]))
    coordinates = dict(zip(names, grid_axes)) if grid_axes is not None else {}
    quantity = "labels" if hasattr(shape, "labels") else "occupancy"
    values = shape.evaluate(
        **coordinates, resolution=resolution, bounds=bounds, quantity=quantity
    )
    ax = ax or plt.subplots()[1]
    if shape.ndim == 1:
        ax.step(
            values.coords[names[0]], values.values, where="mid", color=color, **kwargs
        )
        ax.set_ylabel(quantity)
    else:
        xs, ys = (values.coords[n].values for n in names)
        object_names = getattr(shape, "names", ("geometry",))
        styles = getattr(
            shape, "styles", getattr(getattr(shape, "source", None), "styles", {})
        )
        colors = [
            styles.get(name, {}).get(
                "color",
                styles.get(name, {}).get(
                    "facecolor", color or (f"C{i % 10}" if color_by == "name" else "C0")
                ),
            )
            for i, name in enumerate(object_names)
        ]
        cmap = ListedColormap(colors or [color or "C0"])
        data = np.ma.masked_where(values.values == 0, values.values.astype(float) - 1)
        style = {
            "cmap": cmap,
            "vmin": -0.5,
            "vmax": max(len(colors) - 0.5, 0.5),
            "alpha": opacity,
            **kwargs,
        }
        ax.pcolormesh(xs, ys, data.T, shading="nearest", **style)
        if labels and len(object_names):
            ax.legend(
                handles=[Patch(color=c, label=n) for n, c in zip(object_names, colors)]
            )
        if grid_axes is not None:
            for value in xs:
                ax.axvline(value, color="gray", linewidth=0.4, alpha=0.5)
            for value in ys:
                ax.axhline(value, color="gray", linewidth=0.4, alpha=0.5)
        if nodes:
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            ax.scatter(
                xx.ravel(),
                yy.ravel(),
                c=(values.values != 0).ravel(),
                cmap="coolwarm",
                s=5,
            )
        if normals:
            mesh = shape.to_mesh(resolution=resolution, bounds=bounds)
            stride = max(1, len(mesh.cells) // 48)
            ax.quiver(*mesh.centres[::stride].T, *mesh.normals[::stride].T, angles="xy")
        if show_bounds:
            lower, upper = bounds or (shape.lower_bound, shape.upper_bound)
            ax.plot(
                [lower[0], upper[0], upper[0], lower[0], lower[0]],
                [lower[1], lower[1], upper[1], upper[1], lower[1]],
                color="black",
            )
        ax.set_aspect("equal")
        ax.set_ylabel(names[1])
    ax.set_xlabel(names[0])
    if selectors:
        ax.set_title(", ".join(f"{k} = {v:g}" for k, v in selectors.items()))
    return _finish(ax, display, save)


def _clip_triangles(vertices, faces, origin, normal):
    """Clip each triangle against a half-space, without adding artificial caps."""
    origin, normal = np.asarray(origin, float), np.asarray(normal, float)
    if (
        origin.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite([origin, normal]))
        or np.linalg.norm(normal) == 0
    ):
        raise ValueError("Clip plane needs finite 3D origin and nonzero normal")
    triangles = []
    for triangle in vertices[faces]:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            da, db = np.dot(a - origin, normal), np.dot(b - origin, normal)
            if da >= 0:
                polygon.append(a)
            if (da >= 0) != (db >= 0):
                polygon.append(a + da / (da - db) * (b - a))
        for i in range(1, len(polygon) - 1):
            triangles.append([polygon[0], polygon[i], polygon[i + 1]])
    vertices = np.asarray(triangles).reshape(-1, 3)
    return vertices, np.arange(len(vertices)).reshape(-1, 3)


def _box_edges(lower, upper):
    vertices = np.array(list(product(*zip(lower, upper))))
    return np.array(
        [
            [a, b]
            for i, a in enumerate(vertices)
            for b in vertices[i + 1 :]
            if np.count_nonzero(a != b) == 1
        ]
    )


_POINT_KEYS = ("points", "vectors", "color", "size")


def _as_points(points, label):
    """``(n, 3)`` host points from simgeometry's ``(3, n)`` layout or ``(n, 3)``."""
    import cunumpy as xp

    array = np.asarray(xp.to_numpy(points), dtype=float)
    if array.ndim == 1 and array.size == 3:
        array = array[:, None]
    if array.ndim == 2 and array.shape[0] == 3:
        return array.T
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    raise ValueError(
        f"points {label!r} must have shape (3, n) or (n, 3); got {array.shape}"
    )


def _as_vectors(vectors, count, label):
    """``(count, 3)`` host vectors from ``(3, count)`` or ``(count, 3)``."""
    import cunumpy as xp

    array = np.asarray(xp.to_numpy(vectors), dtype=float)
    if array.shape == (3, count):
        return array.T
    if array.shape == (count, 3):
        return array
    raise ValueError(
        f"vectors of {label!r} must have shape (3, {count}) or ({count}, 3); "
        f"got {array.shape}"
    )


def _point_sets(points, first_color):
    """``show_3d``'s ``points`` as ``{label: {"points", "vectors", "color", "size"}}``
    with ``(n, 3)`` arrays and a colour for each set, and whether they are labelled."""
    named = isinstance(points, dict)
    entries = points if named else {"points": points}
    sets = {}
    for i, (label, value) in enumerate(entries.items()):
        entry = dict(value) if isinstance(value, dict) else {"points": value}
        unknown = sorted(set(entry) - set(_POINT_KEYS))
        if unknown or "points" not in entry:
            raise ValueError(
                f"points {label!r}: give an array or a dict with 'points' and "
                f"optionally 'vectors', 'color', 'size'; got keys {sorted(entry)}"
            )
        coordinates = _as_points(entry["points"], label)
        vectors = entry.get("vectors")
        sets[str(label)] = {
            "points": coordinates,
            "vectors": None
            if vectors is None
            else _as_vectors(vectors, len(coordinates), label),
            "color": entry.get("color") or f"C{(first_color + i) % 10}",
            "size": entry.get("size"),
        }
    return sets, named


def show_3d(
    shape,
    *,
    ax=None,
    resolution=48,
    bounds=None,
    opacity=0.65,
    wireframe=False,
    normals=False,
    labels=True,
    color=None,
    color_by="name",
    grid=None,
    nodes=False,
    show_bounds=True,
    clip=None,
    backend="matplotlib",
    display=True,
    save=None,
    t=0.0,
    points=None,
    **kwargs,
):
    """Render surfaces; return axes or an optional PyVista Plotter.

    clip=(origin, normal) retains the positive half-space, without caps.
    clip=True enables a PyVista clipping widget. PyVista callers own and should
    close the returned plotter. color_by="voltage" (PyVista) colors the
    conductors of a collection by their voltage at time ``t``; objects without
    a voltage are grey. No MPI operations are performed.

    ``points`` overlays point sets, e.g. samples, grid nodes or crossings: an
    array in simgeometry's ``(3, n)`` layout (or ``(n, 3)``; a ``(3, 3)`` array
    is read as ``(3, n)``), or a dict of label to array, or of label to a dict
    with ``"points"`` and optionally ``"vectors"`` (one per point, e.g. normals,
    drawn as arrows; the longest is 5% of the view), ``"color"`` and ``"size"``
    (pixels in PyVista, points in Matplotlib). Each set takes the next colour
    after the objects' and, when labelled, a legend entry beside them (PyVista:
    plasma-plots' ``pyvista_points``).
    """
    if shape.ndim != 3:
        raise ValueError("show_3d requires three-dimensional geometry")
    if backend not in ("matplotlib", "pyvista"):
        raise ValueError("backend must be matplotlib or pyvista")
    if color_by not in (None, "name", "voltage"):
        raise ValueError("color_by must be 'name', 'voltage' or None")
    if color_by == "voltage" and backend != "pyvista":
        raise ValueError("color_by='voltage' needs backend='pyvista'")
    if color_by == "voltage" and not getattr(shape, "conductors", ()):
        raise ValueError("color_by='voltage' needs a collection with voltages")
    if grid is True:
        raise ValueError("Pass a Grid object, or use domain.show_3d(grid=True)")
    if grid is not None and grid.ndim != 3:
        raise ValueError("A 3D view requires a 3D grid")
    objects = getattr(shape, "named_objects", {"geometry": shape})
    styles = getattr(shape, "styles", {})
    lower, upper = bounds or (shape.lower_bound, shape.upper_bound)
    point_sets, named_points = (
        _point_sets(points, len(objects)) if points is not None else ({}, False)
    )
    vector_scale = float(np.max(np.asarray(upper) - lower)) * 0.05
    if backend == "pyvista":
        try:
            from importlib import import_module

            pv = import_module("pyvista")
        except ImportError as exc:
            raise ImportError("Interactive 3D needs simgeometry[interactive]") from exc
        plotter = ax if ax is not None else pv.Plotter(off_screen=not display)
        voltages = shape.voltages_at(t) if color_by == "voltage" else {}
        # plasma-plots draws the surfaces; the clipping widget, wireframes and extra
        # add_mesh options need PyVista's own add_mesh.
        surfaces, surface_colors, surface_opacity = {}, {}, {}
        use_surfaces = clip is not True and not wireframe and not kwargs
    else:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

        if clip is True:
            raise ValueError(
                "Use backend='pyvista' for interactive clipping, or supply a plane"
            )
        ax = ax if ax is not None else plt.figure().add_subplot(projection="3d")
        if not hasattr(ax, "set_zlabel"):
            raise ValueError("Pass 3D axes to show_3d")
        handles = []
    for i, (name, obj) in enumerate(objects.items()):
        mesh = obj.to_mesh(
            resolution=resolution,
            bounds=bounds
            if bounds is not None
            else ((lower, upper) if obj.inverted else None),
        )
        vertices, faces = mesh.vertices, mesh.cells
        if clip is not None and clip is not False and clip is not True:
            vertices, faces = _clip_triangles(vertices, faces, *clip)
        if not len(faces):
            continue
        style = dict(styles.get(name, {}))
        chosen = style.pop(
            "color",
            style.pop(
                "facecolor", color or (f"C{i % 10}" if color_by == "name" else "C0")
            ),
        )
        alpha = style.pop("opacity", style.pop("alpha", opacity))
        if backend == "pyvista":
            # Resolve Matplotlib cycle colors to RGB before passing to VTK.
            from matplotlib.colors import to_rgb

            data = pv.PolyData(
                vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel()
            )
            if color_by == "voltage":
                data.point_data["voltage"] = np.full(
                    data.n_points, voltages.get(name, np.nan)
                )
            if use_surfaces and not style:
                surfaces[name] = data
                surface_colors[name] = to_rgb(chosen)
                surface_opacity[name] = alpha
            else:
                add = plotter.add_mesh_clip_plane if clip is True else plotter.add_mesh
                add(
                    data,
                    color=None if color_by == "voltage" else to_rgb(chosen),
                    scalars="voltage" if color_by == "voltage" else None,
                    opacity=alpha,
                    style="wireframe" if wireframe else "surface",
                    label=name if labels else None,
                    **style,
                    **kwargs,
                )
            if normals:
                stride = max(1, len(mesh.cells) // 64)
                plotter.add_arrows(
                    mesh.centres[::stride],
                    mesh.normals[::stride],
                    mag=float(np.max(np.asarray(upper) - lower)) * 0.05,
                )
        else:
            artist = Poly3DCollection(
                vertices[faces],
                facecolors="none" if wireframe else to_rgba(chosen, alpha),
                edgecolors=chosen if wireframe else "none",
                **style,
                **kwargs,
            )
            ax.add_collection3d(artist)
            handles.append(Patch(color=chosen, label=name))
            if normals:
                triangles = vertices[faces][:: max(1, len(faces) // 64)]
                points = triangles.mean(axis=1).T
                normal = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                )
                normal /= np.linalg.norm(normal, axis=1)[:, None]
                ax.quiver(
                    *points,
                    *normal.T,
                    length=float(np.max(np.asarray(upper) - lower)) * 0.05,
                )
    if backend == "pyvista":
        if surfaces:
            from plasma_plots.pyvista_plots import pyvista_surfaces

            pyvista_surfaces(
                surfaces,
                colors=surface_colors,
                opacity=surface_opacity,
                scalars="voltage" if color_by == "voltage" else None,
                legend=labels,
                plotter=plotter,
            )
        if show_bounds:
            plotter.show_bounds(bounds=np.array([lower, upper]).T.ravel())
        if grid is not None:
            edges = _grid_lines_3d(grid)
            for segment in edges:
                plotter.add_lines(segment, color="gray", width=1)
            if nodes:
                plotter.add_points(grid.node_points().T, point_size=4)
        if labels and len(surfaces) < len(objects) and color_by != "voltage":
            plotter.add_legend()
        if point_sets:
            from plasma_plots.pyvista_plots import pyvista_points

            entries = point_sets.values()
            pyvista_points(
                {label: entry["points"] for label, entry in point_sets.items()}
                if named_points
                else point_sets["points"]["points"],
                colors=[entry["color"] for entry in entries],
                size=[
                    6 if entry["size"] is None else entry["size"] for entry in entries
                ],
                vectors={
                    label: entry["vectors"]
                    for label, entry in point_sets.items()
                    if entry["vectors"] is not None
                }
                if named_points
                else point_sets["points"]["vectors"],
                vector_scale=vector_scale,
                legend=labels,
                plotter=plotter,
            )
        if display or save:
            plotter.show(screenshot=str(save) if save else None, auto_close=False)
        return plotter
    if show_bounds:
        ax.add_collection3d(
            Line3DCollection(_box_edges(lower, upper), colors="black", linewidths=0.6)
        )
    if grid is not None:
        ax.add_collection3d(
            Line3DCollection(
                _grid_lines_3d(grid), colors="gray", linewidths=0.3, alpha=0.4
            )
        )
        if nodes:
            ax.scatter(*grid.node_points(), s=2, alpha=0.25)
    peak = max(
        (
            float(np.linalg.norm(entry["vectors"], axis=1).max(initial=0.0))
            for entry in point_sets.values()
            if entry["vectors"] is not None
        ),
        default=0.0,
    )
    for label, entry in point_sets.items():
        size = {} if entry["size"] is None else {"s": entry["size"] ** 2}
        artist = ax.scatter(
            *entry["points"].T,
            color=entry["color"],
            label=label if named_points else None,
            depthshade=False,
            **size,
        )
        if named_points:
            handles.append(artist)
        if entry["vectors"] is not None and peak > 0:
            ax.quiver(
                *entry["points"].T,
                *entry["vectors"].T,
                length=vector_scale / peak,
                color=entry["color"],
            )
    for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lower, upper):
        setter(a, b)
    from matplotlib.ticker import MaxNLocator

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_box_aspect(np.asarray(upper) - lower)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if labels and handles:
        ax.legend(handles=handles)
    return _finish(ax, display, save)


def _grid_lines_3d(grid):
    axes = grid.get_grid_points()
    segments = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for values in product(*(axes[i] for i in other)):
            a, b = np.zeros(3), np.zeros(3)
            a[other], b[other] = values, values
            a[axis], b[axis] = grid.lower_bound[axis], grid.upper_bound[axis]
            segments.append([a, b])
    return np.asarray(segments)
