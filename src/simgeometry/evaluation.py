"""Labelled coordinate sampling, with no dependency on Picnic fields."""

from __future__ import annotations

import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.operations import mask


def sampling_axes(shape, resolution=128, bounds=None):
    """Return finite axes over the requested bounds."""
    lower, upper = (
        bounds if bounds is not None else (shape.lower_bound, shape.upper_bound)
    )
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    counts = np.broadcast_to(resolution, (shape.ndim,))
    if (
        lower.shape != (shape.ndim,)
        or upper.shape != lower.shape
        or not np.all(np.isfinite([lower, upper]))
        or np.any(upper <= lower)
    ):
        raise ValueError("Sampling requires finite, increasing bounds for every axis")
    if np.any(counts < 2) or np.any(counts != counts.astype(int)):
        raise ValueError("Resolution must be an integer >= 2 for each axis")
    return [np.linspace(a, b, int(n)) for a, b, n in zip(lower, upper, counts)]


@sp.profile("simgeometry.evaluate", tags=_PROFILE_TAGS)
def evaluate(
    shape,
    *,
    x=None,
    y=None,
    z=None,
    grid=None,
    bounds=None,
    resolution=128,
    quantity="occupancy",
):
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("evaluate() requires simgeometry[evaluate]") from exc
    names = getattr(shape, "axis_names", tuple("xyz"[: shape.ndim]))
    requested = dict(zip("xyz", (x, y, z)))
    if any(
        value is not None and name not in names for name, value in requested.items()
    ):
        raise ValueError(f"Available coordinate axes are {names}")
    if grid is not None:
        if grid.ndim != shape.ndim:
            raise ValueError("Grid and geometry dimensions must match")
        if bounds is not None:
            raise ValueError("Use grid or bounds, not both")
        defaults = grid.get_grid_points()
    else:
        defaults = (
            sampling_axes(shape, resolution, bounds)
            if any(requested[n] is None for n in names)
            else [None] * shape.ndim
        )
    coordinates = []
    for name, default in zip(names, defaults):
        value = requested[name] if requested[name] is not None else default
        if isinstance(value, xr.DataArray):
            coord = value.astype(float)
        else:
            value = np.asarray(value, float)
            if value.ndim > 1:
                raise ValueError(
                    "Use DataArrays with shared dimensions for scattered points"
                )
            coord = xr.DataArray(value, dims=(name,) if value.ndim else ())
        if coord.size == 0 or not np.all(np.isfinite(coord.values)):
            raise ValueError("Coordinates must be finite and nonempty")
        coordinates.append(coord)
    coordinates = xr.align(*coordinates, join="exact", copy=False)
    broadcast = xr.broadcast(*coordinates)
    points = np.stack([c.values.ravel() for c in broadcast])
    if quantity == "occupancy":
        values = mask(shape, points)
    elif quantity == "labels":
        values = (
            shape.labels(points)
            if hasattr(shape, "labels")
            else mask(shape, points).astype(int)
        )
    elif quantity == "signed_distance":
        values = shape.signed_distance(points)
    else:
        raise ValueError("quantity must be occupancy, labels, or signed_distance")
    result = xr.DataArray(
        np.asarray(values).reshape(broadcast[0].shape),
        dims=broadcast[0].dims,
        name=quantity,
    )
    for name, coord in zip(names, coordinates):
        result = result.assign_coords({name: coord})
    if quantity == "labels":
        result.attrs["labels"] = {
            0: "background",
            **dict(enumerate(getattr(shape, "names", ("inside",)), 1)),
        }
    result.attrs["sampling"] = "direct geometry queries; host arrays; no MPI gather"
    return result
