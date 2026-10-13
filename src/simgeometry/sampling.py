"""Reproducible interior and boundary sampling."""

import numpy as np

from simgeometry.operations import mask


def _count(count):
    if int(count) != count or count < 0:
        raise ValueError("count must be a nonnegative integer")
    return int(count)


def sample_interior(shape, count, *, rng=None, bounds=None, max_draws=1_000_000):
    count = _count(count)
    if shape.inverted and bounds is None:
        raise ValueError("Inverted/unbounded shapes require explicit sampling bounds")
    lower, upper = (
        bounds if bounds is not None else (shape.lower_bound, shape.upper_bound)
    )
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        lower.shape != (shape.ndim,)
        or upper.shape != lower.shape
        or not np.all(np.isfinite([lower, upper]))
        or np.any(upper <= lower)
    ):
        raise ValueError("Sampling bounds must be finite and increasing")
    if max_draws < count:
        raise ValueError("max_draws must be at least count")
    rng = np.random.default_rng(rng)
    accepted, remaining, drawn = [], count, 0
    while remaining and drawn < max_draws:
        n = min(max(remaining * 2, 256), max_draws - drawn)
        p = rng.uniform(lower[:, None], upper[:, None], (shape.ndim, n))
        p = p[:, mask(shape, p)][:, :remaining]
        accepted.append(p)
        remaining -= p.shape[1]
        drawn += n
    if remaining:
        raise ValueError(
            f"Only {count - remaining}/{count} samples after {drawn} draws; check bounds or increase max_draws"
        )
    return np.concatenate(accepted, axis=1) if count else np.empty((shape.ndim, 0))


def sample_surface(shape, count, *, rng=None, resolution=64, bounds=None):
    count = _count(count)
    if count == 0:
        return np.empty((shape.ndim, 0))
    if shape.ndim == 1:
        from simgeometry.queries import boundary_points_1d

        points = boundary_points_1d(shape, resolution=resolution, bounds=bounds)
        return points[:, np.random.default_rng(rng).integers(0, points.shape[1], count)]
    mesh = shape.to_mesh(resolution=resolution, bounds=bounds)
    if not len(mesh.cells):
        raise ValueError("No surface resolved; enlarge bounds or increase resolution")
    faces = mesh.vertices[mesh.cells]
    if shape.ndim == 2:
        weights = np.linalg.norm(faces[:, 1] - faces[:, 0], axis=1)
    else:
        weights = (
            np.linalg.norm(
                np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
            )
            / 2
        )
    rng = np.random.default_rng(rng)
    faces = faces[rng.choice(len(faces), size=count, p=weights / weights.sum())]
    if shape.ndim == 2:
        t = rng.random((count, 1))
        return ((1 - t) * faces[:, 0] + t * faces[:, 1]).T
    r, s = np.sqrt(rng.random((count, 1))), rng.random((count, 1))
    return ((1 - r) * faces[:, 0] + r * (1 - s) * faces[:, 1] + r * s * faces[:, 2]).T
