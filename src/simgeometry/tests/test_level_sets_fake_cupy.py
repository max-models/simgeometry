"""Level sets, the new solids and the level-set crossing paths on the fake CuPy."""

from __future__ import annotations

import textwrap

import pytest

kernel_testing = pytest.importorskip("cunumpy.kernel_testing")

CODE = textwrap.dedent(
    """
    import cunumpy as xp
    import numpy as np

    from simgeometry import Circle, Cone, Cuboid, Cylinder, Rectangle, Sphere, Torus, Tube

    with xp.use_backend("cupy", strict=True):
        rng = np.random.default_rng(0)
        host = rng.uniform(-1, 1, (3, 500))
        points = xp.asarray(host)
        shapes = [
            Sphere((0, 0, 0), 0.7),
            Cylinder((0, 0, -0.5), (0.2, 0.1, 0.5), 0.4, fillet=0.1),
            Tube((0, 0, -0.5), (0, 0, 0.5), 0.3, 0.6),
            Cone((0, 0, -0.5), (0, 0, 0.5), 0.6, 0.0),
            Torus((0, 0, 0), 0.6, 0.2, axis=(1, 0, 1)),
            Sphere((0, 0, 0), 0.8) - Cylinder((0, 0, -1), (0, 0, 1), 0.3),
            Tube((0, 0, -0.5), (0, 0, 0.5), 0.3, 0.6).rotated(30, axis=(1, 0, 0)),
            Cuboid((0, 0, 0), (1.2, 0.8, 0.6), fillet=0.2),
        ]
        for shape in shapes:
            values = shape.level_set(points)
            inside = shape.contains_points(points)
            assert not isinstance(values, np.ndarray), type(values)
            values, inside = xp.to_numpy(values), xp.to_numpy(inside)
            clear = np.abs(values) > 1e-12
            assert np.array_equal((values < 0)[clear], inside[clear])
            start = xp.zeros((3, 4))
            end = xp.asarray(np.eye(3, 4) + 0.01)
            hits = shape.crossings(start, end)
            hit = xp.to_numpy(hits.hit)
            fraction = xp.to_numpy(hits.fraction)[hit]
            point = xp.to_numpy(hits.point)[:, hit]
            level = xp.to_numpy(shape.level_set(xp.asarray(point)))
            assert np.all(np.abs(level) < 1e-12), (shape, level)
            assert np.all((fraction > 0) & (fraction <= 1))
        rounded = Rectangle((0, 0), (1.2, 0.8), fillet=0.3)
        flat = xp.asarray(host[:2])
        values = rounded.level_set(flat)
        assert not isinstance(values, np.ndarray), type(values)
        inside = xp.to_numpy(rounded.contains_points(flat))
        assert np.array_equal(xp.to_numpy(values) < 0, inside)
        hits = rounded.crossings(xp.zeros((2, 3)), xp.asarray(np.eye(2, 3) + 0.01))
        point = hits.point[:, hits.hit]
        assert np.all(np.abs(xp.to_numpy(rounded.level_set(point))) < 1e-12)
        circle = Circle((0, 0), 0.5)
        assert abs(circle.signed_distance(np.array([1.0, 0.0])) - 0.5) < 1e-15
    print("ok")
    """
)


def test_level_sets_on_the_fake_cupy():
    result = kernel_testing.run_in_fake_cupy_subprocess(CODE, timeout=300)
    assert "ok" in result.stdout
