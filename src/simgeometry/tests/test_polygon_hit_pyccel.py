"""The shared Pyccel hit test (run in Python) against the NumPy version."""

import numpy as np
import pytest

from simgeometry.kernels.polygon_hit_pyccel import object_hit_time, polygon_bounds
from simgeometry.kernels.polygon_hits import object_hit_times


@pytest.mark.parametrize("inverted", [0, 1])
def test_the_shared_hit_test_is_the_numpy_one(inverted):
    rng = np.random.default_rng(3)
    vertices, offsets = [], [0]
    for _ in range(4):
        num = int(rng.integers(3, 7))
        centre, radius = rng.uniform(0.0, 4.0, 2), rng.uniform(0.3, 1.0)
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, num))
        vertices += list(centre + radius * np.c_[np.cos(angles), np.sin(angles)])
        offsets.append(offsets[-1] + num)
    vertices, offsets = np.array(vertices), np.array(offsets)
    lower, upper = np.empty((4, 2)), np.empty((4, 2))
    polygon_bounds(vertices, offsets, lower, upper)
    np.testing.assert_array_equal(
        lower[1], vertices[offsets[1] : offsets[2]].min(axis=0)
    )
    np.testing.assert_array_equal(upper[3], vertices[offsets[3] :].max(axis=0))
    start = rng.uniform(-0.5, 4.5, (2, 300))
    end = start + rng.normal(scale=0.5, size=(2, 300))
    sweep = rng.random(300) < 0.7
    expected = object_hit_times(vertices, offsets, 0, 4, inverted, *start, *end, sweep)
    got = [
        object_hit_time(
            vertices,
            offsets,
            0,
            4,
            inverted,
            *start[:, k],
            *end[:, k],
            int(sweep[k]),
            lower,
            upper,
        )
        for k in range(300)
    ]
    np.testing.assert_array_equal(got, expected)
