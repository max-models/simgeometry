"""Compiled containment kernels agree with the public shape predicates."""

import numpy as np
import pytest

from simgeometry.kernels.contains_ball import kernel as contains_ball
from simgeometry.kernels.contains_box import kernel as contains_box
from simgeometry.kernels.contains_polygon_2d import kernel as contains_polygon_2d
from simgeometry.objects import Cuboid, Polygon, Sphere


@pytest.mark.parametrize("inverted", [False, True])
def test_containment_kernels_match_shapes(inverted: bool) -> None:
    points_3d = np.ascontiguousarray(
        [[0.0, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 0.5], [0.0, 0.5, 1.0, 0.5]]
    )
    points_2d = np.ascontiguousarray(points_3d[:2])
    ball = Sphere(midpoint=(0.5, 0.5, 0.5), radius=0.75, inverted=inverted)
    box = Cuboid(
        midpoint=(0.5, 0.5, 0.5), side_lengths=(1.0, 1.0, 1.0), inverted=inverted
    )
    polygon = Polygon(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)), inverted=inverted)

    cases = (
        (contains_ball, ball, points_3d, (np.array(ball.midpoint), ball.radius)),
        (
            contains_box,
            box,
            points_3d,
            (np.array(box.lower_bound), np.array(box.upper_bound)),
        ),
        (contains_polygon_2d, polygon, points_2d, (np.array(polygon._vertices),)),
    )
    for kernel, shape, points, arguments in cases:
        mask = np.empty(points.shape[1], dtype=bool)
        kernel(points, mask, *arguments, int(inverted))
        expected = [shape.contains(point) for point in points.T]
        np.testing.assert_array_equal(mask, expected)
