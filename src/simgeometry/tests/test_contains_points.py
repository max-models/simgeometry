"""The vectorised ``contains_points`` of every object gives ``contains`` point by point.

The field solver finds the nodes inside an object with ``contains_points`` over
the whole grid (:meth:`Shape.node_mask`), so the two must agree exactly, also on
nodes that lie on a surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry.grid import Grid
from simgeometry.objects import (
    Aperture2D,
    Aperture3D,
    Circle,
    Cuboid,
    Line,
    Polygon,
    Rectangle,
    Sphere,
    Triangle,
)


def _shapes():
    return {
        "line": Line(midpoint=(0.5,), side_lengths=(0.25,)),
        "rectangle": Rectangle(midpoint=(0.5, 0.25), side_lengths=(0.25, 0.5)),
        "triangle": Triangle(vertices=((0.1, 0.1), (0.9, 0.2), (0.4, 0.8))),
        "polygon": Polygon(
            vertices=np.array([(0.25, 0.25), (0.75, 0.25), (0.5, 0.75), (0.3, 0.5)])
        ),
        "circle": Circle(midpoint=(0.5, 0.5), radius=0.25),
        "aperture2d": Aperture2D(
            axis=1, position=0.5, vertices=((0.25, 0.125), (0.5, 0.125), (0.5, 0.25))
        ),
        "cuboid": Cuboid(midpoint=(0.5, 0.5, 0.5), side_lengths=(0.25, 0.5, 0.75)),
        "sphere": Sphere(midpoint=(0.5, 0.5, 0.5), radius=0.25),
        "aperture3d": Aperture3D(
            axis="x",
            center_coords=(0.5, 0.5),
            vertices=((0.25, 0.25), (0.5, 0.125), (0.75, 0.25)),
            lower_bound=(0.25, 0.0, 0.0),
            upper_bound=(0.75, 1.0, 1.0),
        ),
    }


def _points(ndim, seed=0):
    rng = np.random.default_rng(seed)
    random = rng.uniform(-0.1, 1.1, (ndim, 3000))
    # nodes of a fine grid: many lie exactly on the surfaces above
    shape = (33, 17, 9)[:ndim] if ndim < 3 else (17, 17, 17)
    grid = Grid(
        ndim=ndim,
        lower_bound=(0.0,) * ndim,
        upper_bound=(1.0,) * ndim,
        num_gridpoints=shape,
    )
    return np.concatenate([random, grid.node_points()], axis=1), grid


@pytest.mark.parametrize("name", sorted(_shapes()))
@pytest.mark.parametrize("inverted", [False, True])
def test_contains_points_matches_contains(name, inverted):
    shape = _shapes()[name]
    shape._inverted = inverted
    points, grid = _points(shape.ndim)
    expected = np.array([bool(shape.contains(p)) for p in points.T])
    np.testing.assert_array_equal(np.asarray(shape.contains_points(points)), expected)
    # the grid nodes inside, as the field solver finds them
    nodes = grid.node_points()
    expected_nodes = [
        index for index, point in zip(grid.indices, nodes.T) if shape.contains(point)
    ]
    assert shape.get_indices_in_object(grid) == expected_nodes
