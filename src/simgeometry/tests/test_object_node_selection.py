"""Objects must select grid nodes by their physical position, not their index.

The field solver pins the nodes an object selects to its potential, while
particles are absorbed where ``contains`` is true.  Both must therefore agree
on where the object is, also when the domain does not start at the origin.
"""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry.grid import Grid
from simgeometry.objects import (
    Circle,
    Cuboid,
    FunctionObject2D,
    FunctionObject3D,
    Line,
    Polygon,
    Rectangle,
    Sphere,
    Triangle,
)


def _grid(lower: float, upper: float, num: int, ndim: int) -> Grid:
    return Grid(
        ndim=ndim,
        lower_bound=(lower,) * ndim,
        upper_bound=(upper,) * ndim,
        num_gridpoints=(num,) * ndim,
    )


def _objects_1d() -> list:
    return [Line(midpoint=(1.5,), side_lengths=(0.2,))]


def _objects_2d() -> list:
    return [
        Circle(midpoint=(1.5, 1.5), radius=0.2),
        Rectangle(midpoint=(1.5, 1.5), side_lengths=(0.3, 0.2)),
        Triangle(vertices=((1.2, 1.2), (1.8, 1.2), (1.5, 1.8))),
        Polygon(vertices=np.array([[1.2, 1.2], [1.7, 1.3], [1.6, 1.8], [1.25, 1.6]])),
        FunctionObject2D(
            func=lambda c: (c[0] - 1.5) ** 2 + (c[1] - 1.5) ** 2 <= 0.04,
            lower_bound=(1.3, 1.3),
            upper_bound=(1.7, 1.7),
        ),
    ]


def _objects_3d() -> list:
    return [
        Sphere(midpoint=(1.5, 1.5, 1.5), radius=0.3),
        Cuboid(midpoint=(1.5, 1.5, 1.5), side_lengths=(0.4, 0.3, 0.5)),
        FunctionObject3D(
            func=lambda c: np.sum((np.asarray(c) - 1.5) ** 2) <= 0.09,
            lower_bound=(1.2, 1.2, 1.2),
            upper_bound=(1.8, 1.8, 1.8),
        ),
    ]


CASES = (
    [(1, 41, obj) for obj in _objects_1d()]
    + [(2, 41, obj) for obj in _objects_2d()]
    + [(3, 21, obj) for obj in _objects_3d()]
)


@pytest.mark.parametrize(
    ("ndim", "num", "obj"),
    CASES,
    ids=[type(case[2]).__name__ for case in CASES],
)
def test_selected_nodes_are_those_inside_the_object(ndim, num, obj) -> None:
    """On the domain [1, 2] the selection equals ``contains`` at the node positions."""
    lower = 1.0
    grid = _grid(lower, lower + 1.0, num, ndim)

    selected = {tuple(np.atleast_1d(i)) for i in obj.get_indices_in_object(grid)}
    expected = {
        tuple(np.atleast_1d(index))
        for index in grid.indices
        if obj.contains(lower + np.atleast_1d(index) * np.asarray(grid.grid_spacing))
    }

    assert expected, "the test object should contain some nodes"
    assert selected == expected


def test_node_coordinates_include_the_lower_bound() -> None:
    """``Grid.node_coordinates`` is ``lower_bound + index * spacing``."""
    grid = Grid(
        ndim=2,
        lower_bound=(-1.0, 2.0),
        upper_bound=(1.0, 4.0),
        num_gridpoints=(11, 21),
    )

    assert grid.node_coordinates((0, 0)) == pytest.approx([-1.0, 2.0])
    assert grid.node_coordinates((10, 20)) == pytest.approx([1.0, 4.0])
    assert grid.node_coordinates((5, 10)) == pytest.approx([0.0, 3.0])


def test_node_coordinates_in_1d_accept_a_plain_integer() -> None:
    """1D grids index by integers, not tuples."""
    grid = Grid(ndim=1, lower_bound=(2.0,), upper_bound=(4.0,), num_gridpoints=(5,))

    assert grid.node_coordinates(0) == pytest.approx([2.0])
    assert grid.node_coordinates(4) == pytest.approx([4.0])
