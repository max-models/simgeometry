"""Voltages on GeometryCollection: overrides, time dependence and Dirichlet values."""

from __future__ import annotations

import numpy as np
import pytest

from simgeometry import Circle, GeometryCollection, Grid, Rectangle, Tube


def _electrodes(**kwargs):
    return GeometryCollection(
        {
            "wall": Rectangle((0, 0), (4, 0.2)),
            "electrode": Circle((0, 1), 0.5),
            "dielectric": Circle((1, 1), 0.3),
        },
        **kwargs,
    )


def test_voltages_name_conductors_in_object_order():
    lens = _electrodes(voltages={"electrode": 1000, "wall": 0.0})
    assert lens.conductors == ("wall", "electrode")
    assert lens.voltages == {"wall": 0.0, "electrode": 1000.0}
    assert lens.voltages_at() == {"wall": 0.0, "electrode": 1000.0}
    assert _electrodes().conductors == ()


def test_invalid_voltages_raise():
    with pytest.raises(ValueError, match="unknown"):
        _electrodes(voltages={"anode": 1.0})
    with pytest.raises(TypeError):
        _electrodes(voltages={"wall": "0 V"})
    with pytest.raises(TypeError):
        _electrodes(voltages={"wall": True})
    with pytest.raises(ValueError, match="unknown"):
        _electrodes().with_voltages(anode=1.0)


def test_with_voltages_shares_the_geometry():
    lens = _electrodes(voltages={"electrode": 1000.0, "wall": 0.0}, styles={"wall": {}})
    scan = lens.with_voltages(electrode=500.0)
    assert scan is not lens
    assert scan.voltages == {"wall": 0.0, "electrode": 500.0}
    assert lens.voltages["electrode"] == 1000.0
    assert scan["electrode"] is lens["electrode"]
    assert scan.styles == lens.styles
    assert scan.lower_bound == lens.lower_bound
    assert scan.upper_bound == lens.upper_bound
    # a mapping for any name, None to remove a voltage
    other = lens.with_voltages({"dielectric": -5}, wall=None)
    assert other.conductors == ("electrode", "dielectric")


def test_time_dependent_voltages():
    lens = _electrodes(voltages={"wall": 0.0, "electrode": lambda t: 100.0 * np.sin(t)})
    assert lens.voltages_at(0.0)["electrode"] == 0.0
    assert lens.voltages_at(np.pi / 2)["electrode"] == pytest.approx(100.0)
    values = lens.dirichlet_values(np.array([[0.0], [1.0]]), t=np.pi / 2)
    assert values.voltages[0] == pytest.approx(100.0)


def test_dirichlet_values_at_points():
    lens = _electrodes(voltages={"electrode": 1000.0, "wall": 0.0})
    points = np.array([[0.0, 0.0, 1.0, 3.0, 0.4], [0.0, 1.0, 1.0, 3.0, 1.0]])
    values = lens.dirichlet_values(points)
    np.testing.assert_array_equal(values.labels, [1, 2, 3, 0, 2])
    np.testing.assert_array_equal(
        values.voltages, [0.0, 1000.0, np.nan, np.nan, 1000.0]
    )
    np.testing.assert_array_equal(values.conductor, [True, True, False, False, True])
    assert values.names == ("wall", "electrode", "dielectric")


def test_dirichlet_values_follow_overlap_precedence():
    # the later, non-conducting object wins where it overlaps the electrode
    overlap = GeometryCollection(
        {"electrode": Circle((0, 0), 1), "insulator": Circle((1, 0), 0.5)},
        voltages={"electrode": 10.0},
    )
    values = overlap.dirichlet_values(np.array([[0.0, 0.8], [0.0, 0.0]]))
    np.testing.assert_array_equal(values.labels, [1, 2])
    np.testing.assert_array_equal(values.voltages, [10.0, np.nan])


def test_dirichlet_values_on_a_grid():
    rings = GeometryCollection(
        {
            "left": Tube((0, 0, -1.0), (0, 0, -0.6), 0.3, 0.5),
            "middle": Tube((0, 0, -0.2), (0, 0, 0.2), 0.3, 0.5),
            "right": Tube((0, 0, 0.6), (0, 0, 1.0), 0.3, 0.5),
        },
        voltages={"left": 0.0, "middle": 5000.0, "right": 0.0},
    )
    grid = Grid(
        ndim=3,
        lower_bound=(-0.6, -0.6, -1.2),
        upper_bound=(0.6, 0.6, 1.2),
        num_gridpoints=(13, 13, 25),
    )
    values = rings.dirichlet_values(grid)
    assert values.labels.shape == values.voltages.shape == (13, 13, 25)
    np.testing.assert_array_equal(values.conductor, rings.node_mask(grid))
    middle = values.labels == 2
    assert middle.any() and np.all(values.voltages[middle] == 5000.0)
    assert np.all(values.voltages[values.labels == 1] == 0.0)
