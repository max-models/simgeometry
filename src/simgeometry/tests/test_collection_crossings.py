"""Tests for the first crossing of steps against a collection of 3D absorbing surfaces."""

import cunumpy as xp
import numpy as np
import pytest

from simgeometry import Cuboid, GeometryCollection, Sphere
from simgeometry.objects import SlantedTube


def host(array):
    return np.asarray(xp.to_numpy(array))


def beamline():
    """Two dump plates, a duct and a calorimeter panel tilted by 30 degrees, along x."""
    return GeometryCollection(
        {
            "dump_upper": Cuboid((1.0, 0.0, 0.3), (1.0, 0.6, 0.02)),
            "dump_lower": Cuboid((1.0, 0.0, -0.3), (1.0, 0.6, 0.02)),
            "duct": SlantedTube((2.0, 0, 0), (4.0, 0, 0), 0.2, 0.25, (1.0, 0, 0)),
            "calorimeter": Cuboid((5.0, 0.0, 0.0), (0.02, 0.8, 0.8)).rotated(
                30.0, axis=(0, 0, 1), origin=(5.0, 0.0, 0.0)
            ),
        }
    )


def test_pencil_beam_lands_on_the_panel_it_is_aimed_at():
    geometry = beamline()
    n = 5
    # One step per pencil, each far longer than the plates are thick.
    start = np.array([[0.5] * n, [0.0] * n, [0.0, 0.29, -0.29, 0.22, 0.0]])
    end = start.copy()
    end[0] = 6.0
    end[2, 1], end[2, 2] = 0.4, -0.4  # aimed at the upper and lower dump plates
    hits = geometry.crossings(start, end)
    assert hits.exact
    assert host(hits.surface).tolist() == [3, 0, 1, 2, 3]
    names = [hits.names[i] for i in host(hits.surface)]
    assert names == ["calorimeter", "dump_upper", "dump_lower", "duct", "calorimeter"]
    # The calorimeter's face is at x = 5 - 0.01 / cos(30 deg) on the axis.
    x = host(hits.point)[0, 0]
    assert x == pytest.approx(5.0 - 0.01 / np.cos(np.pi / 6))
    normal = host(hits.normal)[:, 0]
    assert normal == pytest.approx([-np.cos(np.pi / 6), -np.sin(np.pi / 6), 0.0])


def test_power_balance_over_surfaces_and_exit_plane():
    geometry = beamline()
    rng = np.random.default_rng(1)
    n = 20_000
    start = np.zeros((3, n))
    direction = rng.normal(0.0, 0.1, (3, n))
    direction[0] = 1.0
    end = start + 6.0 * direction
    power = rng.uniform(0.5, 1.5, n)
    hits = geometry.crossings(start, end)
    totals = geometry.crossings(start, end).per_surface(power)
    assert list(totals) == list(geometry.names)
    escaped = power[~host(hits.hit)].sum()
    assert sum(totals.values()) + escaped == pytest.approx(power.sum())
    assert all(value > 0 for value in totals.values())
    assert host(hits.fraction)[host(hits.hit)].max() <= 1.0


def test_earliest_object_wins_whatever_the_order():
    near = Cuboid((1.0, 0, 0), (0.1, 1, 1))
    far = Cuboid((2.0, 0, 0), (0.1, 1, 1))
    start, end = np.zeros((3, 1)), np.array([[3.0], [0], [0]])
    for objects, index in (
        ({"near": near, "far": far}, 0),
        ({"far": far, "near": near}, 1),
    ):
        hits = GeometryCollection(objects).crossings(start, end)
        assert host(hits.surface).tolist() == [index]
        assert hits.per_surface() == {**dict.fromkeys(objects, 0.0), "near": 1.0}


def test_misses_and_sampled_members():
    geometry = GeometryCollection(
        {
            "ball": Sphere((1.0, 0, 0), 0.2),
            "union": Sphere((3, 0, 0), 0.2) | Sphere((3, 1, 0), 0.2),
        }
    )
    start = np.zeros((3, 3))
    end = np.array([[2.0, 4.0, 4.0], [0, 0, 5.0], [0, 0, 0]])
    hits = geometry.crossings(start, end)
    assert not hits.exact  # the union is sampled
    assert host(hits.surface).tolist() == [0, 0, -1]
    assert np.isnan(host(hits.normal)[:, 2]).all()


def test_single_shape_surface_and_per_surface_error():
    hits = Sphere((1.0, 0, 0), 0.2).crossings(
        np.zeros((3, 2)), np.array([[2.0, 0], [0, 2.0], [0, 0]])
    )
    assert host(hits.surface).tolist() == [0, -1]
    with pytest.raises(ValueError, match="GeometryCollection"):
        hits.per_surface()
