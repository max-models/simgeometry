"""Fourier surfaces: a circular torus against the closed forms, offsets, the W7-X data."""

from __future__ import annotations

import json
import types

import cunumpy as xp
import numpy as np
import pytest

from simgeometry import FourierSurface
from simgeometry.data import w7x_boundary, w7x_boundary_data, w7x_like_wall
from simgeometry.objects.fourier_surface import _circular_torus, _fourier_fit

MAJOR, MINOR = 3.0, 0.5


def _host(array):
    return np.asarray(xp.to_numpy(array))


def _tube_distance(points, major=MAJOR, minor=MINOR):
    """Signed distance ``(n,)`` of points ``(n, 3)`` to the circular torus."""
    rho = np.hypot(points[:, 0], points[:, 1]) - major
    return np.hypot(rho, points[:, 2]) - minor


def _ray_torus(start, end, major=MAJOR, minor=MINOR):
    """The first root in (1e-12, 1] of the ray-torus quartic, NaN for none."""
    poly = np.polynomial.polynomial
    out = np.full(start.shape[0], np.nan)
    for i, (a, b) in enumerate(zip(start, end)):
        d = b - a
        quadratic = [a @ a + major**2 - minor**2, 2 * a @ d, d @ d]
        planar = [
            a[0] ** 2 + a[1] ** 2,
            2 * (a[0] * d[0] + a[1] * d[1]),
            d[0] ** 2 + d[1] ** 2,
        ]
        quartic = poly.polysub(
            poly.polymul(quadratic, quadratic), 4 * major**2 * np.array(planar)
        )
        roots = poly.polyroots(quartic)
        roots = roots[np.abs(roots.imag) < 1e-7].real
        roots = roots[(roots > 1e-12) & (roots <= 1.0)]
        if roots.size:
            out[i] = roots.min()
    return out


@pytest.fixture(scope="module")
def torus():
    base = _circular_torus(MAJOR, MINOR)
    return FourierSurface(base.r_cos, base.z_sin, 1, resolution=(48, 288))


def test_torus_volume_area_and_box(torus):
    assert np.isclose(torus.volume(), 2 * np.pi**2 * MAJOR * MINOR**2, rtol=1e-13)
    assert np.isclose(torus.surface_area(), 4 * np.pi**2 * MAJOR * MINOR, rtol=1e-13)
    assert torus.upper_bound[2] >= MINOR and torus.lower_bound[2] <= -MINOR
    assert torus.upper_bound[0] >= MAJOR + MINOR
    assert torus.stellarator_symmetric
    assert (torus.n_poloidal, torus.n_toroidal) == (1, 0)


def test_torus_points_and_normals(torus):
    theta = np.linspace(0, 2 * np.pi, 7)[:, None]
    phi = np.linspace(0, 2 * np.pi, 5)[None, :]
    points = torus.point(theta, phi)
    assert points.shape == (7, 5, 3)
    assert np.allclose(_tube_distance(points.reshape(-1, 3)), 0.0, atol=1e-14)
    r, z = torus.rz(theta, phi)
    assert np.allclose(r, MAJOR + MINOR * np.cos(theta)) and np.allclose(
        z, MINOR * np.sin(theta)
    )
    normal = torus.normal(theta, phi)
    centre = np.stack(
        np.broadcast_arrays(MAJOR * np.cos(phi), MAJOR * np.sin(phi), 0 * phi), -1
    )
    assert np.allclose(normal, (points - centre) / MINOR, atol=1e-14)


def test_inside_against_the_closed_form(torus):
    rng = np.random.default_rng(0)
    points = rng.uniform([-3.7, -3.7, -0.7], [3.7, 3.7, 0.7], (50000, 3))
    inside = _host(torus.inside_at(points))
    distance = _tube_distance(points)
    error = torus.triangulation_error
    assert 1e-4 < error < 2e-3
    assert np.array_equal(
        inside[np.abs(distance) > 1.5 * error],
        distance[np.abs(distance) > 1.5 * error] < 0,
    )
    assert torus.contains((MAJOR, 0.0, 0.0)) and not torus.contains((0.0, 0.0, 0.0))


def test_ray_torus_intersections_against_the_quartic(torus):
    rng = np.random.default_rng(1)
    start = rng.uniform(-4.0, 4.0, (300, 3))
    end = rng.uniform(-4.0, 4.0, (300, 3))
    exact = _ray_torus(start, end)
    hits = torus.first_hit(start, end)
    agree = np.isfinite(exact) == hits.hit
    assert agree.mean() > 0.99  # only grazing segments within the sagitta may differ
    both = np.isfinite(exact) & hits.hit & hits.refined
    assert both.sum() > 50
    length = np.linalg.norm(end - start, axis=1)
    assert np.max(np.abs(hits.fraction[both] - exact[both]) * length[both]) < 1e-9
    assert np.allclose(_tube_distance(hits.point[both]), 0.0, atol=1e-10)
    assert np.allclose(hits.distance[both], hits.fraction[both] * length[both])
    # the angles of the hits give back their points
    assert np.allclose(
        torus.point(hits.theta[both], hits.phi[both]), hits.point[both], atol=1e-9
    )
    found = torus.crossings(start.T, end.T)
    assert np.allclose(_host(found.fraction), hits.fraction, equal_nan=True)


def test_unrefined_hits_are_second_order():
    base = _circular_torus(MAJOR, MINOR)
    rng = np.random.default_rng(2)
    start = rng.uniform(-4.0, 4.0, (300, 3))
    end = rng.uniform(-4.0, 4.0, (300, 3))
    exact = _ray_torus(start, end)
    errors = []
    for n in (16, 32):
        surface = FourierSurface(
            base.r_cos, base.z_sin, 1, resolution=(n, 6 * n), refine=False
        )
        hits = surface.first_hit(start, end)
        assert not np.any(hits.refined)
        assert not surface.crossings(start[:5].T, end[:5].T).exact
        both = np.isfinite(exact) & hits.hit
        # across the surface, the hit is within the triangles' distance from it
        assert (
            np.max(np.abs(_tube_distance(hits.point[both])))
            <= 1.01 * surface.triangulation_error
        )
        errors.append(surface.triangulation_error)
    assert 3.5 < errors[0] / errors[1] < 4.5


def test_a_hit_beyond_the_step_keeps_the_triangle():
    base = _circular_torus(MAJOR, MINOR)
    coarse = FourierSurface(base.r_cos, base.z_sin, 1, resolution=(8, 48))
    # from the tube's centre to just outside the chord between two vertices
    theta = np.pi / 8
    chord = MINOR * np.cos(np.pi / 8)
    start = np.array([[MAJOR, 0.0, 0.0]])
    end = start + 0.5 * (chord + MINOR) * np.array(
        [[np.cos(theta), 0.0, np.sin(theta)]]
    )
    hits = coarse.first_hit(start, end)
    assert hits.hit[0] and not hits.refined[0]
    assert np.isclose(np.linalg.norm(hits.point[0] - start[0]), chord, rtol=1e-12)
    assert not coarse.crossings(start.T, end.T).exact


def test_closest_points_on_the_torus(torus):
    rng = np.random.default_rng(3)
    points = rng.uniform([-3.6, -3.6, -0.6], [3.6, 3.6, 0.6], (500, 3))
    points = points[np.abs(np.hypot(points[:, 0], points[:, 1]) - MAJOR) > 0.05]
    theta, phi, distance = torus.closest(points)
    assert np.allclose(distance, np.abs(_tube_distance(points)), atol=1e-10)
    assert np.all((theta >= 0) & (theta < 2 * np.pi))


def test_offset_of_the_torus_is_exact():
    base = _circular_torus(MAJOR, MINOR, nfp=2)
    for distance in (0.2, -0.2):
        wall = base.offset(distance)
        assert wall.offset_residual < 1e-12
        assert np.isclose(wall.r_cos[0, wall.n_toroidal], MAJOR)
        assert np.isclose(wall.r_cos[1, wall.n_toroidal], MINOR + distance)
        assert np.isclose(wall.z_sin[1, wall.n_toroidal], MINOR + distance)
        assert wall.nfp == 2


def test_offset_tolerance_and_modes():
    # an elliptical torus: its offset is not a finite series
    r_cos = np.array([[3.0], [0.6]])
    z_sin = np.array([[0.0], [0.3]])
    ellipse = FourierSurface(r_cos, z_sin, 1)
    with pytest.warns(UserWarning, match="tolerance"):
        rough = ellipse.offset(
            0.1, tolerance=1e-14, poloidal_modes=4, toroidal_modes=None
        )
    assert rough.n_poloidal == 4 and rough.offset_residual > 1e-14
    fixed = ellipse.offset(0.1, poloidal_modes=6, toroidal_modes=0)
    assert (fixed.n_poloidal, fixed.n_toroidal) == (6, 0)
    fine = ellipse.offset(0.1, tolerance=1e-6)
    assert fine.offset_residual <= 1e-6
    sample = fine.point(*np.meshgrid(np.linspace(0, 2 * np.pi, 64), [0.0, 1.0]))
    _, _, distance = ellipse.closest(sample)
    assert np.allclose(distance, 0.1, atol=2e-6)
    with pytest.raises(ValueError, match="poloidal_modes"):
        ellipse.offset(0.1, poloidal_modes=0)
    assert ellipse.offset(0.0).offset_residual < 1e-12


def test_fourier_fit_recovers_a_series():
    theta = 2 * np.pi * np.arange(16) / 16
    zeta = 2 * np.pi * np.arange(12) / 12
    t, z = np.meshgrid(theta, zeta, indexing="ij")
    values = 1.0 + 0.5 * np.cos(2 * t - z) + 0.25 * np.sin(t + 2 * z) + 0.1 * np.cos(z)
    cos, sin, residual = _fourier_fit(values, 3, 2)
    assert residual < 1e-13
    assert np.isclose(cos[0, 2], 1.0) and np.isclose(cos[2, 3], 0.5)
    assert np.isclose(sin[1, 0], 0.25) and np.isclose(cos[0, 3], 0.1)


def test_not_symmetric_and_reversed_surfaces():
    # a circular torus whose poloidal angle starts at theta0 and runs backwards
    theta0 = 0.4
    r_cos = np.array([[MAJOR], [MINOR * np.cos(theta0)]])
    r_sin = np.array([[0.0], [-MINOR * np.sin(theta0)]])
    z_sin = np.array([[0.0], [-MINOR * np.cos(theta0)]])
    z_cos = np.array([[0.0], [-MINOR * np.sin(theta0)]])
    surface = FourierSurface(
        r_cos, z_sin, 1, r_sin=r_sin, z_cos=z_cos, resolution=(32, 192)
    )
    assert not surface.stellarator_symmetric
    assert np.isclose(surface.volume(), 2 * np.pi**2 * MAJOR * MINOR**2)
    assert surface.mesh.closed and surface.mesh.volume() > 0
    points = surface.point([0.0, 1.0, 2.0], 0.3)
    assert np.allclose(_tube_distance(points), 0.0, atol=1e-14)
    normal = surface.normal([0.0, 1.0, 2.0], 0.3)
    centre = np.array([MAJOR * np.cos(0.3), MAJOR * np.sin(0.3), 0.0])
    assert np.allclose(normal, (points - centre) / MINOR)
    wall = surface.offset(0.1)
    assert not wall.stellarator_symmetric
    assert np.allclose(
        _tube_distance(wall.point([0.5, 2.5], 1.0), minor=MINOR + 0.1), 0.0, atol=1e-12
    )
    hits = wall.first_hit([[MAJOR, 0.0, 0.0]], [[MAJOR + 1.0, 0.0, 0.0]])
    assert np.isclose(hits.distance[0], MINOR + 0.1) and np.allclose(
        hits.normal[0], [1, 0, 0]
    )


def test_inverted_surface(torus):
    hole = torus.copy()
    hole = FourierSurface(
        torus.r_cos, torus.z_sin, 1, resolution=torus.resolution, inverted=True
    )
    assert hole.contains((0.0, 0.0, 0.0)) and not hole.contains((MAJOR, 0.0, 0.0))
    assert not _host(hole.inside_at([[MAJOR, 0.0, 0.0]]))[0]
    hits = hole.first_hit([[MAJOR, 0.0, 0.0]], [[MAJOR + 1.0, 0.0, 0.0]])
    assert np.allclose(hits.normal[0], [-1.0, 0.0, 0.0])
    found = hole.crossings(
        np.array([[MAJOR], [0.0], [0.0]]), np.array([[MAJOR + 1.0], [0.0], [0.0]])
    )
    assert np.allclose(_host(found.normal)[:, 0], [-1.0, 0.0, 0.0])
    mesh = hole.to_mesh(resolution=8)
    assert mesh.cells.shape[0] == hole.mesh.n_faces


def test_boundary_layouts_and_errors():
    data = w7x_boundary_data()
    from_dict = FourierSurface.from_boundary(data)
    from_json = FourierSurface.from_boundary(json.dumps(data))
    from_object = FourierSurface.from_boundary(types.SimpleNamespace(**data))
    assert from_dict.nfp == from_json.nfp == from_object.nfp == 5
    round_trip = FourierSurface.from_boundary(from_dict.to_boundary())
    assert np.array_equal(round_trip.r_cos, from_dict.r_cos)
    assert round_trip.to_boundary()["is_stellarator_symmetric"]
    with pytest.raises(ValueError, match="lacks"):
        FourierSurface.from_boundary({"r_cos": [[1.0]]})
    with pytest.raises(ValueError, match="one shape"):
        FourierSurface(np.ones((2, 2)), np.ones((2, 2)), 1)
    with pytest.raises(ValueError, match="r_sin"):
        FourierSurface(np.ones((2, 1)), np.ones((2, 1)), 1, r_sin=np.ones((3, 1)))
    with pytest.raises(ValueError, match="n_field_periods"):
        FourierSurface(np.ones((2, 1)), np.ones((2, 1)), 0)
    with pytest.raises(ValueError, match="resolution"):
        FourierSurface(np.ones((2, 1)), np.ones((2, 1)), 1, resolution=(2, 1))
    with pytest.raises(ValueError, match="no volume"):
        FourierSurface(np.array([[3.0], [0.5]]), np.zeros((2, 1)), 1)


def test_w7x_boundary_and_stand_in_wall():
    plasma = w7x_boundary()
    assert plasma.nfp == 5 and plasma.r_cos.shape == (13, 25)
    assert 5.4 < plasma.r_cos[0, 12] < 5.6
    wall = w7x_like_wall(0.1, resolution=(48, 24))
    assert wall.offset_residual <= 1e-3
    theta, phi = np.meshgrid(
        np.linspace(0, 2 * np.pi, 24), np.linspace(0, 2 * np.pi / 5, 6)
    )
    _, _, distance = plasma.closest(wall.point(theta, phi))
    assert np.allclose(distance, 0.1, atol=2e-3)
    # lines from outside through the plasma's centre meet the wall before the plasma
    phi = np.linspace(0, 2 * np.pi, 10)
    r, z = plasma.rz(np.linspace(0, 2 * np.pi, 32, endpoint=False)[:, None], phi)
    r, z = r.mean(axis=0), z.mean(axis=0)
    starts = np.stack((7.5 * np.cos(phi), 7.5 * np.sin(phi), z), axis=1)
    centres = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)
    wall_hits = wall.first_hit(starts, centres)
    plasma_hits = plasma.first_hit(starts, centres)
    assert wall_hits.hit.all() and plasma_hits.hit.all()
    assert np.all(plasma_hits.distance - wall_hits.distance > 0.09)
    assert np.all(wall.inside_at(plasma_hits.point))


def test_lines_are_followed_from_hit_to_hit(torus):
    # through the tube near the x axis, between the triangulation's vertices
    start = np.array([[0.0, 0.05, 0.1]])
    end = np.array([[5.0, 0.05, 0.1]])
    exact = MAJOR + np.array([-1.0, 1.0]) * np.sqrt(MINOR**2 - 0.1**2)
    first = torus.first_hit(start, end)
    assert np.isclose(np.hypot(*first.point[0, :2]), exact[0], atol=1e-10)
    second = torus.first_hit(first.point, end)
    assert second.hit[0] and second.refined[0]
    assert np.isclose(np.hypot(*second.point[0, :2]), exact[1], atol=1e-10)
    third = torus.first_hit(second.point, end)
    assert not third.hit[0]
    found = torus.crossings(first.point.T, end.T)
    assert found.exact and np.allclose(_host(found.point)[:, 0], second.point[0])
