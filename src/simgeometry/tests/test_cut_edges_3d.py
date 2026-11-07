"""Crossing points of bracketed edges and 3D views of cut edges."""

import numpy as np
import pytest

from simgeometry import Grid, Rectangle, Sphere, Tube
from simgeometry.cut_edges import bracketed_edge_geometry

CENTRE = np.array([0.03, 0.02, -0.01])
RADIUS = 0.62


def _grid3():
    return Grid(
        ndim=3,
        lower_bound=(-1, -1, -1),
        upper_bound=(1, 1, 1),
        num_gridpoints=(9, 9, 9),
    )


def _ball():
    return Sphere(tuple(CENTRE), RADIUS)


def test_bracketed_points_lie_on_the_sphere_with_outward_normals():
    edges = bracketed_edge_geometry(_grid3(), [_ball()])
    count = sum(np.isfinite(f).sum() for f in edges.fractions.values())
    assert edges.points.shape == edges.normals.shape == (3, count) and count > 0
    radial = edges.points - CENTRE[:, None]
    np.testing.assert_allclose(np.linalg.norm(radial, axis=0), RADIUS, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(edges.normals, axis=0), 1.0)
    np.testing.assert_allclose(np.sum(edges.normals * radial, axis=0), RADIUS)
    assert np.all(edges.crossing_owners == 0)


def test_bracketed_points_follow_the_fraction_order():
    grid = _grid3()
    edges = bracketed_edge_geometry(grid, [_ball()])
    h = np.asarray(grid.grid_spacing)
    start = 0
    for (axis, side), fraction in edges.fractions.items():
        cut = np.isfinite(fraction)
        expected = edges.nodes[:, cut].copy()
        expected[axis] += (1 if side else -1) * h[axis] * fraction[cut]
        stop = start + cut.sum()
        np.testing.assert_allclose(edges.points[:, start:stop], expected)
        start = stop
    assert start == edges.points.shape[1]


def test_bracketed_normals_belong_to_the_entered_object():
    grid = _grid3()
    left, right = Sphere((-0.5, 0, 0), 0.35), Sphere((0.5, 0, 0), 0.3)
    edges = bracketed_edge_geometry(grid, [left, right])
    for number, ball in enumerate((left, right)):
        mine = edges.crossing_owners == number
        assert mine.any()
        radial = edges.points[:, mine] - np.asarray(ball.midpoint)[:, None]
        np.testing.assert_allclose(np.linalg.norm(radial, axis=0), ball.radius)
        assert np.all(np.sum(edges.normals[:, mine] * radial, axis=0) > 0)


def test_bracketed_points_on_a_level_set_lie_on_the_surface():
    grid = Grid(
        ndim=3,
        lower_bound=(-2, -2, -1),
        upper_bound=(2, 2, 1),
        num_gridpoints=(17, 17, 9),
    )
    tube = Tube((0, 0, -0.4), (0, 0, 0.4), 1.0, 1.6, fillet=0.1)
    edges = bracketed_edge_geometry(grid, [tube])
    assert edges.points.shape[1] > 0
    assert np.abs(tube.level_set(edges.points)).max() < 1e-10
    outward = tube.level_set(edges.points + 1e-3 * edges.normals)
    assert np.all(outward > 0)


def test_periodic_crossings_are_wrapped_into_the_domain():
    grid = Grid(
        ndim=2,
        lower_bound=(0, -1),
        upper_bound=(1, 1),
        num_gridpoints=(4, 3),
        periodic=(True, False),
    )
    edges = bracketed_edge_geometry(grid, [Rectangle((0, 0), (0.2, 1))])
    assert edges.points.shape[0] == 2
    assert np.all((edges.points[0] >= 0) & (edges.points[0] < 1))
    seam = edges.fractions[0, 1][-1, 1]  # from node 0.75 to its image of node 0
    assert seam == pytest.approx(1.0)
    start = np.isfinite(edges.fractions[0, 0]).sum()
    np.testing.assert_allclose(edges.points[:, start], [0.0, 0.0])


def test_show_3d_needs_three_dimensions():
    grid = Grid(ndim=2, lower_bound=(-1, -1), upper_bound=(1, 1), num_gridpoints=(5, 5))
    edges = bracketed_edge_geometry(grid, [Rectangle((-0.3, -0.3), (0.6, 0.6))])
    with pytest.raises(ValueError, match="three-dimensional"):
        edges.show_3d(display=False)


def test_cut_edges_show_3d_pyvista_actors(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("plasma_plots")
    cuts = _ball().cut_edges(_grid3())
    viewer = cuts.show_3d(
        backend="pyvista",
        normals=True,
        nodes=True,
        resolution=16,
        display=False,
        save=tmp_path / "cuts.png",
    )
    try:
        assert {"crossings", "normals", "solid nodes"} <= set(viewer.actors)
        crossings = viewer.actors["crossings"].mapper.dataset
        np.testing.assert_allclose(crossings.points, cuts.points.T)
        assert len(viewer.meshes) >= 1  # the surface
        assert (tmp_path / "cuts.png").stat().st_size > 0
    finally:
        viewer.close()


def test_bracketed_show_3d_pyvista_without_surface():
    pytest.importorskip("pyvista")
    edges = bracketed_edge_geometry(_grid3(), [_ball()])
    viewer = edges.show_3d(backend="pyvista", surface=False, display=False)
    try:
        assert set(viewer.actors) == {"crossings"}
    finally:
        viewer.close()


def test_show_3d_matplotlib_scatter_and_quiver():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("skimage")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = bracketed_edge_geometry(_grid3(), [_ball(), Sphere((0.7, 0.7, 0.7), 0.2)])
    try:
        ax = edges.show_3d(normals=True, nodes=True, resolution=16, display=False)
        assert ax.name == "3d"
        # two surfaces, the solid nodes, the crossings and the normals
        assert len(ax.collections) == 5
    finally:
        plt.close("all")
