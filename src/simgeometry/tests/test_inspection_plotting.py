"""Render smoke tests for slices, holes, 3D surfaces and clipping."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("xarray")
pytest.importorskip("skimage")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simgeometry import Circle, GeometryCollection, Grid, SimulationDomain, Sphere


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_views_save_and_compose(tmp_path):
    ring = Circle((0, 0), 1) - Circle((0, 0), 0.4)
    figure, axes = plt.subplots(1, 2)
    assert ring.plot(ax=axes[0], resolution=40) is axes[0]
    ring.show(
        ax=axes[1],
        resolution=40,
        nodes=True,
        normals=True,
        display=False,
        save=tmp_path / "ring.png",
    )
    assert (tmp_path / "ring.png").stat().st_size > 0
    assert np.ma.getmaskarray(axes[0].collections[0].get_array()).any()
    sphere = Sphere((0, 0, 0), 1)
    axis = sphere.show_3d(
        resolution=18,
        normals=True,
        display=False,
        clip=((0, 0, 0), (1, 0, 0)),
        save=tmp_path / "sphere.png",
    )
    assert axis.name == "3d" and (tmp_path / "sphere.png").stat().st_size > 0
    axis = sphere.show(x=0.2, display=False)
    assert axis.get_xlabel() == "y" and axis.get_ylabel() == "z"


def test_domain_and_empty_grid_views(tmp_path):
    grid = Grid(
        ndim=3,
        lower_bound=(-1, -1, -1),
        upper_bound=(1, 1, 1),
        num_gridpoints=(5, 5, 5),
    )
    assert grid.show(z=0, display=False).name == "rectilinear"
    assert grid.show_3d(nodes=True, display=False).name == "3d"
    domain = SimulationDomain(
        ndim=3,
        lower_bound=(-1, -1, -1),
        upper_bound=(1, 1, 1),
        num_gridpoints=(5, 5, 5),
        objects=[Sphere((0, 0, 0), 0.5)],
    )
    assert domain.evaluate(z=0).shape == (5, 5)
    domain.show(z=0, grid=True, display=False)
    domain.show_3d(grid=True, resolution=16, display=False)
    geometry = GeometryCollection(
        {"a": Sphere((0, 0, 0), 0.5), "b": Sphere((0.5, 0, 0), 0.3)},
        styles={"b": {"color": "red", "alpha": 0.4}},
    )
    geometry.show_3d(display=False, resolution=16)
    hits = domain.cut_edges()
    hits.show(z=0, normals=True, display=False, save=tmp_path / "hits.png")
    assert (tmp_path / "hits.png").stat().st_size > 0


def test_pyvista_offscreen_clip_widget(tmp_path):
    pytest.importorskip("pyvista")
    viewer = Sphere((0, 0, 0), 1).show_3d(
        backend="pyvista",
        clip=True,
        resolution=14,
        display=False,
        save=tmp_path / "interactive.png",
    )
    try:
        assert getattr(viewer, "widgets", viewer).plane_clipped_meshes
        assert (tmp_path / "interactive.png").stat().st_size > 0
    finally:
        viewer.close()


def _lens():
    from simgeometry import GeometryCollection, Tube

    return GeometryCollection(
        {
            "outer": Tube((0, 0, -1), (0, 0, -0.4), 0.5, 0.7),
            "middle": Tube((0, 0, -0.2), (0, 0, 0.2), 0.5, 0.7),
            "probe": Sphere((0, 0, 0.8), 0.2),
        },
        voltages={"outer": 0.0, "middle": 1.0e4},
    )


def test_pyvista_draws_the_surfaces_with_plasma_plots(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("plasma_plots")
    viewer = _lens().show_3d(
        backend="pyvista", resolution=16, display=False, save=tmp_path / "lens.png"
    )
    try:
        assert len(viewer.meshes) >= 3
        assert viewer.legend is not None
        assert (tmp_path / "lens.png").stat().st_size > 0
    finally:
        viewer.close()


def test_pyvista_colors_conductors_by_voltage():
    pytest.importorskip("pyvista")
    pytest.importorskip("plasma_plots")
    viewer = _lens().show_3d(
        backend="pyvista", resolution=16, display=False, color_by="voltage"
    )
    try:
        voltages = [
            set(mesh.point_data["voltage"].tolist())
            for mesh in viewer.meshes
            if "voltage" in mesh.point_data
        ]
        assert {0.0} in voltages and {1.0e4} in voltages
        assert any(np.isnan(next(iter(v))) for v in voltages)  # the probe
        assert viewer.scalar_bars
    finally:
        viewer.close()


def test_color_by_voltage_needs_pyvista_and_voltages():
    with pytest.raises(ValueError, match="backend='pyvista'"):
        _lens().show_3d(color_by="voltage", display=False)
    with pytest.raises(ValueError, match="voltages"):
        Sphere((0, 0, 0), 1).show_3d(
            backend="pyvista", color_by="voltage", display=False
        )


def test_level_set_quantity_matches_the_level_set():
    ball = Sphere((0, 0, 0), 1)
    field = ball.evaluate(x=[0.0, 0.5, 2.0], y=0.0, z=0.0, quantity="level_set")
    np.testing.assert_allclose(field.values, [-1.0, -0.5, 1.0])


def test_level_set_field_is_drawn_by_plasma_plots():
    pytest.importorskip("pyvista")
    plasma_plots = pytest.importorskip("plasma_plots")
    field = Sphere((0.5, 0, 0), 1).level_set_field(resolution=12)
    assert field.dims == ("eta1", "eta2", "eta3")
    assert {"X", "Y", "Z"} <= set(field.coords)
    assert float(field.X.min()) == pytest.approx(-0.5)
    assert float(field.min()) < 0 < float(field.max())
    viewer = plasma_plots.pyvista_plots.pyvista_isosurface(
        field, values=[0.0], show_domain=False
    )
    try:
        assert viewer.meshes
    finally:
        viewer.close()


def _samples():
    """Interior samples of a sphere in simgeometry's (3, n) layout, surface ones with
    their outward normals."""
    ball = Sphere((0, 0, 0), 1)
    surface = ball.sample_surface(40, rng=1, resolution=16)
    return ball, ball.sample_interior(60, rng=1), surface, ball.surface_normal(surface)


def test_pyvista_draws_points_with_plasma_plots_in_one_legend():
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("plasma_plots")
    from matplotlib.colors import to_rgb

    ball, interior, surface, normals = _samples()
    viewer = ball.show_3d(
        backend="pyvista",
        resolution=16,
        display=False,
        points={
            "interior": interior,
            "surface": {"points": surface.T, "vectors": normals, "size": 4},
        },
    )
    try:
        clouds = [
            actor
            for actor in viewer.renderer.actors.values()
            if isinstance(getattr(actor, "mapper", None), pv.DataSetMapper)
            and actor.mapper.dataset.n_verts == actor.mapper.dataset.n_points > 0
        ]
        assert [actor.mapper.dataset.n_points for actor in clouds] == [60, 40]
        np.testing.assert_allclose(clouds[0].mapper.dataset.points, interior.T)
        # the next colours after the sphere's, and one legend of all three
        colors = [actor.prop.color.float_rgb for actor in clouds]
        np.testing.assert_allclose(colors, [to_rgb("C1"), to_rgb("C2")])
        assert viewer.legend.GetNumberOfEntries() == 3
        assert len(viewer.meshes) == 4  # the sphere, two point sets, the arrows
    finally:
        viewer.close()
    single = ball.show_3d(
        backend="pyvista", resolution=16, display=False, points=interior
    )
    try:
        assert single.legend is None  # an unlabelled set needs no legend
        assert len(single.meshes) == 2
    finally:
        single.close()


def test_matplotlib_scatters_points_with_arrows_and_legend_entries():
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection

    ball, interior, surface, normals = _samples()
    axis = ball.show_3d(
        resolution=16,
        display=False,
        points={
            "interior": interior,
            "surface": {"points": surface, "vectors": normals.T, "color": "red"},
        },
    )
    texts = [text.get_text() for text in axis.get_legend().get_texts()]
    assert texts == ["geometry", "interior", "surface"]
    scatters = [c for c in axis.collections if isinstance(c, Path3DCollection)]
    assert [len(c.get_offsets()) for c in scatters] == [60, 40]
    np.testing.assert_allclose(
        scatters[1].get_facecolor()[0][:3], matplotlib.colors.to_rgb("red")
    )
    lines = [c for c in axis.collections if isinstance(c, Line3DCollection)]
    assert len(lines) == 2  # the bounding box and the arrows


def test_points_need_three_coordinates_and_known_keys():
    ball = Sphere((0, 0, 0), 1)
    with pytest.raises(ValueError, match="shape"):
        ball.show_3d(display=False, resolution=8, points=np.zeros((2, 5)))
    with pytest.raises(ValueError, match="'vectors'"):
        ball.show_3d(display=False, resolution=8, points={"a": {"pts": np.zeros(3)}})
    with pytest.raises(ValueError, match="vectors of 'a'"):
        ball.show_3d(
            display=False,
            resolution=8,
            points={"a": {"points": np.zeros((3, 4)), "vectors": np.zeros((3, 2))}},
        )
