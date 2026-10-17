"""Geometry contracts: coordinate layouts, topology, surfaces, and grid cuts."""

import numpy as np
import pytest

from simgeometry import (
    Circle,
    Cuboid,
    GeometryCollection,
    Grid,
    Rectangle,
    Sphere,
    Triangle,
)
from simgeometry.cut_edges import bracketed_edge_geometry
from simgeometry.objects import Aperture2D, FunctionObject2D
from simgeometry.queries import intersection_fractions


def test_boolean_topology_and_transform_snapshots():
    outer, hole = Circle((0, 0), 2), Circle((0, 0), 1)
    ring = outer - hole
    assert not ring.contains((0, 0))
    assert ring.contains((1.5, 0))
    assert not ring.contains((3, 0))
    assert (outer & hole).contains((0, 0))
    assert not (outer & hole).contains((1.5, 0))
    pair = hole | hole.translated((4, 0))
    assert pair.contains((4, 0)) and not pair.contains((2, 0))
    rotated = Rectangle((0, 0), (4, 1)).rotated(90).translated((2, 0))
    assert rotated.contains((2, 1.5))
    assert not rotated.contains((3.5, 0))
    outer.translate((10, 0))
    assert ring.contains((1.5, 0))  # snapshot independent of source mutation
    with pytest.raises(ValueError, match="invertible"):
        hole.scaled(0)


def test_axis_and_oblique_sections_keep_holes_and_world_coordinates():
    shell = Sphere((0, 0, 0), 2) - Sphere((0, 0, 0), 1)
    section = shell.slice(z=0)
    assert not section.contains((0, 0))
    assert section.contains((1.5, 0))
    oblique = shell.slice(origin=(0, 0, 0), normal=(1, 1, 1))
    p = np.array([[0.0, 1.5], [0.0, 0.0]])
    np.testing.assert_allclose(oblique.to_local(oblique.to_world(p)), p, atol=1e-14)
    np.testing.assert_allclose(oblique.to_world(p).sum(axis=0), 0, atol=1e-14)
    assert oblique.contains_points(p).tolist() == [False, True]
    assert not shell.slice(z=3).contains((0, 0))
    with pytest.raises(ValueError):
        shell.slice(x=0, y=0)


def test_evaluation_mesh_scattered_and_section_names():
    xr = pytest.importorskip("xarray")
    ball = Sphere((0, 0, 0), 1)
    result = ball.evaluate(x=[0, 2], y=[0, 0.5], z=0)
    assert result.dims == ("x", "y") and result.coords["z"].item() == 0
    np.testing.assert_array_equal(result, [[True, True], [False, False]])
    scattered = ball.evaluate(
        x=xr.DataArray([0, 2], dims="points"),
        y=xr.DataArray([0, 0], dims="points"),
        z=0,
    )
    assert scattered.dims == ("points",)
    assert scattered.values.tolist() == [True, False]
    section = ball.slice(x=0.2)
    result = section.evaluate(y=[0, 2], z=[0, 0.1])
    assert result.dims == ("y", "z")
    with pytest.raises(ValueError, match="Available"):
        section.evaluate(x=0)
    assert ball.evaluate(x=0, y=0, z=0).item()


def test_collection_precedence_and_labels():
    pytest.importorskip("xarray")
    geometry = GeometryCollection(
        {"left": Circle((0, 0), 1), "right": Circle((1, 0), 1)}
    )
    result = geometry.evaluate(x=[-0.5, 0.5, 3], y=0)
    assert result.values.tolist() == [1, 2, 0]
    assert result.attrs["labels"][2] == "right"
    assert geometry["left"].contains((0, 0))
    np.testing.assert_array_equal(
        geometry.evaluate(x=[0, 3], y=0, quantity="occupancy"), [True, False]
    )


@pytest.mark.parametrize(
    "shape, expected",
    [
        (Circle((0, 0), 1), [0.25, 0.75]),
        (Rectangle((0, 0), (2, 2)), [0.25, 0.75]),
        (Triangle([(-1, -1), (1, -1), (0, 1)]), [0.375, 0.625]),
        (Circle((0, 0), 1).scaled((2, 1)), [0, 1]),
    ],
)
def test_analytic_segment_intersections(shape, expected):
    hits = shape.intersect_segment((-2, 0), (2, 0))
    assert hits.exact
    np.testing.assert_allclose(hits.fractions, expected, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(hits.normals, axis=0), 1)
    assert hits.entering.tolist() == [1, -1]


def test_multiple_crossings_tangency_and_hidden_union_interfaces():
    ring = Circle((0, 0), 1.5) - Circle((0, 0), 0.5)
    hit = ring.intersect_segment((-2, 0), (2, 0))
    np.testing.assert_allclose(hit.fractions, [0.125, 0.375, 0.625, 0.875])
    assert hit.entering.tolist() == [1, -1, 1, -1]
    hit = Circle((0, 0), 1).intersect_segment((-2, 1), (2, 1))
    np.testing.assert_allclose(hit.fractions, [0.5])
    assert hit.entering.tolist() == [0]
    union = Circle((0, 0), 1) | Circle((1, 0), 1)
    np.testing.assert_allclose(
        union.intersect_segment((-2, 0), (3, 0)).fractions, [0.2, 0.8]
    )
    thin = Circle((0.5, 0), 1e-4)
    assert len(thin.intersect_segment((0, 0), (1, 0))) == 2


def test_exact_distances_closest_points_inversion_and_normal_transform():
    circle = Circle((0, 0), 1)
    np.testing.assert_allclose(
        circle.signed_distance([[0, 2, 0], [0, 0, 0.5]]), [-1, 1, -0.5]
    )
    np.testing.assert_allclose(circle.closest_point([2, 0]), [1, 0])
    np.testing.assert_allclose(
        Circle((0, 0), 1, inverted=True).surface_normal([1, 0]), [-1, 0]
    )
    box = Cuboid((0, 0, 0), (2, 4, 6))
    assert box.signed_distance([0, 0, 0]) == -1
    assert box.signed_distance([2, 0, 0]) == 1
    ellipse = circle.scaled((2, 1))
    p = np.array([np.sqrt(2), 1 / np.sqrt(2)])
    expected = np.array([p[0] / 4, p[1]])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(ellipse.surface_normal(p), expected)


def test_generic_intersection_reports_sampling_and_unclamped_bisection():
    obj = FunctionObject2D(
        lambda p: p[0] ** 2 + p[1] ** 2 <= 1, lower_bound=(-1, -1), upper_bound=(1, 1)
    )
    hit = obj.intersect_segment((-2, 0), (2, 0))
    assert not hit.exact
    np.testing.assert_allclose(hit.fractions, [0.25, 0.75])
    circle = Circle((1, 0), 1 - 1e-5)
    fractions = intersection_fractions(circle, [[0], [0]], [[1], [0]])
    assert fractions[0] == pytest.approx(1e-5, abs=1e-13)
    with pytest.raises(ValueError, match="brackets"):
        intersection_fractions(circle, [[1], [0]], [[0], [0]])


def test_cut_edges_thin_objects_and_diagnostics():
    grid = Grid(ndim=2, lower_bound=(0, -1), upper_bound=(1, 1), num_gridpoints=(2, 3))
    thin = Circle((0.5, 0), 0.01)
    hits = thin.cut_edges(grid)
    assert len(hits.fractions) == 2 and hits.exact
    np.testing.assert_allclose(hits.points[0], [0.49, 0.51])
    report = thin.diagnose(grid)
    assert report["objects_without_solid_nodes"] == ["geometry"]
    assert report["multiple_hit_edges"] == 1
    assert not bracketed_edge_geometry(grid, [thin]).cut.any()


def test_bracketed_edges_periodic_raw_fraction():
    grid = Grid(
        ndim=2,
        lower_bound=(0, -1),
        upper_bound=(1, 1),
        num_gridpoints=(4, 3),
        periodic=(True, False),
    )
    wall = Rectangle((0.925, 0), (0.15, 1))
    # Last node 0.75 is fluid; first node 0 is not solid for this object, so no
    # bracket exists even though a wall lies in the seam edge. Full hits see it.
    assert not bracketed_edge_geometry(grid, [wall]).cut.any()
    assert len(wall.cut_edges(grid).fractions) >= 1
    obj = Rectangle((0, 0), (0.2, 1))
    geometry = bracketed_edge_geometry(grid, [obj])
    assert geometry.fractions[0, 1][-1, 1] == pytest.approx(1.0)


def test_mesh_fallback_distance_sampling_and_exports(tmp_path):
    pytest.importorskip("skimage")
    sphere = Sphere((0, 0, 0), 1)
    mesh = sphere.to_mesh(resolution=24)
    assert mesh.cells.shape[1] == 3
    assert np.max(abs(np.linalg.norm(mesh.vertices, axis=1) - 1)) < 0.005
    for extension in ("npz", "obj", "stl"):
        path = mesh.save(tmp_path / f"sphere.{extension}")
        assert path.stat().st_size > 0
    ring = Circle((0, 0), 2) - Circle((0, 0), 1)
    assert ring.signed_distance([0, 0], resolution=96) == pytest.approx(1, abs=0.06)
    assert ring.signed_distance([1.5, 0], resolution=96) == pytest.approx(
        -0.5, abs=0.06
    )
    points = sphere.sample_surface(500, rng=4, resolution=24)
    np.testing.assert_allclose(np.linalg.norm(points, axis=0), 1, atol=0.01)
    inside = sphere.sample_interior(500, rng=4)
    assert sphere.contains_points(inside).all()
    np.testing.assert_array_equal(inside, sphere.sample_interior(500, rng=4))
    with pytest.raises(ValueError, match="bounds"):
        Circle((0, 0), 1, inverted=True).sample_interior(1)


def test_cell_fractions_resolve_known_flat_cut_and_periodic_faces():
    grid = Grid(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=(3, 3))
    wall = Rectangle((0, 0.5), (0.5, 2))
    result = wall.cell_fractions(grid, samples_per_axis=8)
    np.testing.assert_allclose(result.volume, [[0.5, 0.5], [0, 0]])
    assert result.faces[0].shape == (3, 2)
    assert result.faces[1].shape == (2, 3)
    np.testing.assert_allclose(result.faces[0][0], 1)
    periodic = Grid(
        ndim=2,
        lower_bound=(0, 0),
        upper_bound=(1, 1),
        num_gridpoints=(4, 3),
        periodic=(True, False),
    )
    faces = wall.cell_fractions(periodic).faces[0]
    assert faces.shape == (5, 2)
    np.testing.assert_array_equal(faces[0], faces[-1])


def test_named_transforms_and_slices_preserve_labels():
    pytest.importorskip("xarray")
    shapes = GeometryCollection(
        {"outer": Sphere((0, 0, 0), 2), "inner": Sphere((0, 0, 0), 1)}
    )
    moved = shapes.translated((4, 0, 0))
    assert moved.evaluate(x=4, y=0, z=0).item() == 2
    assert moved.slice(z=0).evaluate(x=4, y=0).item() == 2
    assert moved.names == shapes.names
    hits = moved.intersect_segment((1, 0, 0), (7, 0, 0))
    assert hits.object_names == ("outer", "inner", "inner", "outer")


@pytest.mark.parametrize("ndim", [2, 3])
def test_mesh_normals_point_outward_and_inverted_inward(ndim):
    pytest.importorskip("skimage")
    cls = Circle if ndim == 2 else Sphere
    for inverted, expected_sign in ((False, 1), (True, -1)):
        mesh = cls((0,) * ndim, 1, inverted=inverted).to_mesh(resolution=18)
        dot = np.sum(mesh.normals * mesh.centres, axis=1)
        assert np.all(dot * expected_sign > 0.9)


def test_arbitrary_function_distance_and_affine_fallback():
    pytest.importorskip("skimage")
    fn = FunctionObject2D(
        lambda p: p[0] ** 2 + p[1] ** 2 < 1, lower_bound=(-1, -1), upper_bound=(1, 1)
    )
    np.testing.assert_allclose(
        fn.signed_distance([[0, 2], [0, 0]], resolution=96), [-1, 1], atol=0.03
    )
    assert np.isnan(fn.surface_normal([2, 2])).all()
    ellipse = Circle((0, 0), 1).scaled((2, 1))
    np.testing.assert_allclose(
        ellipse.closest_point([3, 0], resolution=96), [2, 0], atol=0.04
    )
    reflected = Circle((0, 0), 1).scaled((-1, 1))
    np.testing.assert_allclose(reflected.surface_normal([-1, 0]), [-1, 0])


def test_nested_slices_map_back_to_world():
    pytest.importorskip("xarray")
    line = Sphere((0, 0, 0), 1).slice(x=0.2).slice(y=0.3)
    assert line.axis_names == ("z",)
    np.testing.assert_allclose(line.to_world([0.4]), [0.2, 0.3, 0.4])
    np.testing.assert_allclose(line.to_local([0.2, 0.3, 0.4]), [0.4])
    assert line.contains(0.4)
    assert line.evaluate(z=[0.4, 1]).values.tolist() == [True, False]


def test_one_dimensional_section_samples_actual_boundary():
    section = Circle((0, 0), 1).slice(y=0.6)
    np.testing.assert_allclose(abs(section.sample_surface(20, rng=0)), 0.8, atol=1e-12)
    np.testing.assert_allclose(
        section.signed_distance([[0, 1]]), [-0.8, 0.2], atol=1e-12
    )


def test_bracketed_edges_wrap_a_seam_image_short_of_the_period():
    # y is periodic with period 0.012 and 13 nodes: the last node plus one step
    # is 0.011999999999999999, the first node's image, not the top of the domain
    grid = Grid(
        ndim=2,
        lower_bound=(0, 0),
        upper_bound=(0.031, 0.012),
        num_gridpoints=(32, 13),
        periodic=(False, True),
    )
    assert grid.node_points()[1, 12] + grid.grid_spacing[1] < 0.012
    aperture = Aperture2D(
        axis=1,
        position=0.006,
        vertices=((0.008, 0.006), (0.0095, 0.002), (0.011, 0.006)),
    )
    geometry = bracketed_edge_geometry(grid, [aperture])
    assert geometry.cut.any()
