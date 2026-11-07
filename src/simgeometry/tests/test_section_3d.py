"""Outlines of plane sections, in plane and world coordinates, and their 3D view."""

import numpy as np
import pytest

pytest.importorskip("skimage")

from simgeometry import Cuboid, Sphere
from simgeometry.shapes import _signed_area


def _shell():
    return Sphere((0, 0, 0), 1) - Sphere((0, 0, 0), 0.55)


def test_oblique_section_of_a_sphere_is_a_circle_in_the_plane():
    origin, normal = np.array([0.1, -0.2, 0.3]), np.array([1.0, 2.0, -0.5])
    section = Sphere((0, 0, 0), 1).slice(origin=origin, normal=normal)
    unit = normal / np.linalg.norm(normal)
    offset = float(origin @ unit)
    centre, radius = offset * unit, np.sqrt(1 - offset**2)

    (loop,) = section.outlines(world=True)
    assert loop.shape[0] == 3 and loop.shape[1] > 100
    np.testing.assert_allclose(unit @ loop, offset, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(loop - centre[:, None], axis=0), radius, atol=2e-3
    )
    (local,) = section.outlines()
    np.testing.assert_allclose(section.to_world(local), loop)
    assert _signed_area(local) > 0


def test_section_of_a_shell_has_an_outer_loop_and_a_hole():
    outer, hole = sorted(_shell().slice(z=0.2).outlines(), key=_signed_area)[::-1]
    assert _signed_area(outer) > 0 > _signed_area(hole)
    np.testing.assert_allclose(np.hypot(*outer), np.sqrt(1 - 0.04), atol=2e-3)
    np.testing.assert_allclose(np.hypot(*hole), np.sqrt(0.55**2 - 0.04), atol=2e-3)


def test_axis_aligned_section_keeps_the_remaining_axes():
    box = Cuboid((1, 0.5, 1.5), (2, 1, 3))  # [0, 2] x [0, 1] x [0, 3]
    section = box.slice(y=0.5)
    assert section.axis_names == ("x", "z")
    (loop,) = section.outlines(max_spacing=0.05)
    assert np.all(np.hypot(*np.diff(loop, axis=1)) <= 0.05 + 1e-12)
    np.testing.assert_allclose(loop.min(axis=1), (0, 0), atol=0.05)
    np.testing.assert_allclose(loop.max(axis=1), (2, 3), atol=0.05)
    (world,) = section.outlines(world=True)
    np.testing.assert_allclose(world[1], 0.5)


def test_sections_missing_the_shape_or_of_2d_shapes_have_no_outlines():
    assert (
        Sphere((0, 0, 0), 1).slice(origin=(0, 0, 2), normal=(0, 0, 1)).outlines() == []
    )
    from simgeometry import Circle

    with pytest.raises(NotImplementedError, match="2D section"):
        Circle((0, 0), 1).slice(x=0).outlines()
    with pytest.raises(ValueError, match="positive"):
        Sphere((0, 0, 0), 1).slice(z=0).outlines(max_spacing=0)


def test_show_3d_draws_the_shape_the_plane_and_the_outline(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("plasma_plots")
    section = _shell().slice(origin=(0, 0, 0.1), normal=(1, 1, 1))
    viewer = section.show_3d(
        backend="pyvista",
        resolution=16,
        outline_resolution=48,
        display=False,
        save=tmp_path / "section.png",
    )
    try:
        plane = viewer.actors["section plane"].mapper.dataset
        outline = viewer.actors["section outline"].mapper.dataset
        assert plane.n_cells == 1 and outline.n_lines == 2
        normal = np.ones(3) / np.sqrt(3)
        for points in (plane.points, outline.points):
            np.testing.assert_allclose(points @ normal, 0.1 / np.sqrt(3), atol=1e-12)
        surfaces = [
            actor.mapper.dataset
            for name, actor in viewer.actors.items()
            if not name.startswith("section")
            and hasattr(getattr(actor, "mapper", None), "dataset")
        ]
        assert any(getattr(s, "n_faces_strict", 0) > 100 for s in surfaces)
        assert (tmp_path / "section.png").stat().st_size > 0
    finally:
        viewer.close()


def test_show_3d_with_matplotlib(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = (
        Sphere((0, 0, 0), 1)
        .slice(z=0.3)
        .show_3d(
            resolution=12, outline_resolution=32, display=False, save=tmp_path / "s.png"
        )
    )
    try:
        assert hasattr(ax, "set_zlabel")
        (line,) = ax.lines
        np.testing.assert_allclose(line.get_data_3d()[2], 0.3)
        assert (tmp_path / "s.png").stat().st_size > 0
    finally:
        plt.close("all")


def test_show_3d_needs_a_plane_section():
    from simgeometry import Circle

    with pytest.raises(ValueError, match="plane section"):
        Circle((0, 0), 1).slice(x=0).show_3d(display=False)
