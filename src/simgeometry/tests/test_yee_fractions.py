import numpy as np
import pytest

from simgeometry import (
    Circle,
    Cuboid,
    FunctionObject2D,
    Grid,
    Polygon,
    Rectangle,
    Sphere,
)
from simgeometry.fractions import yee_fractions


def _clip(square, normal, offset):
    """Sutherland-Hodgman: the part of a convex polygon (k, 2) with normal . p < offset."""
    out = []
    for p, q in zip(square, np.roll(square, -1, axis=0)):
        fp, fq = normal @ p - offset, normal @ q - offset
        if fp < 0:
            out.append(p)
        if fp * fq < 0:
            out.append(p + fp / (fp - fq) * (q - p))
    return np.array(out).reshape(-1, 2)


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    x, y = polygon.T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _half_space_reference(normal, offset, lower, h, nodes):
    """Exact edge and face fractions and traces of ``normal . x < offset``."""
    ndim = len(nodes)
    cells = [n - 1 for n in nodes]
    edges = []
    for a in range(ndim):
        shape = tuple(cells[b] if b == a else nodes[b] for b in range(ndim))
        start = lower[:, None] + np.indices(shape).reshape(ndim, -1) * h[:, None]
        f0 = normal @ start - offset
        f1 = f0 + normal[a] * h[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.clip(f0 / (f0 - f1), 0, 1)
        inside = np.where(
            f0 < 0, np.where(f1 < 0, 1.0, t), np.where(f1 < 0, 1 - t, 0.0)
        )
        edges.append(inside.reshape(shape))
    families = [(1, 2), (2, 0), (0, 1)] if ndim == 3 else [(0, 1)]
    faces, traces, projected = [], [], []
    for number, (b, c) in enumerate(families):
        normal_axis = number if ndim == 3 else None
        shape = tuple(nodes[d] if d == normal_axis else cells[d] for d in range(ndim))
        corners = lower[:, None] + np.indices(shape).reshape(ndim, -1) * h[:, None]
        square = np.array([[0, 0], [h[b], 0], [h[b], h[c]], [0, h[c]]])
        in_plane = np.array([normal[b], normal[c]])
        area, trace = [], []
        for p in corners.T:
            local = offset - normal @ p
            poly = _clip(square, in_plane, local)
            area.append(_area(poly) / (h[b] * h[c]))
            # the trace: the clipped polygon's sides on the line
            edges_len = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
            mid = 0.5 * (np.roll(poly, -1, axis=0) + poly)
            interior = np.abs(mid @ in_plane - local) < 1e-12 * max(h)
            trace.append(np.sum(edges_len[interior]) if len(poly) >= 3 else 0.0)
        faces.append(np.reshape(area, shape))
        traces.append(np.reshape(trace, shape))
        projected.append(traces[-1] * np.linalg.norm(in_plane) / np.linalg.norm(normal))
    return edges, faces, traces, projected


def _rotation(angles):
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_plane_in_3d_is_exact():
    """A tilted half-space (a big rotated box): fractions and traces to round-off."""
    rotation = _rotation((0.3, -0.5, 0.8))
    # the box x < 0 (local), 40 wide, rotated: the half-space n . x < 0 near the grid
    box = Cuboid((-20, 0, 0), (40, 40, 40)).transformed(rotation, (0.03, -0.02, 0.01))
    normal = rotation[:, 0]
    offset = normal @ np.array([0.03, -0.02, 0.01])
    nodes = (6, 5, 7)
    grid = Grid(3, (-1, -0.9, -1.1), (1.1, 0.8, 1.2), nodes)
    result = yee_fractions(box, grid)
    h = np.asarray(result.spacing)
    edges, faces, traces, projected = _half_space_reference(
        normal, offset, np.array([-1, -0.9, -1.1]), h, nodes
    )
    for a in range(3):
        np.testing.assert_allclose(result.edges[a], edges[a], atol=1e-9)
        np.testing.assert_allclose(result.faces[a], faces[a], atol=1e-9)
        np.testing.assert_allclose(result.traces[a], traces[a], atol=1e-9)
        np.testing.assert_allclose(result.projected_traces[a], projected[a], atol=1e-9)
    assert 0 < np.mean(result.faces[0]) < 1
    assert np.count_nonzero((result.faces[2] > 0) & (result.faces[2] < 1)) > 5


def test_line_in_2d_is_exact():
    """A polygon edge across a 2D grid: edges, cells and traces exactly."""
    normal = np.array([0.6, -0.8])
    offset = 0.137
    # a large polygon whose only edge near the grid is normal . x = offset
    t = np.array([-normal[1], normal[0]])
    base = offset * normal
    vertices = [base + 50 * t, base - 50 * t, base - 50 * t - 50 * normal]
    vertices.append(base + 50 * t - 50 * normal)
    shape = Polygon([tuple(v) for v in vertices])
    nodes = (8, 7)
    grid = Grid(2, (-1, -1), (1, 0.9), nodes)
    result = yee_fractions(shape, grid)
    h = np.asarray(result.spacing)
    edges, faces, traces, projected = _half_space_reference(
        normal, offset, np.array([-1.0, -1.0]), h, nodes
    )
    np.testing.assert_allclose(result.edges[0], edges[0], atol=1e-9)
    np.testing.assert_allclose(result.edges[1], edges[1], atol=1e-9)
    np.testing.assert_allclose(result.faces[0], faces[0], atol=1e-9)
    np.testing.assert_allclose(result.traces[0], traces[0], atol=1e-9)
    np.testing.assert_array_equal(result.traces[0], result.projected_traces[0])


def test_layout_and_periodic_edges():
    grid = Grid(3, (0, 0, 0), (1, 1, 1), (4, 5, 6), periodic=(True, False, False))
    result = yee_fractions(Sphere((0.5, 0.5, 0.5), 0.3), grid)
    assert result.edges[0].shape == (4, 5, 6)  # periodic: as many cells as nodes
    assert result.edges[1].shape == (4, 4, 6)
    assert result.edges[2].shape == (4, 5, 5)
    assert result.faces[0].shape == (4, 4, 5)
    assert result.faces[1].shape == (4, 5, 5)
    assert result.faces[2].shape == (4, 4, 6)
    flat = yee_fractions(Circle((0.5, 0.5), 0.3), Grid(2, (0, 0), (1, 1), (5, 4)))
    assert flat.edges[0].shape == (4, 4)
    assert flat.edges[1].shape == (5, 3)
    assert flat.faces[0].shape == (4, 3)
    assert len(flat.faces) == 1


@pytest.mark.parametrize("radius", [0.7, 1.0])
def test_circle_area_and_perimeter_converge(radius):
    errors = []
    for n in (14, 27, 53, 105):
        grid = Grid(2, (-1.3, -1.3), (1.3, 1.3), (n, n))
        result = yee_fractions(Circle((0.011, 0.023), radius), grid)
        h = np.asarray(result.spacing)
        area = result.faces[0].sum() * h[0] * h[1]
        perimeter = result.traces[0].sum()
        errors.append(
            (
                abs(area / (np.pi * radius**2) - 1),
                abs(perimeter / (2 * np.pi * radius) - 1),
            )
        )
    errors = np.array(errors)
    assert np.all(errors[-1] < 5e-5)
    # second order: about a factor four per halving, from the first to the last grid
    order = np.log2(errors[0] / errors[-1]) / 3
    assert np.all(order > 1.8), order


def test_sphere_area_volume_and_sections_converge():
    """Twice the area from the projected traces; the volume; a section's area."""
    errors = []
    for n in (9, 17, 33):
        grid = Grid(3, (-1.3,) * 3, (1.3,) * 3, (n,) * 3)
        centre = np.array([0.01, 0.02, -0.03])
        result = yee_fractions(Sphere(tuple(centre), 1.0), grid)
        h = np.asarray(result.spacing)
        area = 0.5 * sum(result.projected_traces[a].sum() * h[a] for a in range(3))
        volume = result.faces[2].sum() * np.prod(h)
        k = n // 2  # the z-faces in one node plane: the disc of the section
        z = -1.3 + k * h[2] - centre[2]
        section = result.faces[2][:, :, k].sum() * h[0] * h[1]
        errors.append(
            (
                abs(area / (4 * np.pi) - 1),
                abs(volume / (4 * np.pi / 3) - 1),
                abs(section / (np.pi * (1 - z * z)) - 1),
            )
        )
    errors = np.array(errors)
    assert np.all(errors[-1] < [5e-4, 1e-3, 2e-4]), errors[-1]
    assert errors[0, 0] / errors[-1, 0] > 30  # area: second order or better


def test_slit_narrower_than_a_cell_is_kept():
    """A slit 0.4 cells wide in a plate: the edges across keep the gap exactly."""
    h = 0.25
    plate = Rectangle((0, 0), (20, 20)) - Rectangle((0.35, 0), (0.1, 40))
    grid = Grid(2, (-1, -1), (1, 1), (9, 9))
    result = yee_fractions(plate, grid)
    x_edges = result.edges[0]
    # the x-edges from x = 0.25 to 0.5 cross the slit: 0.6 of their length is metal
    np.testing.assert_allclose(x_edges[5], 0.6, atol=1e-12)
    np.testing.assert_allclose(np.delete(x_edges, 5, axis=0), 1.0)
    np.testing.assert_allclose(result.faces[0][5], 0.6, atol=1e-12)
    # both walls of the slit lie in those cells: twice the cell height each
    np.testing.assert_allclose(result.traces[0][5], 2 * h, atol=1e-12)
    assert result.traces[0].sum() == pytest.approx(2 * 2.0)


def test_sampled_shape_without_level_set():
    disc = FunctionObject2D(
        lambda p: (p[0] - 0.05) ** 2 + (p[1] + 0.02) ** 2 < 0.36,
        lower_bound=(-1, -1),
        upper_bound=(1, 1),
    )
    grid = Grid(2, (-1, -1), (1, 1), (41, 41))
    result = yee_fractions(disc, grid)
    h = np.asarray(result.spacing)
    assert result.faces[0].sum() * h[0] * h[1] == pytest.approx(np.pi * 0.36, rel=2e-3)
    assert result.traces[0].sum() == pytest.approx(2 * np.pi * 0.6, rel=2e-3)


def test_method_and_validation():
    grid = Grid(2, (0, 0), (1, 1), (5, 5))
    circle = Circle((0.5, 0.5), 0.3)
    np.testing.assert_array_equal(
        circle.yee_fractions(grid).faces[0], yee_fractions(circle, grid).faces[0]
    )
    with pytest.raises(ValueError, match="dimensions"):
        yee_fractions(Sphere((0, 0, 0), 1), grid)
    with pytest.raises(ValueError, match="max_hits"):
        yee_fractions(circle, grid, max_hits=0)


def test_surfaces_on_grid_lines_like_a_staircase():
    """Walls on node planes: their edges count as inside, the trace as the outside's.

    The plate minus a slit has its walls on grid lines, where ``contains`` of a
    difference is false; a box conductor's surface, where it is true. Both give the
    staircase: the edges in a wall are inside, and each wall counts for the cell on
    its outside.
    """
    h = 0.25
    grid = Grid(2, (-1, -1), (1, 1), (9, 9))
    plate = Rectangle((0, 0), (20, 20)) - Rectangle((0.375, 0), (0.25, 40))
    result = yee_fractions(plate, grid)
    np.testing.assert_array_equal(result.edges[1][[5, 6]], 1.0)  # the walls
    np.testing.assert_array_equal(result.faces[0][5], 0.0)  # the slit's cells
    np.testing.assert_allclose(result.traces[0][5], 2 * h)
    np.testing.assert_allclose(np.delete(result.traces[0], 5, axis=0), 0.0)
    box = Cuboid((0, 0.125, 0), (1.0, 0.75, 1.0))  # faces on node planes
    grid3 = Grid(3, (-1, -1, -1), (1, 1, 1), (9, 9, 9))
    solid = yee_fractions(box, grid3)
    total = sum(solid.traces[a].sum() * h for a in range(3))
    # each wall is counted by the faces normal to its two tangents, in every node
    # plane across it including those at its ends (the staircase's corner edges):
    # x-walls 4 y-planes * 1.0 + 5 z-planes * 0.75, y-walls 2 * 5 * 1.0, z-walls
    # 5 * 0.75 + 4 * 1.0, each times h, both sides
    expected = 2 * h * ((4 * 1.0 + 5 * 0.75) + (5 + 5) + (5 * 0.75 + 4 * 1.0))
    assert total == pytest.approx(expected)
    np.testing.assert_array_equal(solid.traces[0], solid.projected_traces[0])
    inside = (solid.faces[2] > 0) & (solid.faces[2] < 1)
    assert not inside.any()
