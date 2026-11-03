"""The swept polygon-object hit test without Pyccel, vectorized over the markers.

Used by the NumPy version of ``absorb_polygons_2d`` and by NumPy kernels of other
packages that fuse the test (picnic's ``push_electrostatic_2d``): the same
arithmetic as ``_object_hit_time`` of ``absorb_polygons_2d_pyccel.py``.
"""

from __future__ import annotations

import numpy as np
from methodcard import Rule, method


def _hit_time_rules() -> dict:
    """The SymPy rule of the crossing of a step with one edge (for the tests)."""
    import sympy

    x0, y0, x1, y1, a, b = sympy.symbols("x0 y0 x1 y1 a b", real=True)

    def hit_time(x0, y0, x1, y1, a, b):
        edge = np.array([[a, -2.0], [b, 2.0]])  # a polygon of two vertices: one edge
        times = object_hit_times(
            edge,
            np.array([0, 2]),
            0,
            1,
            0,
            *(np.array([v]) for v in (x0, y0, x1, y1)),
            np.array([True]),
        )
        return float(times[0])

    dx, dy, ex, ey, wx, wy = x1 - x0, y1 - y0, b - a, 4, a - x0, -2 - y0
    rule = (wx * ey - wy * ex) / (dx * ey - dy * ex)
    ranges = {"x0": (-2.0, -1.0), "y0": (-0.5, 0.5), "x1": (1.0, 2.0)}
    ranges |= {"y1": (-0.5, 0.5), "a": (-0.5, 0.5), "b": (-0.5, 0.5)}
    return {"crossing": Rule(rule, implementation=hit_time, ranges=ranges)}


@method(
    equations={
        "crossing": r"t = \frac{w \times e}{d \times e}, \quad u = \frac{w \times d}{d \times e}, \qquad d = x_1 - x_0, \quad e = v_i - v_j, \quad w = v_j - x_0",
        "inside": r"\text{toggle at edge } v_j v_i \iff (y_i > y_1) \ne (y_j > y_1) \;\text{and}\; x_1 < \frac{(x_j - x_i)(y_1 - y_i)}{(y_j - y_i) + 10^{-16}} + x_i",
        "hit_time": r"\tau = \begin{cases} \min\{t : 0 < t \le 1,\ 0 \le u \le 1\} & \text{a crossing (with sweep)} \\ 0 & x_1 \text{ in the solid} \\ 2 & \text{no hit} \end{cases}",
    },
    assumptions=(
        r"the object is the union of its polygons, negated if inverted; the end $x_1$ is tested by the even-odd rule over the vertices, without the bounding box of `Polygon.contains_points`",
        r"edges parallel to the step ($d \times e = 0$) are never crossings; a crossing at $t = 0$ is not a hit, at $t = 1$ it is",
        "the same arithmetic as `_object_hit_time` of the Pyccel kernel `absorb_polygons_2d` and its CUDA form `polygon_hit.cuh`",
    ),
    symbolic=_hit_time_rules,
)
def object_hit_times(
    vertices: np.ndarray,
    polygon_offsets: np.ndarray,
    first: int,
    last: int,
    inverted: int,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    sweep: np.ndarray,
) -> np.ndarray:
    """Where along each step from (x0, y0) to (x1, y1) the marker enters the object.

    The parameter in (0, 1] of the first edge crossing (where ``sweep``), else
    0 where (x1, y1) lies in the solid, else 2 (no hit).
    """
    inside = np.zeros(x1.shape, dtype=bool)
    crossing = np.full(x1.shape, 2.0)
    dx = x1 - x0
    dy = y1 - y0
    for polygon in range(first, last):
        begin, end = int(polygon_offsets[polygon]), int(polygon_offsets[polygon + 1])
        in_polygon = np.zeros(x1.shape, dtype=bool)
        xj, yj = vertices[end - 1]
        for xi, yi in vertices[begin:end]:
            in_polygon ^= ((yi > y1) != (yj > y1)) & (
                x1 < (xj - xi) * (y1 - yi) / ((yj - yi) + 1e-16) + xi
            )
            ex, ey = xi - xj, yi - yj
            denominator = dx * ey - dy * ex
            wx, wy = xj - x0, yj - y0
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wx * ey - wy * ex) / denominator
                u = (wx * dy - wy * dx) / denominator
            meets = (
                sweep
                & (denominator != 0.0)
                & (t > 0.0)
                & (t <= 1.0)
                & (u >= 0.0)
                & (u <= 1.0)
                & (t < crossing)
            )
            crossing = np.where(meets, t, crossing)
            xj, yj = xi, yi
        inside |= in_polygon
    if inverted == 1:
        inside = ~inside
    return np.where(crossing <= 1.0, crossing, np.where(inside, 0.0, 2.0))


def first_hits(
    vertices: np.ndarray,
    polygon_offsets: np.ndarray,
    object_first: np.ndarray,
    object_last: np.ndarray,
    object_inverted: np.ndarray,
    tested: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    sweep: np.ndarray,
) -> np.ndarray:
    """The object each marker meets first among those ``tested[k]`` marks, or -1."""
    best = np.full(x1.shape, 2.0)
    hit = np.full(x1.shape, -1, dtype=np.int64)
    for obj in range(object_first.shape[0]):
        if object_last[obj] <= object_first[obj] or not tested[obj].any():
            continue
        time = object_hit_times(
            vertices,
            polygon_offsets,
            int(object_first[obj]),
            int(object_last[obj]),
            int(object_inverted[obj]),
            x0,
            y0,
            x1,
            y1,
            sweep,
        )
        earlier = tested[obj] & (time < best)
        best = np.where(earlier, time, best)
        hit = np.where(earlier, obj, hit)
    return hit
