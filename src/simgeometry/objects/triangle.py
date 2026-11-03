"""Triangle helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from methodcard import Rule, method

from simgeometry._types import Array, ConfigValue, Vertices2DLike
from simgeometry.shapes import Shape2D, _component_major, _level_set_at
from simgeometry.utils import vertices_to_bounds


def _triangle_rules() -> dict:
    """The SymPy rule of the `Triangle` level set (for the tests).

    Membership by the sides of the edges (a counter-clockwise triangle), the distance
    to the nearest edge.
    """
    import sympy

    x, y = sympy.symbols("x y", real=True)
    corners = [(0.0, 0.0), (1.2, 0.1), (0.2, 0.9)]
    triangle = Triangle(corners)

    def level_set(x, y):
        return _level_set_at(triangle, x, y)

    distances, sides = [], []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        ex, ey = bx - ax, by - ay
        t = sympy.Min(
            sympy.Max(((x - ax) * ex + (y - ay) * ey) / (ex**2 + ey**2), 0), 1
        )
        distances.append(sympy.sqrt((x - ax - t * ex) ** 2 + (y - ay - t * ey) ** 2))
        sides.append(ex * (y - ay) - ey * (x - ax) > 0)
    distance = sympy.Min(*distances)
    rule = sympy.Piecewise((-distance, sympy.And(*sides)), (distance, True))
    ranges = {"x": (-0.5, 1.5), "y": (-0.5, 1.2)}
    return {"level_set": Rule(rule, implementation=level_set, ranges=ranges)}


@method(
    equations={
        "inside": r"p = v_0 + u\,(v_2 - v_0) + v\,(v_1 - v_0), \qquad p \in S \iff u \ge 0,\; v \ge 0,\; u + v \le 1",
        "level_set": r"\phi(p) = \pm \min_\text{edges} \left| p - a - \operatorname{clip}\!\left(\tfrac{(p - a)\cdot e}{|e|^2}, 0, 1\right) e \right|, \qquad e = b - a",
    },
    assumptions=(
        r"barycentric coordinates $u, v$ with closed edges, inside the half-open bounding box",
        "a degenerate triangle contains no point",
        r"$\phi$: the distance to the nearest edge, negative inside; exact",
    ),
    references=(
        "[Quílez, 2D distance functions](https://iquilezles.org/articles/distfunctions2d/)",
    ),
    symbolic=_triangle_rules,
)
class Triangle(Shape2D):
    """Represent a triangular geometry object.

    Its `level_set` is the exact signed distance to its edges.
    """

    exact_outline = True
    has_level_set = True
    level_set_exact = True

    def __init__(
        self,
        vertices: Vertices2DLike,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a triangle in 2D space.

        Parameters:
        vertices (list or tuple): List of three tuples, each representing a vertex (x, y).
        bc (optional): Boundary condition information.
        inverted (bool, optional): Cover everything except the triangle.
        """
        assert len(vertices) == 3, "Triangle must have exactly three vertices."
        assert all(len(vertex) == 2 for vertex in vertices), (
            "Each vertex must have two coordinates."
        )

        self._vertices = [xp.array(vertex) for vertex in vertices]
        self._bc = bc
        self._inverted = bool(inverted)

        # Calculate bounds
        xs = [vertex[0] for vertex in self._vertices]
        ys = [vertex[1] for vertex in self._vertices]

        lower_bound = (min(xs), min(ys))
        upper_bound = (max(xs), max(ys))

        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the triangle across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis).
            position (float): Position along the axis to mirror.
        """
        for i in range(3):
            if axis == 0:  # Mirror across vertical line x = position
                self._vertices[i][0] = 2 * position - self._vertices[i][0]
            elif axis == 1:  # Mirror across horizontal line y = position
                self._vertices[i][1] = 2 * position - self._vertices[i][1]
            else:
                raise ValueError("Axis must be 0 (x-axis) or 1 (y-axis).")

        lower_bound, upper_bound = vertices_to_bounds(xp.array(self._vertices))

        self.set_bounding_box(lower_bound, upper_bound)

    def copy(self) -> Triangle:
        """
        Create a copy of the triangle.

        Returns:
            Triangle: A new instance of Triangle with the same vertices and boundary conditions.
        """
        vertices_copy = [vertex.copy() for vertex in self._vertices]
        return Triangle(vertices=vertices_copy, bc=self._bc, inverted=self._inverted)

    def translate(self, offset: tuple[float, float]) -> None:
        """
        Translate the xyz by the given offset.

        Parameters:
            offset (tuple): (dx, dy) translation vector.
        """
        offset_vec = xp.asarray(offset)

        # Apply translation to each vertex
        for i in range(3):
            self._vertices[i] += offset_vec

        lower_bound, upper_bound = vertices_to_bounds(xp.array(self._vertices))

        self.set_bounding_box(lower_bound, upper_bound)

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the (possibly inverted) triangle.

        Parameters:
        coords (list or tuple): Coordinates (x, y).

        Returns:
        bool: True if the point is inside the object, False otherwise.
        """
        # Use the same arithmetic as the batch path. Separate scalar dot
        # products can round differently at an edge and disagree with node_mask.
        point = xp.asarray(coords, dtype=float).reshape(2, 1)
        return bool(xp.to_numpy(self.contains_points(point))[0])

    def contains_points(self, points: Array) -> Array:
        """Vectorised `contains`: a mask of the ``(2, n)`` points inside."""
        points = _component_major(points, 2)
        vertices = [xp.asarray(vertex, dtype=float) for vertex in self._vertices]
        lower = xp.asarray(self.bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.bounding_box.upper_bound, dtype=float)[:, None]
        in_box = xp.all((points >= lower) & (points < upper), axis=0)
        # Barycentric coordinates for every point, including the edges.
        v0 = vertices[2] - vertices[0]
        v1 = vertices[1] - vertices[0]
        v2x = points[0] - vertices[0][0]
        v2y = points[1] - vertices[0][1]
        dot00 = v0[0] * v0[0] + v0[1] * v0[1]
        dot01 = v0[0] * v1[0] + v0[1] * v1[1]
        dot02 = v0[0] * v2x + v0[1] * v2y
        dot11 = v1[0] * v1[0] + v1[1] * v1[1]
        dot12 = v1[0] * v2x + v1[1] * v2y
        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            inside = xp.zeros(points.shape[1], dtype=bool)  # degenerate triangle
        else:
            inv_denom = 1 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            inside = in_box & (u >= 0) & (v >= 0) & (u + v <= 1)
        return ~inside if self.inverted else inside

    def _level_set(self, points: Array) -> Array:
        """Exact signed distance to the triangle's edges."""
        return self._edge_level_set(points)

    def _corner_loops(self) -> list[np.ndarray]:
        """The three vertices as one ``(2, 3)`` loop."""
        return [
            np.array(
                [[float(xp.to_numpy(v)[i]) for v in self._vertices] for i in (0, 1)]
            )
        ]
