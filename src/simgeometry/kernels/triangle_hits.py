"""The segment–triangle test without Pyccel, vectorized over (segment, triangle) pairs.

Used by the NumPy versions of the ``mesh_first_hits`` and ``mesh_inside`` kernels and
by `simgeometry.objects.TriangleMesh` for the barycentric coordinates of its hits:
the same arithmetic as ``_triangle_fraction`` of ``mesh_first_hits_pyccel.py``.
"""

from __future__ import annotations

import numpy as np
from methodcard import Rule, method


def _fraction_rules() -> dict:
    """The SymPy rule of the fraction where a segment meets a triangle (for the tests)."""
    import sympy

    z0, z1, h = sympy.symbols("z0 z1 h", real=True)
    # the triangle (0, 0, h), (1, 0, h), (0, 1, h) and a segment along z through it
    rule = (h - z0) / (z1 - z0)

    def fraction(z0, z1, h):
        vertices = np.array([[0.0, 0.0, h], [1.0, 0.0, h], [0.0, 1.0, h]])
        t, _, _ = triangle_fractions(
            vertices,
            np.array([[0, 1, 2]]),
            np.array([0]),
            np.array([[0.25], [0.3], [z0]]),
            np.array([[0.0], [0.0], [z1 - z0]]),
        )
        return float(t[0])

    ranges = {"z0": (-2.0, -1.0), "z1": (1.0, 2.0), "h": (-0.5, 0.5)}
    return {"fraction": Rule(rule, implementation=fraction, ranges=ranges)}


@method(
    equations={
        "fraction": r"t = \frac{e_2 \cdot q}{e_1 \cdot p}, \quad u = \frac{s \cdot p}{e_1 \cdot p}, \quad v = \frac{d \cdot q}{e_1 \cdot p}, \qquad p = d \times e_2, \quad q = s \times e_1, \quad s = a - v_0",
        "hit": r"e_1 \cdot p \ne 0, \quad u \ge 0, \quad v \ge 0, \quad u + v \le 1",
    },
    assumptions=(
        r"triangle $v_0 v_1 v_2$ with $e_1 = v_1 - v_0$, $e_2 = v_2 - v_0$; segment $a + t d$, $t$ the fraction of the step",
        r"edges and vertices belong to the triangle; a segment in the triangle's plane ($e_1 \cdot p = 0$) never meets it",
        "the same arithmetic as the Pyccel kernels `mesh_first_hits` and `mesh_inside`",
    ),
    references=(
        "[Möller and Trumbore, J. Graphics Tools 2, 21 (1997)](https://doi.org/10.1080/10867651.1997.10487468)",
    ),
    symbolic=_fraction_rules,
)
def triangle_fractions(vertices, faces, face, a, d):
    """Fractions ``t`` and barycentric ``u``, ``v`` where segments meet triangles.

    ``a`` and ``d`` are ``(3, n)`` starts and steps, ``face`` ``(n,)`` the triangle of
    each pair; ``t`` is 2 where the line misses the triangle (whatever ``t`` would be,
    it is not limited to the segment here).
    """
    corners = vertices[faces[face]]  # (n, 3, 3)
    v0 = corners[:, 0].T
    e1 = corners[:, 1].T - v0
    e2 = corners[:, 2].T - v0
    p = np.cross(d, e2, axis=0)
    det = np.sum(e1 * p, axis=0)
    proper = det != 0.0
    inv = 1.0 / np.where(proper, det, 1.0)
    s = a - v0
    u = np.sum(s * p, axis=0) * inv
    q = np.cross(s, e1, axis=0)
    v = np.sum(d * q, axis=0) * inv
    t = np.sum(e2 * q, axis=0) * inv
    hit = proper & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
    return np.where(hit, t, 2.0), u, v
