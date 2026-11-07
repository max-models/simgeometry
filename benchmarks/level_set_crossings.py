"""Edge crossings found on level sets against bisection of the membership test.

For each shape, random grid edges that start outside and end inside the shape are
handed to ``intersection_fractions`` twice: once as they are (Illinois steps on the
level set) and once through a wrapper that hides the level set (48 bisection steps on
``contains``). The script prints the time of both, the number of shape evaluations,
and the largest distance of the found crossings from the surface (the exact level set
there, as all these shapes have one). Evaluations are counted per edge: one per
point at which ``contains`` or the level set is evaluated.

    uv run python benchmarks/level_set_crossings.py
"""

from __future__ import annotations

import time

import numpy as np

from simgeometry import Circle, Cylinder, Sphere, Torus, Tube
from simgeometry.queries import intersection_fractions
from simgeometry.shapes import Shape

EDGES = 100_000
SPACING = 0.01
REPEATS = 5


class Counted(Shape):
    """A shape that counts evaluated points; ``level_set=False`` hides the level set."""

    def __init__(self, source, *, level_set):
        self.source = source
        self.has_level_set = level_set and source.has_level_set
        self.evaluations = 0
        super().__init__(source.ndim, source.lower_bound, source.upper_bound)

    def contains_points(self, points):
        self.evaluations += points.shape[1]
        return self.source.contains_points(points)

    def contains(self, coords):
        return self.source.contains(coords)

    def level_set(self, points):
        self.evaluations += points.shape[1]
        return self.source.level_set(points)


def bracketing_edges(shape, rng):
    """``EDGES`` outside-to-inside grid edges of length ``SPACING`` along random axes."""
    lower = np.asarray(shape.lower_bound) - SPACING
    upper = np.asarray(shape.upper_bound) + SPACING
    outside, inside, found = [], [], 0
    while found < EDGES:
        start = rng.uniform(lower[:, None], upper[:, None], (shape.ndim, 20 * EDGES))
        end = start.copy()
        end[rng.integers(0, shape.ndim, start.shape[1]), np.arange(start.shape[1])] += (
            SPACING
        )
        keep = ~shape.contains_points(start) & shape.contains_points(end)
        outside.append(start[:, keep])
        inside.append(end[:, keep])
        found += int(keep.sum())
    return (
        np.concatenate(outside, axis=1)[:, :EDGES],
        np.concatenate(inside, axis=1)[:, :EDGES],
    )


def run(name, shape, rng):
    outside, inside = bracketing_edges(shape, rng)
    results = {}
    for method, level_set in (("bisection", False), ("level set", True)):
        counted = Counted(shape, level_set=level_set)
        start = time.perf_counter()
        for _ in range(REPEATS):
            fractions = intersection_fractions(counted, outside, inside)
        elapsed = (time.perf_counter() - start) / REPEATS
        crossing = outside + fractions * (inside - outside)
        error = float(np.max(np.abs(shape.level_set(crossing))))
        results[method] = (elapsed, counted.evaluations / (REPEATS * EDGES), error)
    (t_b, n_b, e_b), (t_l, n_l, e_l) = results["bisection"], results["level set"]
    print(
        f"{name:24s} {t_b * 1e3:7.1f} ms {n_b:5.1f} evals {e_b:8.1e} | "
        f"{t_l * 1e3:7.1f} ms {n_l:5.1f} evals {e_l:8.1e} | {t_b / t_l:4.1f}x"
    )


def main():
    rng = np.random.default_rng(0)
    rings = Tube((0, 0, -0.1), (0, 0, 0.1), 0.5, 0.7)
    for k in range(1, 8):
        rings = rings | Tube((0, 0, 0.3 * k - 0.1), (0, 0, 0.3 * k + 0.1), 0.5, 0.7)
    shapes = {
        "Circle": Circle((0.1, 0.2), 1.0),
        "Sphere": Sphere((0.1, 0.2, 0.3), 1.0),
        "Cylinder (oblique axis)": Cylinder((0, 0, -1), (0.3, 0.2, 1), 0.5),
        "Torus": Torus((0, 0, 0), 1.0, 0.3, axis=(1, 1, 0)),
        "Sphere - Cylinder": Sphere((0, 0, 0), 1)
        - Cylinder((0, 0, -2), (0, 0, 2), 0.4),
        "union of 8 Tubes": rings,
    }
    print(
        f"{EDGES} edges of length {SPACING}; time per call, shape evaluations and "
        "the largest |level set| at the crossings\n"
        f"{'':24s} {'bisection (48 steps)':>32s} | {'level set (Illinois)':>32s} | speedup"
    )
    for name, shape in shapes.items():
        run(name, shape, rng)


if __name__ == "__main__":
    main()
