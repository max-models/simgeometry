"""The animated figure at the top of the docs home page: ``docs/src/assets/hero.svg``.

simgeometry describes a small ion-optics column on a grid: a rounded source plate, three slit
apertures and a round target. The figure shows their outlines and the grid nodes inside them
(``node_mask``), with the domain decomposition behind them as blocks
(``DomainDecomposition.preview``); the blocks reshuffle as the rank count steps through 2, 4, 6,
8, 12 and 16. Hovering a block gives its rank and how many electrode nodes it holds. Takes well
under a second.

Rerun with ``make docs-hero`` or ``python docs/hero.py`` (needs ``plasma_plots.hero``).
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

from simgeometry import Aperture2D, Circle, Rectangle
from simgeometry.domain_decomposition import DomainDecomposition
from simgeometry.grid import Grid

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
LOWER, UPPER, NODES = (0.0, 0.0), (8.0, 3.0), (161, 61)
RANKS = (2, 4, 6, 8, 12, 16)
STEP = 3.0  # seconds per rank count

grid = Grid(2, LOWER, UPPER, NODES)
shapes = [
    Rectangle(midpoint=(3.3, 1.5), side_lengths=(0.3, 1.6), fillet=0.12),
    Aperture2D.slit(4.3, half_width=0.35, thickness=0.15, length=1.0, position=1.5),
    Aperture2D.slit(5.3, half_width=0.25, thickness=0.15, length=1.0, position=1.5),
    Aperture2D.slit(6.3, half_width=0.15, thickness=0.15, length=1.0, position=1.5),
    Circle(midpoint=(7.3, 1.5), radius=0.3),
]
mask = np.zeros(NODES, dtype=bool)
for shape in shapes:
    mask |= shape.node_mask(grid)
nodes = grid.node_points()
inside = mask.ravel()

fig = HeroFigure(xlim=(LOWER[0], UPPER[0]), ylim=(LOWER[1], UPPER[1]))

# the decomposition for each rank count, shown in turn
phases = len(RANKS)
for phase, size in enumerate(RANKS):
    preview = DomainDecomposition.preview(size, decompose=[True, True])
    shown = [0.0] * phases
    shown[phase] = 1.0
    for rank, (lo, hi) in enumerate(preview.subdomain_boxes(LOWER, UPPER, NODES)):
        owned = (
            (nodes[0] >= lo[0])
            & (nodes[0] < hi[0])
            & (nodes[1] >= lo[1])
            & (nodes[1] < hi[1])
        )
        coord = preview.get_proc_coord(rank)
        block = fig.rect(
            *lo,
            *hi,
            tone="low",
            opacity=0.15 + 0.4 * (sum(coord) % 2),
            outline="muted",
            title=f"rank {rank} of {size}: {int((owned & inside).sum())} electrode nodes",
        )
        block.attrs["stroke-width"] = "1"
        # hold each rank count for a step, crossfading over a sixth of it
        values = [v for v in shown for _ in range(6)]
        block.animate("opacity", values + values[:1], STEP * phases)

for shape in shapes:
    for loop in shape.outlines(max_spacing=0.05):
        poly = fig.polygon(*loop, tone="bg", outline="accent")
        poly.attrs["stroke-width"] = "2"
fig_nodes = np.flatnonzero(inside)
for x, y in zip(nodes[0][fig_nodes], nodes[1][fig_nodes], strict=True):
    fig.circle(x, y, 1.6, tone="high", opacity=0.8)

print(f"Saved {fig.save(OUTPUT)}")
