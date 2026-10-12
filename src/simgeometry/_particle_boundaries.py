"""Apply particle boundary conditions on the active array backend."""

from __future__ import annotations

import cunumpy as xp


def apply_particle_boundaries(
    positions, velocities, alive, reinject, lower, upper, lower_type, upper_type
) -> None:
    """Update component-major particle arrays in place.

    Boundary codes are kill=0, reflect=1, periodic=2, reinject=3.
    """
    reinject[:] = False
    in_domain = alive.copy()
    for axis in range(positions.shape[0]):
        active = alive & ~reinject
        x = positions[axis]
        low, high = lower[axis], upper[axis]
        below = active & (x < low)
        above = active & ((x >= high) if upper_type[axis] == 2 else (x > high))
        for mask, bound, side_type in (
            (below, low, lower_type[axis]),
            (above, high, upper_type[axis]),
        ):
            if side_type == 1:
                x[mask] = 2.0 * bound - x[mask]
                velocities[axis, mask] *= -1.0
            elif side_type == 2:
                wrapped = low + (x[mask] - low) % (high - low)
                x[mask] = xp.where(wrapped >= high, low, wrapped)
            elif side_type == 3:
                reinject |= mask
        checked = alive & ~reinject
        in_domain &= ~checked | ((x >= low) & (x < high))
    alive &= in_domain
