"""Array bounds shared by geometric regions and grids."""

from __future__ import annotations


class Bounds:
    def __init__(self, lower_bound, upper_bound) -> None:
        assert len(lower_bound) == len(upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.ndim = len(lower_bound)
        assert self.ndim in (1, 2, 3)
        for axis in range(self.ndim):
            assert upper_bound[axis] >= lower_bound[axis]
        self.sidelengths = upper_bound - lower_bound
        self.midpoint = lower_bound + 0.5 * self.sidelengths
