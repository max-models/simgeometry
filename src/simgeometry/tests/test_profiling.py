"""simgeometry's scope-profiler regions are off unless the "simgeometry" tag is enabled."""

from __future__ import annotations

import pytest
import scope_profiler as sp

from simgeometry import Circle, Grid


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _node_mask_regions(**setup) -> list[str]:
    sp.setup(output=None, **setup)
    grid = Grid(ndim=2, lower_bound=(0, 0), upper_bound=(1, 1), num_gridpoints=(16, 16))
    Circle(midpoint=(0.5, 0.5), radius=0.25).node_mask(grid)
    results = sp.finalize(verbose=False, return_results=True)
    return [name for name in results.region_names if name.startswith("simgeometry.")]


def test_regions_are_off_by_default() -> None:
    assert _node_mask_regions() == []


def test_enabling_the_tag_records_the_regions() -> None:
    assert _node_mask_regions(enable_tags=["simgeometry"]) == ["simgeometry.node_mask"]
