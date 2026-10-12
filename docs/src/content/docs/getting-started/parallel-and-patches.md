---
title: Parallel and multipatch grids
description: Split a grid across MPI ranks or combine separate grid patches.
---

## MPI decomposition

Install the `mpi` extra and run the following script with `mpiexec -n 4
python script.py`. Pass a communicator and choose which axes may be split:

```python
from mpi4py import MPI
from simgeometry.grid import Grid

grid = Grid(
    ndim=2,
    lower_bound=(0.0, 0.0),
    upper_bound=(1.0, 1.0),
    num_gridpoints=(18, 12),
    comm=MPI.COMM_WORLD,
    domain_decomposition=(True, True),
    num_ghost_points=1,
)

print(grid.mpi_rank, grid.proc_index_bounds, grid.num_gridpoints_local)
```

`proc_index_bounds` contains half-open global node-index ranges for this rank.
`num_gridpoints_local_with_ghosts` adds ghost nodes on both sides of each axis.
`neighbor_ranks` gives the lower and upper neighboring ranks per axis, using
`MPI.PROC_NULL` at nonperiodic walls. Set `periodic=(True, False)`, for example,
to wrap neighbors on the first axis.

The `domain_decomposition_class` property gives access to the
`DomainDecomposition` instance. Its `layout(shape, halo=...)` method returns a
matching `mpiarray.Layout` for field arrays; trailing component axes are not
split.

## Separate patches

`MultiPatchGridComposite` collects named `Grid` instances of the same
dimension. This example places two patches side by side:

```python
from simgeometry.grid import Grid, MultiPatchGridComposite

patches = {
    "left": Grid(2, (0.0, 0.0), (0.5, 1.0), (6, 11)),
    "right": Grid(2, (0.5, 0.0), (1.0, 1.0), (6, 11)),
}
composite = MultiPatchGridComposite(patches)

assert composite.find_patch((0.75, 0.25)) == "right"
patch_id, local = composite.global_to_local((0.75, 0.25))
assert patch_id == "right"
assert local.tolist() == [0.25, 0.25]
assert composite.local_to_global(local, patch_id).tolist() == [0.75, 0.25]
```

Use `patch_ids`, `get_patch(id)`, and `get_patch_bounds(id)` to inspect the
children. The composite inherits `Grid`; inherited single-grid properties
describe its first patch (alphabetically), so use the child grids when you
need coordinates or node counts for every patch.

`PatchLibrary` in `simgeometry.multipatch` also creates descriptors for single,
L, T, and cross layouts. Those descriptors supply bounds and node counts for
constructing child grids. Review their bounds before composing them: some
predefined layouts overlap, and `find_patch()` returns the first match.
