---
title: Installation
description: Create a Python environment and install simgeometry.
---

Python 3.10 or newer is required. Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the package and its requirements with pip:

```bash
pip install simgeometry
```

Import the package with `from simgeometry.grid import Grid`.

## Optional extras

| Extra  | Installs |
| ------ | -------- |
| `evaluate` | xarray for labelled coordinate samples |
| `mesh` | scikit-image for contour and surface extraction |
| `plot` | Matplotlib plus the `evaluate` and `mesh` extras |
| `interactive` | PyVista plus the `plot` extra for interactive 3D clipping |
| `mpi` | mpi4py for MPI runs |
| `test` | pytest, pytest-cov, and mpi4py |
| `docs` | the notebook runner, Matplotlib, and griffe for the API reference |
| `dev` | development tools plus the `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```

Run the tests from the repository root with `pytest src/simgeometry/tests`.
