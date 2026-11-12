"""Reference geometry: W7-X's plasma boundary and a W7-X-like stand-in for its wall.

The boundary is that of DESC's W7-X example equilibrium (standard configuration, five
field periods), stored in ConStellaration's ``boundary.json`` layout (see the
``README.md`` next to this file). W7-X's real first wall (heat shield, baffles,
divertor targets, NBI armour) is IPP's CAD and not public; `w7x_like_wall` is a
uniform offset of the boundary, a stand-in to develop and test with, not a model of
the machine. Load the real wall with `TriangleMesh.read` where it is available.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from simgeometry.objects.fourier_surface import FourierSurface


def w7x_boundary_data() -> dict:
    """Return DESC's W7-X boundary as a ``boundary.json`` dict (with its ``source``)."""
    text = files(__name__).joinpath("w7x_desc_boundary.json").read_text()
    return json.loads(text)


def w7x_boundary(**kwargs: Any) -> FourierSurface:
    """Return the plasma boundary of DESC's W7-X example (standard configuration).

    Arguments go to `FourierSurface`. Major radius about 5.5 m, minor radius
    about 0.5 m, five field periods.
    """
    return FourierSurface.from_boundary(w7x_boundary_data(), **kwargs)


def w7x_like_wall(gap: float = 0.10, **kwargs: Any) -> FourierSurface:
    """Return a W7-X-like stand-in wall: the W7-X boundary offset outward by ``gap``.

    NOT the W7-X first wall. The real plasma-wall distance varies around the
    machine (the divertor targets cut into the edge islands, the heat shield and
    the baffles stand back by different amounts) and the ports and NBI armour are
    missing; ``gap`` (default 0.10 m) is a round number, not a W7-X value. Use it
    to exercise the wall-hit machinery on a W7-X-shaped surface. Arguments go to
    `FourierSurface.offset` (``tolerance``, modes) and on to `FourierSurface`; the
    default ``resolution`` is 128 by 64 points per field period (the triangles
    within about 4 mm of the surface).
    """
    kwargs.setdefault("resolution", (128, 64))
    return w7x_boundary().offset(gap, **kwargs)


__all__ = ["w7x_boundary", "w7x_boundary_data", "w7x_like_wall"]
