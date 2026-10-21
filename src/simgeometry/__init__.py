"""Shapes, grids, and domains for numerical simulations."""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.collection import GeometryCollection
from simgeometry.grid import Grid
from simgeometry.objects import (
    Aperture2D,
    Aperture3D,
    Circle,
    Cuboid,
    FunctionObject2D,
    FunctionObject3D,
    Line,
    Polygon,
    Polyline,
    Rectangle,
    Sphere,
    Triangle,
)
from simgeometry.simulation_domain import SimulationDomain

# simgeometry's regions stay off unless the application enables the "simgeometry" tag.
sp.register_opt_in_tag(*_PROFILE_TAGS)

try:
    __version__ = version("simgeometry")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"
__all__ = [
    "GeometryCollection",
    "Grid",
    "SimulationDomain",
    "Aperture2D",
    "Aperture3D",
    "Circle",
    "Cuboid",
    "FunctionObject2D",
    "FunctionObject3D",
    "Line",
    "Polygon",
    "Polyline",
    "Rectangle",
    "Sphere",
    "Triangle",
]
