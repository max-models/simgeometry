"""Shapes, grids, and domains for numerical simulations."""

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

__version__ = "0.1.0"
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
