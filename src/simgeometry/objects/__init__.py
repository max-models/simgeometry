"""Package exports for geometric object definitions."""

# 1D objects
# 2D objects
from simgeometry.objects.aperture2d import Aperture2D
from simgeometry.objects.aperture3d import Aperture3D
from simgeometry.objects.circle import Circle
from simgeometry.objects.cuboid import Cuboid
from simgeometry.objects.function_object import FunctionObject2D, FunctionObject3D
from simgeometry.objects.line import Line
from simgeometry.objects.polygon import Polygon
from simgeometry.objects.rectangle import Rectangle

# 3D objects
from simgeometry.objects.sphere import Sphere
from simgeometry.objects.triangle import Triangle

__all__ = [
    # 2D objects
    "Aperture2D",
    "Aperture3D",
    "Circle",
    "Cuboid",
    "FunctionObject2D",
    "FunctionObject3D",
    # 1D objects
    "Line",
    "Polygon",
    "Rectangle",
    # 3D objects
    "Sphere",
    "Triangle",
]
