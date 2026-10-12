"""Package exports for geometric object definitions."""

# 1D objects
# 2D objects
from ..objects.aperture2d import Aperture2D
from ..objects.aperture3d import Aperture3D
from ..objects.circle import Circle
from ..objects.cuboid import Cuboid
from ..objects.function_object import FunctionObject2D, FunctionObject3D
from ..objects.line import Line
from ..objects.polygon import Polygon
from ..objects.rectangle import Rectangle

# 3D objects
from ..objects.sphere import Sphere
from ..objects.triangle import Triangle

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
