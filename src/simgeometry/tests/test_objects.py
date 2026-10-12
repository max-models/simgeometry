"""Tests for objects."""

from simgeometry.objects import Rectangle  # , Circle, Polygon

# from simgeometry.objects_3D import Cuboid, Sphere


def test_objects_2D(verbose: bool = False) -> None:
    """Verify two-dimensional object construction."""
    midpoint = [0, 0]
    side_lengths = [1, 2]
    bc = None
    square = Rectangle(midpoint, side_lengths, bc)

    assert square.contains([0, 0])
    assert square.contains([0.45, 0])
    assert not square.contains([0.55, 0])


def test_objects_3D(verbose: bool = False) -> None:
    """Verify three-dimensional object construction."""


if __name__ == "__main__":
    test_objects_2D()
    test_objects_3D()
