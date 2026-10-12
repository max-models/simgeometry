"""Tests for FunctionObject2D and FunctionObject3D."""

import math

import numpy as np
import pytest

from simgeometry.objects.function_object import FunctionObject2D, FunctionObject3D


class MockGrid2D:
    """Lightweight 2-D grid mock for tests."""

    def __init__(self) -> None:
        """Initialize the mock grid."""
        self.indices = [(i, j) for i in range(5) for j in range(5)]
        self.grid_spacing = np.array([1.0, 1.0])

    def node_coordinates(self, index):
        """Return node coordinates for a grid whose lower bound is the origin."""
        return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing


class MockGrid3D:
    """Lightweight 3-D grid mock for tests."""

    def __init__(self) -> None:
        """Initialize the mock grid."""
        self.indices = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]
        self.grid_spacing = np.array([1.0, 1.0, 1.0])

    def node_coordinates(self, index):
        """Return node coordinates for a grid whose lower bound is the origin."""
        return np.atleast_1d(np.asarray(index, dtype=float)) * self.grid_spacing


# ---------------------------------------------------------------------------
# FunctionObject2D
# ---------------------------------------------------------------------------


class TestFunctionObject2D:
    """Tests for FunctionObject2D."""

    def _make_ring(
        self,
        r_inner: float = 1.0,
        r_outer: float = 2.0,
    ) -> FunctionObject2D:
        """Return a ring-shaped 2-D object."""

        def ring(coords: np.ndarray) -> bool:
            x, y = coords
            r2 = x**2 + y**2
            return r_inner**2 <= r2 <= r_outer**2

        return FunctionObject2D(
            func=ring,
            lower_bound=(-r_outer, -r_outer),
            upper_bound=(r_outer, r_outer),
        )

    def test_init_valid(self) -> None:
        """FunctionObject2D initialises correctly."""
        obj = self._make_ring()
        assert obj.lower_bound == (-2.0, -2.0)
        assert obj.upper_bound == (2.0, 2.0)
        assert obj.ndim == 2

    def test_init_invalid_func(self) -> None:
        """Non-callable func raises AssertionError."""
        with pytest.raises(AssertionError, match="func must be callable"):
            FunctionObject2D(
                func="not_a_function",
                lower_bound=(0.0, 0.0),
                upper_bound=(1.0, 1.0),
            )

    def test_init_invalid_lower_bound(self) -> None:
        """Wrong-length lower_bound raises AssertionError."""
        with pytest.raises(AssertionError):
            FunctionObject2D(
                func=lambda c: True,
                lower_bound=(0.0,),
                upper_bound=(1.0, 1.0),
            )

    def test_init_invalid_upper_bound(self) -> None:
        """Wrong-length upper_bound raises AssertionError."""
        with pytest.raises(AssertionError):
            FunctionObject2D(
                func=lambda c: True,
                lower_bound=(0.0, 0.0),
                upper_bound=(1.0, 1.0, 1.0),
            )

    def test_contains_inside(self) -> None:
        """Points inside the ring are reported as inside."""
        obj = self._make_ring()
        # On the annular band at radius 1.5
        assert obj.contains((1.5, 0.0))
        assert obj.contains((0.0, 1.5))

    def test_contains_outside(self) -> None:
        """Points outside the ring are reported as outside."""
        obj = self._make_ring()
        # Centre is inside inner hole
        assert not obj.contains((0.0, 0.0))
        # Far outside outer radius
        assert not obj.contains((3.0, 0.0))

    def test_contains_on_inner_boundary(self) -> None:
        """Points on the inner boundary are inside (inclusive)."""
        obj = self._make_ring()
        assert obj.contains((1.0, 0.0))

    def test_contains_on_outer_boundary(self) -> None:
        """Points on the outer boundary are inside (inclusive)."""
        obj = self._make_ring()
        assert obj.contains((2.0, 0.0))

    def test_func_property(self) -> None:
        """The func property returns the original callable."""

        def sentinel(c):
            return True

        obj = FunctionObject2D(
            func=sentinel,
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
        )
        assert obj.func is sentinel

    def test_get_vertices_returns_empty(self) -> None:
        """get_vertices returns an empty list."""
        obj = self._make_ring()
        assert obj.get_vertices() == []

    def test_get_indices_in_object(self) -> None:
        """Only grid indices inside the ring are returned."""
        obj = self._make_ring(r_inner=1.0, r_outer=2.0)
        grid = MockGrid2D()
        indices = obj.get_indices_in_object(grid)

        # Verify every returned index truly lies inside the ring
        for idx in indices:
            x, y = float(idx[0]), float(idx[1])
            r2 = x**2 + y**2
            assert 1.0 <= math.sqrt(r2) <= 2.0

    def test_bc_stored(self) -> None:
        """Boundary condition value is stored."""
        obj = FunctionObject2D(
            func=lambda c: True,
            lower_bound=(0.0, 0.0),
            upper_bound=(1.0, 1.0),
            bc=42.0,
        )
        assert obj._bc == 42.0


# ---------------------------------------------------------------------------
# FunctionObject3D
# ---------------------------------------------------------------------------


class TestFunctionObject3D:
    """Tests for FunctionObject3D."""

    def _make_torus(
        self,
        major_r: float = 3.0,
        tube_r: float = 1.0,
    ) -> FunctionObject3D:
        """Return a torus-shaped 3-D object."""

        def torus(coords: np.ndarray) -> bool:
            x, y, z = coords
            r = math.sqrt(x**2 + y**2)
            return (r - major_r) ** 2 + z**2 <= tube_r**2

        bound = major_r + tube_r
        return FunctionObject3D(
            func=torus,
            lower_bound=(-bound, -bound, -tube_r),
            upper_bound=(bound, bound, tube_r),
        )

    def test_init_valid(self) -> None:
        """FunctionObject3D initialises correctly."""
        obj = self._make_torus()
        assert obj.lower_bound == (-4.0, -4.0, -1.0)
        assert obj.upper_bound == (4.0, 4.0, 1.0)
        assert obj.ndim == 3

    def test_init_invalid_func(self) -> None:
        """Non-callable func raises AssertionError."""
        with pytest.raises(AssertionError, match="func must be callable"):
            FunctionObject3D(
                func=123,
                lower_bound=(0.0, 0.0, 0.0),
                upper_bound=(1.0, 1.0, 1.0),
            )

    def test_init_invalid_lower_bound(self) -> None:
        """Wrong-length lower_bound raises AssertionError."""
        with pytest.raises(AssertionError):
            FunctionObject3D(
                func=lambda c: True,
                lower_bound=(0.0, 0.0),
                upper_bound=(1.0, 1.0, 1.0),
            )

    def test_contains_inside(self) -> None:
        """Points inside the torus tube are reported as inside."""
        obj = self._make_torus(major_r=3.0, tube_r=1.0)
        # Point on the torus surface at (major_r, 0, 0)
        assert obj.contains((3.0, 0.0, 0.0))
        # Point slightly inside tube
        assert obj.contains((3.5, 0.0, 0.0))

    def test_contains_outside(self) -> None:
        """Points outside the torus are reported as outside."""
        obj = self._make_torus(major_r=3.0, tube_r=1.0)
        # Origin is in the torus hole
        assert not obj.contains((0.0, 0.0, 0.0))
        # Far outside
        assert not obj.contains((10.0, 10.0, 10.0))

    def test_func_property(self) -> None:
        """The func property returns the original callable."""

        def sentinel(c):
            return True

        obj = FunctionObject3D(
            func=sentinel,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        )
        assert obj.func is sentinel

    def test_get_vertices_returns_empty(self) -> None:
        """get_vertices returns an empty array."""
        obj = self._make_torus()
        verts, inverted = obj.get_vertices(slice_axis=0, slice_position=0.0)
        assert len(verts) == 0
        assert inverted is False

    def test_get_indices_in_object(self) -> None:
        """Only grid indices inside the torus are returned."""
        obj = self._make_torus(major_r=3.0, tube_r=1.0)
        grid = MockGrid3D()
        indices = obj.get_indices_in_object(grid)

        for idx in indices:
            coords = np.asarray(idx, dtype=float)
            assert obj.contains(coords)

    def test_volume_raises(self) -> None:
        """volume() raises NotImplementedError."""
        obj = self._make_torus()
        with pytest.raises(NotImplementedError):
            obj.volume()

    def test_surface_area_raises(self) -> None:
        """surface_area() raises NotImplementedError."""
        obj = self._make_torus()
        with pytest.raises(NotImplementedError):
            obj.surface_area()

    def test_bc_stored(self) -> None:
        """Boundary condition value is stored."""
        obj = FunctionObject3D(
            func=lambda c: True,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            bc=-5.0,
        )
        assert obj._bc == -5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
