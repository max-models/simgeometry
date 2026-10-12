"""Function-based geometric object definitions."""

from __future__ import annotations

from typing import Callable

import cunumpy as xp

from simgeometry._types import Array, BoundsLike, ConfigValue
from simgeometry.shapes import Shape2D, Shape3D


class FunctionObject2D(Shape2D):
    """Represent an arbitrary 2D geometry object defined by a Boolean function.

    The object interior is the set of all points for which ``func(coords)``
    returns ``True``.  A bounding box must be supplied by the caller because
    it cannot in general be derived from an arbitrary function.

    Example
    -------
    Create a ring (annulus) with inner radius 1 and outer radius 2::

        def ring(coords):
            x, y = coords
            r2 = x**2 + y**2
            return 1.0**2 <= r2 <= 2.0**2

        obj = FunctionObject2D(
            func=ring,
            lower_bound=(-2.0, -2.0),
            upper_bound=(2.0, 2.0),
        )
    """

    def __init__(
        self,
        func: Callable[[Array], bool],
        lower_bound: BoundsLike,
        upper_bound: BoundsLike,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a function-based 2D object.

        Parameters
        ----------
        func:
            A callable that accepts a coordinate array ``(x, y)`` and returns
            ``True`` if the point lies inside the object.
        lower_bound:
            Lower corner ``(x_min, y_min)`` of the bounding box.
        upper_bound:
            Upper corner ``(x_max, y_max)`` of the bounding box.
        bc:
            Optional boundary condition information.
        inverted:
            If ``True``, the object covers every point where ``func`` is falsy.
        """
        assert callable(func), "func must be callable."
        assert len(lower_bound) == 2, "lower_bound must have two coordinates."
        assert len(upper_bound) == 2, "upper_bound must have two coordinates."

        self._func = func
        self._bc = bc
        self._inverted = bool(inverted)

        super().__init__(
            lower_bound=tuple(lower_bound),
            upper_bound=tuple(upper_bound),
        )

    @property
    def func(self) -> Callable[[Array], bool]:
        """Return the Boolean membership function."""
        return self._func

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the object.

        Parameters
        ----------
        coords:
            ``(x, y)`` coordinates to test.

        Returns
        -------
        bool
            ``True`` if ``func(coords)`` is truthy (falsy when inverted).
        """
        return self._apply_inversion(bool(self._func(xp.to_numpy(coords))))

    def get_vertices(self) -> list:
        """Return an empty list; vertices cannot be derived from an arbitrary function."""
        return []


class FunctionObject3D(Shape3D):
    """Represent an arbitrary 3D geometry object defined by a Boolean function.

    The object interior is the set of all points for which ``func(coords)``
    returns ``True``.  A bounding box must be supplied by the caller because
    it cannot in general be derived from an arbitrary function.

    Example
    -------
    Create a solid torus with major radius 3 and tube radius 1::

        def torus(coords):
            x, y, z = coords
            r = (x**2 + y**2)**0.5
            return (r - 3.0)**2 + z**2 <= 1.0**2

        obj = FunctionObject3D(
            func=torus,
            lower_bound=(-4.0, -4.0, -1.0),
            upper_bound=(4.0, 4.0, 1.0),
        )
    """

    def __init__(
        self,
        func: Callable[[Array], bool],
        lower_bound: BoundsLike,
        upper_bound: BoundsLike,
        bc: ConfigValue = None,
        inverted: bool = False,
    ) -> None:
        """
        Initialize a function-based 3D object.

        Parameters
        ----------
        func:
            A callable that accepts a coordinate array ``(x, y, z)`` and
            returns ``True`` if the point lies inside the object.
        lower_bound:
            Lower corner ``(x_min, y_min, z_min)`` of the bounding box.
        upper_bound:
            Upper corner ``(x_max, y_max, z_max)`` of the bounding box.
        bc:
            Optional boundary condition information.
        inverted:
            If ``True``, the object covers every point where ``func`` is falsy.
        """
        assert callable(func), "func must be callable."
        assert len(lower_bound) == 3, "lower_bound must have three coordinates."
        assert len(upper_bound) == 3, "upper_bound must have three coordinates."

        self._func = func
        self._bc = bc
        self._inverted = bool(inverted)

        super().__init__(
            lower_bound=tuple(lower_bound),
            upper_bound=tuple(upper_bound),
        )

    @property
    def func(self) -> Callable[[Array], bool]:
        """Return the Boolean membership function."""
        return self._func

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the object.

        Parameters
        ----------
        coords:
            ``(x, y, z)`` coordinates to test.

        Returns
        -------
        bool
            ``True`` if ``func(coords)`` is truthy (falsy when inverted).
        """
        return self._apply_inversion(bool(self._func(xp.to_numpy(coords))))

    def get_vertices(
        self,
        slice_axis: int = 0,
        slice_position: float | None = None,
    ) -> tuple:
        """Return an empty array; vertices cannot be derived from an arbitrary function."""
        import cunumpy as xp

        return xp.array([]), self.inverted

    def volume(self) -> float:
        """Volume cannot be analytically derived from an arbitrary function."""
        raise NotImplementedError(
            "volume() is not available for FunctionObject3D. "
            "Use numerical integration if needed.",
        )

    def surface_area(self) -> float:
        """Surface area cannot be analytically derived from an arbitrary function."""
        raise NotImplementedError(
            "surface_area() is not available for FunctionObject3D. "
            "Use numerical methods if needed.",
        )
